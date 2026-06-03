use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::future::Future;
use std::sync::{
    Arc, Condvar, Mutex,
    atomic::{AtomicU64, Ordering as AtomicOrdering},
};
use std::thread;

use anyhow::{Result, anyhow};
use tokio::sync::oneshot;
use vasr_data::{Timeline, Waveform};
use vasr_runtime::{AsrModel, AsrOptions, StreamingAsrModel};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferencePriority {
    Realtime,
    Transcribe,
}

impl InferencePriority {
    fn value(self) -> u8 {
        match self {
            Self::Realtime => 0,
            Self::Transcribe => 10,
        }
    }
}

#[derive(Clone)]
pub struct InferenceScheduler {
    inner: Arc<SchedulerInner>,
}

struct SchedulerInner {
    label: String,
    queue: Mutex<BinaryHeap<QueuedTask>>,
    available: Condvar,
    sequence: AtomicU64,
}

struct QueuedTask {
    priority: InferencePriority,
    sequence: u64,
    label: String,
    run: Box<dyn FnOnce() + Send + 'static>,
}

impl InferenceScheduler {
    pub fn start(label: impl Into<String>) -> Self {
        let scheduler = Self {
            inner: Arc::new(SchedulerInner {
                label: label.into(),
                queue: Mutex::new(BinaryHeap::new()),
                available: Condvar::new(),
                sequence: AtomicU64::new(0),
            }),
        };
        scheduler.spawn_worker();
        scheduler
    }

    pub fn submit<T, F>(
        &self,
        label: impl Into<String>,
        priority: InferencePriority,
        task: F,
    ) -> impl Future<Output = Result<T>> + Send + 'static
    where
        T: Send + 'static,
        F: FnOnce() -> Result<T> + Send + 'static,
    {
        let label = label.into();
        let (tx, rx) = oneshot::channel();
        let queued = QueuedTask {
            priority,
            sequence: self.inner.sequence.fetch_add(1, AtomicOrdering::Relaxed),
            label,
            run: Box::new(move || {
                let _ = tx.send(task());
            }),
        };

        {
            let mut queue = self.inner.queue.lock().expect("inference queue poisoned");
            queue.push(queued);
        }
        self.inner.available.notify_one();

        async move {
            rx.await
                .map_err(|_| anyhow!("inference scheduler worker stopped"))?
        }
    }

    pub fn submit_blocking<T, F>(
        &self,
        label: impl Into<String>,
        priority: InferencePriority,
        task: F,
    ) -> Result<T>
    where
        T: Send + 'static,
        F: FnOnce() -> Result<T> + Send + 'static,
    {
        let label = label.into();
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        let queued = QueuedTask {
            priority,
            sequence: self.inner.sequence.fetch_add(1, AtomicOrdering::Relaxed),
            label,
            run: Box::new(move || {
                let _ = tx.send(task());
            }),
        };

        {
            let mut queue = self.inner.queue.lock().expect("inference queue poisoned");
            queue.push(queued);
        }
        self.inner.available.notify_one();

        rx.recv()
            .map_err(|_| anyhow!("inference scheduler worker stopped"))?
    }

    fn spawn_worker(&self) {
        let inner = Arc::clone(&self.inner);
        thread::Builder::new()
            .name(format!("{}-inference-worker", inner.label))
            .spawn(move || worker_loop(inner))
            .expect("failed to spawn inference scheduler worker");
    }
}

fn worker_loop(inner: Arc<SchedulerInner>) {
    loop {
        let task = {
            let mut queue = inner.queue.lock().expect("inference queue poisoned");
            loop {
                if let Some(task) = queue.pop() {
                    break task;
                }
                queue = inner
                    .available
                    .wait(queue)
                    .expect("inference queue poisoned");
            }
        };
        let _label = task.label;
        (task.run)();
    }
}

impl Ord for QueuedTask {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .priority
            .value()
            .cmp(&self.priority.value())
            .then_with(|| other.sequence.cmp(&self.sequence))
    }
}

impl PartialOrd for QueuedTask {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for QueuedTask {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.sequence == other.sequence
    }
}

impl Eq for QueuedTask {}

pub struct ScheduledAsrModel {
    inner: Arc<dyn AsrModel>,
    scheduler: InferenceScheduler,
}

impl ScheduledAsrModel {
    pub fn new(inner: Arc<dyn AsrModel>, scheduler: InferenceScheduler) -> Self {
        Self { inner, scheduler }
    }
}

impl AsrModel for ScheduledAsrModel {
    fn transcribe(&self, waveform: &Waveform, options: &AsrOptions) -> Result<Timeline> {
        let inner = Arc::clone(&self.inner);
        let waveform = waveform.clone();
        let options = options.clone();
        self.scheduler
            .submit_blocking("asr.transcribe", InferencePriority::Transcribe, move || {
                inner.transcribe(&waveform, &options)
            })
    }

    fn transcribe_batch(
        &self,
        waveforms: &[Waveform],
        options: &AsrOptions,
    ) -> Result<Vec<Timeline>> {
        let inner = Arc::clone(&self.inner);
        let waveforms = waveforms.to_vec();
        let options = options.clone();
        self.scheduler.submit_blocking(
            "asr.transcribe_batch",
            InferencePriority::Transcribe,
            move || inner.transcribe_batch(&waveforms, &options),
        )
    }

    fn start_stream(&self, options: &AsrOptions) -> Result<Box<dyn StreamingAsrModel>> {
        self.inner.start_stream(options)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use std::thread;
    use std::time::Duration;

    use anyhow::Result;
    use vasr_data::{Timeline, Waveform};
    use vasr_runtime::{AsrModel, AsrOptions, StreamingAsrModel};

    use super::{InferencePriority, InferenceScheduler, ScheduledAsrModel};

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn scheduler_serializes_concurrent_submissions() {
        let scheduler = InferenceScheduler::start("test");
        let active = Arc::new(AtomicUsize::new(0));
        let max_active = Arc::new(AtomicUsize::new(0));

        let make_task = |value: usize| {
            let active = Arc::clone(&active);
            let max_active = Arc::clone(&max_active);
            move || {
                let now = active.fetch_add(1, Ordering::SeqCst) + 1;
                max_active.fetch_max(now, Ordering::SeqCst);
                thread::sleep(Duration::from_millis(20));
                active.fetch_sub(1, Ordering::SeqCst);
                Ok(value)
            }
        };

        let first = scheduler.submit("first", InferencePriority::Transcribe, make_task(1));
        let second = scheduler.submit("second", InferencePriority::Transcribe, make_task(2));

        assert_eq!(first.await.expect("first task"), 1);
        assert_eq!(second.await.expect("second task"), 2);
        assert_eq!(max_active.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn scheduler_runs_realtime_before_queued_transcribe_work() {
        let scheduler = InferenceScheduler::start("priority-test");

        let first = scheduler.submit("first", InferencePriority::Transcribe, || {
            thread::sleep(Duration::from_millis(20));
            Ok("first")
        });
        let transcribe = scheduler.submit("transcribe", InferencePriority::Transcribe, || {
            Ok("transcribe")
        });
        let realtime = scheduler.submit("realtime", InferencePriority::Realtime, || Ok("realtime"));

        assert_eq!(first.await.expect("first task"), "first");
        assert_eq!(realtime.await.expect("realtime task"), "realtime");
        assert_eq!(transcribe.await.expect("transcribe task"), "transcribe");
    }

    #[test]
    fn scheduled_asr_model_preserves_batch_calls() {
        struct BatchOnlyAsr {
            batch_calls: Arc<AtomicUsize>,
            single_calls: Arc<AtomicUsize>,
        }

        impl AsrModel for BatchOnlyAsr {
            fn transcribe(&self, _waveform: &Waveform, _options: &AsrOptions) -> Result<Timeline> {
                self.single_calls.fetch_add(1, Ordering::SeqCst);
                Ok(Timeline::new("single"))
            }

            fn transcribe_batch(
                &self,
                waveforms: &[Waveform],
                _options: &AsrOptions,
            ) -> Result<Vec<Timeline>> {
                self.batch_calls.fetch_add(1, Ordering::SeqCst);
                Ok(waveforms.iter().map(|_| Timeline::new("batch")).collect())
            }

            fn start_stream(&self, _options: &AsrOptions) -> Result<Box<dyn StreamingAsrModel>> {
                unimplemented!("streaming is not used in this test")
            }
        }

        let batch_calls = Arc::new(AtomicUsize::new(0));
        let single_calls = Arc::new(AtomicUsize::new(0));
        let model = ScheduledAsrModel::new(
            Arc::new(BatchOnlyAsr {
                batch_calls: Arc::clone(&batch_calls),
                single_calls: Arc::clone(&single_calls),
            }),
            InferenceScheduler::start("batch-test"),
        );
        let waveforms = vec![
            Waveform::new(vec![0.0; 16_000], 16_000),
            Waveform::new(vec![0.0; 8_000], 16_000),
        ];

        let timelines = model
            .transcribe_batch(&waveforms, &AsrOptions::default())
            .expect("batch transcribe");

        assert_eq!(timelines.len(), 2);
        assert_eq!(batch_calls.load(Ordering::SeqCst), 1);
        assert_eq!(single_calls.load(Ordering::SeqCst), 0);
    }
}
