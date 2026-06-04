use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Result;
use vasr_data::{
    Annotation, AnnotationPayload, AnnotationSource, AnnotationStatus, TextSegment, TimeRange,
    Timeline, Waveform,
};
use vasr_runtime::{
    AsrModel, AsrOptions, AsyncOfflinePipeline, OfflinePipeline, ParallelTranscribeOptions,
    StreamingAsrModel, VadModel, VadOptions, VadSegment,
};

#[derive(Default)]
struct SlowFakeAsr {
    seen: Mutex<Vec<usize>>,
}

impl AsrModel for SlowFakeAsr {
    fn transcribe(&self, waveform: &Waveform, _options: &AsrOptions) -> Result<Timeline> {
        std::thread::sleep(Duration::from_millis(30));
        self.seen
            .lock()
            .expect("seen poisoned")
            .push(waveform.samples.len());
        let mut timeline = Timeline::new("slow_asr");
        timeline.push(Annotation::new(
            TimeRange::default(),
            AnnotationPayload::Segment(TextSegment::new("ok")),
            AnnotationSource::Model("slow_asr".to_string()),
            AnnotationStatus::Final,
        ));
        Ok(timeline)
    }

    fn start_stream(&self, _options: &AsrOptions) -> Result<Box<dyn StreamingAsrModel>> {
        anyhow::bail!("not implemented")
    }
}

struct FakeVad;

impl VadModel for FakeVad {
    fn detect(&self, waveform: &Waveform, _options: &VadOptions) -> Result<Vec<VadSegment>> {
        std::thread::sleep(Duration::from_millis(20));
        Ok(vec![VadSegment {
            range: TimeRange::new(
                vasr_data::DurationMs(0),
                vasr_data::DurationMs(waveform.duration_ms().round() as u64),
            ),
            probability: 0.9,
        }])
    }

    fn start_stream(
        &self,
        _options: &VadOptions,
    ) -> Result<Box<dyn vasr_runtime::StreamingVadModel>> {
        anyhow::bail!("not implemented")
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn async_pipeline_overlaps_vad_and_asr_across_jobs() -> Result<()> {
    let pipeline = AsyncOfflinePipeline::new(OfflinePipeline {
        vad: Some(Arc::new(FakeVad)),
        asr: Arc::new(SlowFakeAsr::default()),
    });
    let waveforms = vec![
        Waveform::new(vec![0.1; 8_000], 16_000),
        Waveform::new(vec![0.2; 8_000], 16_000),
        Waveform::new(vec![0.3; 8_000], 16_000),
    ];
    let options = ParallelTranscribeOptions::default();

    let start = Instant::now();
    let timelines = pipeline.transcribe_many(waveforms, &options).await?;
    let elapsed = start.elapsed();

    assert_eq!(timelines.len(), 3);
    assert!(
        elapsed < Duration::from_millis(170),
        "expected overlap (<170ms), got {:?}",
        elapsed
    );
    Ok(())
}
