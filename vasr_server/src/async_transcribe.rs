//! Async parallel transcribe: loader, VAD, and ASR stages overlap via channels.

use std::sync::Arc;

use anyhow::Result;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use vasr_audio::{AudioLoadOptions, AudioLoader};
use vasr_data::{AudioSource, Timeline, Waveform};
use vasr_runtime::pipeline::{OfflinePipeline, VadPrepared};
use vasr_runtime::{AsrOptions, ParallelTranscribeOptions, VadOptions};

const DEFAULT_STAGE_BUFFER: usize = 4;

#[derive(Clone)]
pub struct AsyncTranscribePipeline {
    loader: AudioLoader,
    load_options: AudioLoadOptions,
    offline: Arc<OfflinePipeline>,
    parallel_options: ParallelTranscribeOptions,
    stage_buffer: usize,
}

#[derive(Debug, Clone)]
pub struct TranscribeInput {
    pub index: usize,
    pub source: AudioSource,
}

#[derive(Debug)]
pub struct TranscribeItemOutcome {
    pub index: usize,
    pub result: Result<Timeline>,
    pub audio_seconds: f64,
    pub bad_component: Option<&'static str>,
}

impl AsyncTranscribePipeline {
    pub fn new(
        loader: AudioLoader,
        offline: Arc<OfflinePipeline>,
        asr_options: AsrOptions,
    ) -> Self {
        Self {
            loader,
            load_options: AudioLoadOptions::default(),
            offline,
            parallel_options: ParallelTranscribeOptions::from(asr_options),
            stage_buffer: DEFAULT_STAGE_BUFFER,
        }
    }

    pub fn with_stage_buffer(mut self, stage_buffer: usize) -> Self {
        self.stage_buffer = stage_buffer.max(1);
        self
    }

    pub async fn transcribe_source(
        &self,
        index: usize,
        source: AudioSource,
    ) -> TranscribeItemOutcome {
        let outcomes = self
            .transcribe_many(vec![TranscribeInput { index, source }])
            .await;
        outcomes
            .into_iter()
            .next()
            .unwrap_or(TranscribeItemOutcome {
                index,
                result: Err(anyhow::anyhow!("transcribe returned no outcome")),
                audio_seconds: 0.0,
                bad_component: Some("recognizer"),
            })
    }

    pub async fn transcribe_many(
        &self,
        inputs: Vec<TranscribeInput>,
    ) -> Vec<TranscribeItemOutcome> {
        if inputs.is_empty() {
            return Vec::new();
        }
        if inputs.len() == 1 {
            let input = inputs
                .into_iter()
                .next()
                .expect("checked single input above");
            return vec![self.transcribe_single(input).await];
        }

        let job_count = inputs.len();
        let buffer = self.stage_buffer;
        let (loaded_tx, loaded_rx) = mpsc::channel(buffer);
        let (asr_tx, asr_rx) = mpsc::channel(buffer);
        let (result_tx, mut result_rx) = mpsc::channel(job_count);

        let loader = self.loader.clone();
        let load_options = self.load_options.clone();
        let loader_handle = spawn_loader_worker(inputs, loader, load_options, loaded_tx);

        let offline = Arc::clone(&self.offline);
        let vad_options = self.parallel_options.vad_options.clone();
        let vad_handle = spawn_vad_worker(offline, vad_options, loaded_rx, asr_tx);

        let offline = Arc::clone(&self.offline);
        let asr_options = self.parallel_options.asr_options.clone();
        let asr_handle = spawn_asr_worker(offline, asr_options, asr_rx, result_tx);

        let mut outcomes = Vec::with_capacity(job_count);
        while let Some(outcome) = result_rx.recv().await {
            outcomes.push(outcome);
        }

        let _ = loader_handle.await;
        let _ = vad_handle.await;
        let _ = asr_handle.await;

        outcomes.sort_unstable_by_key(|outcome| outcome.index);
        outcomes
    }

    async fn transcribe_single(&self, input: TranscribeInput) -> TranscribeItemOutcome {
        let loaded = match tokio::task::spawn_blocking({
            let loader = self.loader.clone();
            let load_options = self.load_options.clone();
            let source = input.source.clone();
            move || loader.load(&source, &load_options)
        })
        .await
        {
            Ok(result) => result,
            Err(error) => {
                return TranscribeItemOutcome {
                    index: input.index,
                    result: Err(anyhow::anyhow!("loader join error: {error}")),
                    audio_seconds: 0.0,
                    bad_component: Some("loader"),
                };
            }
        };

        let waveform = match loaded {
            Ok(waveform) => waveform,
            Err(error) => {
                return TranscribeItemOutcome {
                    index: input.index,
                    result: Err(error),
                    audio_seconds: 0.0,
                    bad_component: Some("loader"),
                };
            }
        };

        let audio_seconds = waveform.duration_seconds();
        let result = tokio::task::spawn_blocking({
            let offline = Arc::clone(&self.offline);
            let asr_options = self.parallel_options.asr_options.clone();
            let vad_options = self.parallel_options.vad_options.clone();
            move || offline.transcribe_with_vad_options(&waveform, &asr_options, &vad_options)
        })
        .await
        .unwrap_or_else(|error| Err(anyhow::anyhow!("transcribe join error: {error}")));

        let bad_component = if result.is_err() {
            Some("recognizer")
        } else {
            None
        };
        TranscribeItemOutcome {
            index: input.index,
            result,
            audio_seconds,
            bad_component,
        }
    }
}

#[derive(Debug)]
struct LoadedJob {
    index: usize,
    waveform: Waveform,
}

#[derive(Debug)]
struct AsrJob {
    index: usize,
    prepared: Option<VadPrepared>,
    waveform: Waveform,
}

fn spawn_loader_worker(
    inputs: Vec<TranscribeInput>,
    loader: AudioLoader,
    load_options: AudioLoadOptions,
    loaded_tx: mpsc::Sender<Result<LoadedJob, TranscribeItemOutcome>>,
) -> JoinHandle<Result<()>> {
    tokio::spawn(async move {
        for input in inputs {
            let loaded = tokio::task::spawn_blocking({
                let loader = loader.clone();
                let load_options = load_options.clone();
                let source = input.source.clone();
                move || loader.load(&source, &load_options)
            })
            .await;

            let message = match loaded {
                Ok(Ok(waveform)) => Ok(LoadedJob {
                    index: input.index,
                    waveform,
                }),
                Ok(Err(error)) => Err(TranscribeItemOutcome {
                    index: input.index,
                    result: Err(error),
                    audio_seconds: 0.0,
                    bad_component: Some("loader"),
                }),
                Err(error) => Err(TranscribeItemOutcome {
                    index: input.index,
                    result: Err(anyhow::anyhow!("loader join error: {error}")),
                    audio_seconds: 0.0,
                    bad_component: Some("loader"),
                }),
            };

            loaded_tx
                .send(message)
                .await
                .map_err(|_| anyhow::anyhow!("loaded channel closed"))?;
        }
        Ok(())
    })
}

fn spawn_vad_worker(
    offline: Arc<OfflinePipeline>,
    vad_options: VadOptions,
    mut loaded_rx: mpsc::Receiver<Result<LoadedJob, TranscribeItemOutcome>>,
    asr_tx: mpsc::Sender<Result<AsrJob, TranscribeItemOutcome>>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        while let Some(message) = loaded_rx.recv().await {
            match message {
                Ok(job) => {
                    let prepared = tokio::task::spawn_blocking({
                        let offline = Arc::clone(&offline);
                        let waveform = job.waveform.clone();
                        let vad_options = vad_options.clone();
                        move || offline.prepare_vad(&waveform, &vad_options)
                    })
                    .await;

                    let asr_message = match prepared {
                        Ok(Ok(prepared)) => Ok(AsrJob {
                            index: job.index,
                            prepared,
                            waveform: job.waveform,
                        }),
                        Ok(Err(error)) => Err(TranscribeItemOutcome {
                            index: job.index,
                            result: Err(error),
                            audio_seconds: job.waveform.duration_seconds(),
                            bad_component: Some("vad"),
                        }),
                        Err(error) => Err(TranscribeItemOutcome {
                            index: job.index,
                            result: Err(anyhow::anyhow!("VAD join error: {error}")),
                            audio_seconds: job.waveform.duration_seconds(),
                            bad_component: Some("vad"),
                        }),
                    };
                    let _ = asr_tx.send(asr_message).await;
                }
                Err(outcome) => {
                    let _ = asr_tx.send(Err(outcome)).await;
                }
            }
        }
    })
}

fn spawn_asr_worker(
    offline: Arc<OfflinePipeline>,
    asr_options: AsrOptions,
    mut asr_rx: mpsc::Receiver<Result<AsrJob, TranscribeItemOutcome>>,
    result_tx: mpsc::Sender<TranscribeItemOutcome>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        while let Some(message) = asr_rx.recv().await {
            let outcome = match message {
                Ok(job) => {
                    let index = job.index;
                    let audio_seconds = job.waveform.duration_seconds();
                    let result = tokio::task::spawn_blocking({
                        let offline = Arc::clone(&offline);
                        let asr_options = asr_options.clone();
                        move || run_asr_job(&offline, job, &asr_options)
                    })
                    .await
                    .unwrap_or_else(|error| Err(anyhow::anyhow!("ASR join error: {error}")));
                    let bad_component = if result.is_err() {
                        Some("recognizer")
                    } else {
                        None
                    };
                    TranscribeItemOutcome {
                        index,
                        result,
                        audio_seconds,
                        bad_component,
                    }
                }
                Err(outcome) => outcome,
            };
            let _ = result_tx.send(outcome).await;
        }
    })
}

fn run_asr_job(
    offline: &OfflinePipeline,
    job: AsrJob,
    asr_options: &AsrOptions,
) -> Result<Timeline> {
    let mut timeline = Timeline::new("offline_audio");
    match job.prepared {
        Some(prepared) => {
            timeline
                .annotations
                .extend(offline.transcribe_prepared(prepared, asr_options)?);
        }
        None => {
            timeline.annotations.extend(
                offline
                    .asr
                    .transcribe(&job.waveform, asr_options)?
                    .annotations,
            );
        }
    }
    Ok(timeline)
}
