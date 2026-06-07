//! Async parallel offline transcribe: VAD and ASR stages overlap across jobs.

use std::sync::Arc;

use anyhow::Result;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use vasr_data::{Timeline, Waveform};

use super::{OfflinePipeline, VadPreparation};
use crate::model::{AsrOptions, VadOptions};

const DEFAULT_STAGE_BUFFER: usize = 4;

#[derive(Clone)]
pub struct AsyncOfflinePipeline {
    inner: Arc<OfflinePipeline>,
}

#[derive(Debug, Clone)]
pub struct ParallelTranscribeOptions {
    pub asr_options: AsrOptions,
    pub vad_options: VadOptions,
    pub stage_buffer: usize,
}

impl Default for ParallelTranscribeOptions {
    fn default() -> Self {
        Self {
            asr_options: AsrOptions::default(),
            vad_options: VadOptions::default(),
            stage_buffer: DEFAULT_STAGE_BUFFER,
        }
    }
}

impl From<AsrOptions> for ParallelTranscribeOptions {
    fn from(asr_options: AsrOptions) -> Self {
        Self {
            asr_options,
            ..Self::default()
        }
    }
}

#[derive(Debug)]
struct AsrJob {
    index: usize,
    prepared: VadPreparation,
    waveform: Waveform,
}

#[derive(Debug)]
enum StageResult {
    Ready { index: usize, timeline: Timeline },
    Failed { index: usize, error: anyhow::Error },
}

impl AsyncOfflinePipeline {
    pub fn new(pipeline: OfflinePipeline) -> Self {
        Self {
            inner: Arc::new(pipeline),
        }
    }

    pub fn from_arc(pipeline: Arc<OfflinePipeline>) -> Self {
        Self { inner: pipeline }
    }

    pub fn inner(&self) -> &Arc<OfflinePipeline> {
        &self.inner
    }

    pub async fn transcribe(
        &self,
        waveform: Waveform,
        options: &ParallelTranscribeOptions,
    ) -> Result<Timeline> {
        let mut timelines = self.transcribe_many(vec![waveform], options).await?;
        timelines
            .pop()
            .ok_or_else(|| anyhow::anyhow!("async transcribe returned no timeline"))
    }

    /// Transcribe multiple waveforms with overlapping VAD and ASR stages.
    pub async fn transcribe_many(
        &self,
        waveforms: Vec<Waveform>,
        options: &ParallelTranscribeOptions,
    ) -> Result<Vec<Timeline>> {
        if waveforms.is_empty() {
            return Ok(Vec::new());
        }
        let job_count = waveforms.len();
        if job_count == 1 {
            let waveform = waveforms
                .into_iter()
                .next()
                .ok_or_else(|| anyhow::anyhow!("missing waveform"))?;
            let timeline = tokio::task::spawn_blocking({
                let pipeline = Arc::clone(&self.inner);
                let asr_options = options.asr_options.clone();
                let vad_options = options.vad_options.clone();
                move || pipeline.transcribe_with_vad_options(&waveform, &asr_options, &vad_options)
            })
            .await??;
            return Ok(vec![timeline]);
        }

        let buffer = options.stage_buffer.max(1);
        let (asr_tx, asr_rx) = mpsc::channel(buffer);
        let (result_tx, mut result_rx) = mpsc::channel(job_count);

        let asr_handle = spawn_asr_worker(
            Arc::clone(&self.inner),
            options.asr_options.clone(),
            asr_rx,
            result_tx,
        );
        let vad_handle = spawn_vad_worker(
            Arc::clone(&self.inner),
            options.vad_options.clone(),
            waveforms,
            asr_tx,
        );

        let mut indexed_results = Vec::with_capacity(job_count);
        while let Some(result) = result_rx.recv().await {
            match result {
                StageResult::Ready { index, timeline } => {
                    indexed_results.push((index, Ok(timeline)))
                }
                StageResult::Failed { index, error } => indexed_results.push((index, Err(error))),
            }
        }

        vad_handle.await??;
        asr_handle.await??;

        indexed_results.sort_unstable_by_key(|(index, _)| *index);
        indexed_results
            .into_iter()
            .map(|(_, result)| result)
            .collect()
    }
}

fn spawn_vad_worker(
    pipeline: Arc<OfflinePipeline>,
    vad_options: VadOptions,
    waveforms: Vec<Waveform>,
    asr_tx: mpsc::Sender<AsrJob>,
) -> JoinHandle<Result<()>> {
    tokio::spawn(async move {
        for (index, waveform) in waveforms.into_iter().enumerate() {
            let prepared = tokio::task::spawn_blocking({
                let pipeline = Arc::clone(&pipeline);
                let waveform = waveform.clone();
                let vad_options = vad_options.clone();
                move || pipeline.prepare_vad_stage(&waveform, &vad_options)
            })
            .await??;

            asr_tx
                .send(AsrJob {
                    index,
                    prepared,
                    waveform,
                })
                .await
                .map_err(|_| anyhow::anyhow!("ASR stage channel closed"))?;
        }
        Ok(())
    })
}

fn spawn_asr_worker(
    pipeline: Arc<OfflinePipeline>,
    asr_options: AsrOptions,
    mut asr_rx: mpsc::Receiver<AsrJob>,
    result_tx: mpsc::Sender<StageResult>,
) -> JoinHandle<Result<()>> {
    tokio::spawn(async move {
        while let Some(job) = asr_rx.recv().await {
            let index = job.index;
            let result = tokio::task::spawn_blocking({
                let pipeline = Arc::clone(&pipeline);
                let asr_options = asr_options.clone();
                move || run_asr_job(&pipeline, job, &asr_options)
            })
            .await;

            let stage_result = match result {
                Ok(Ok(stage_result)) => stage_result,
                Ok(Err(error)) => StageResult::Failed { index, error },
                Err(error) => StageResult::Failed {
                    index,
                    error: anyhow::anyhow!("ASR worker join error: {error}"),
                },
            };
            result_tx
                .send(stage_result)
                .await
                .map_err(|_| anyhow::anyhow!("result channel closed"))?;
        }
        Ok(())
    })
}

fn run_asr_job(
    pipeline: &OfflinePipeline,
    job: AsrJob,
    asr_options: &AsrOptions,
) -> Result<StageResult> {
    let mut timeline = Timeline::new("offline_audio");
    match job.prepared {
        VadPreparation::Speech(prepared) => {
            timeline
                .annotations
                .extend(pipeline.transcribe_prepared(prepared, asr_options)?);
        }
        VadPreparation::NoSpeech => {}
        VadPreparation::Disabled => {
            timeline.annotations.extend(
                pipeline
                    .asr
                    .transcribe(&job.waveform, asr_options)?
                    .annotations,
            );
        }
    }
    Ok(StageResult::Ready {
        index: job.index,
        timeline,
    })
}
