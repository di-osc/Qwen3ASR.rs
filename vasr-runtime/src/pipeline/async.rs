//! Async parallel offline transcribe: VAD and ASR stages overlap across jobs.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use anyhow::{Result, bail};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use vasr_data::{Annotation, DurationMs, Timeline, Waveform};

use super::{OfflinePipeline, VadPreparation, VadPrepared, offset_annotations};
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

#[derive(Debug, Clone)]
struct AsrSegmentJob {
    job_index: usize,
    speech_annotations: Vec<Annotation>,
    segment_index: usize,
    segment_count: usize,
    offset: DurationMs,
    waveform: Waveform,
}

#[derive(Debug)]
enum AsrPipelineMessage {
    Job(AsrJob),
    Segment(AsrSegmentJob),
}

#[derive(Debug)]
enum StageResult {
    Ready { index: usize, timeline: Timeline },
    Failed { index: usize, error: anyhow::Error },
}

#[derive(Debug)]
struct SegmentJobState {
    index: usize,
    speech_annotations: Vec<Annotation>,
    timeline: Timeline,
    segment_count: usize,
    segments_done: usize,
}

#[derive(Default)]
struct SegmentAsrState {
    jobs: HashMap<usize, SegmentJobState>,
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
        let max_batch_size = options.asr_options.max_batch_size;
        let (asr_tx, asr_rx) = mpsc::channel(buffer);
        let (result_tx, mut result_rx) = mpsc::channel(job_count);

        let asr_handle = spawn_asr_worker(
            Arc::clone(&self.inner),
            options.asr_options.clone(),
            max_batch_size,
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

fn speech_job_to_segments(job: AsrJob) -> Result<Vec<AsrSegmentJob>> {
    let VadPreparation::Speech(prepared) = job.prepared else {
        bail!("speech_job_to_segments requires Speech preparation");
    };
    let VadPrepared {
        speech_annotations,
        segments,
        slices,
    } = prepared;
    if segments.len() != slices.len() {
        bail!(
            "VAD prepared mismatch: segments={} slices={}",
            segments.len(),
            slices.len()
        );
    }
    let segment_count = segments.len();
    Ok(segments
        .into_iter()
        .zip(slices)
        .enumerate()
        .map(|(segment_index, (segment, waveform))| AsrSegmentJob {
            job_index: job.index,
            speech_annotations: if segment_index == 0 {
                speech_annotations.clone()
            } else {
                Vec::new()
            },
            segment_index,
            segment_count,
            offset: segment.range.start,
            waveform,
        })
        .collect())
}

fn spawn_vad_worker(
    pipeline: Arc<OfflinePipeline>,
    vad_options: VadOptions,
    waveforms: Vec<Waveform>,
    asr_tx: mpsc::Sender<AsrPipelineMessage>,
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

            let asr_job = AsrJob {
                index,
                prepared,
                waveform,
            };
            match asr_job.prepared {
                VadPreparation::Speech(_) => {
                    for segment in speech_job_to_segments(asr_job)? {
                        asr_tx
                            .send(AsrPipelineMessage::Segment(segment))
                            .await
                            .map_err(|_| anyhow::anyhow!("ASR stage channel closed"))?;
                    }
                }
                _ => {
                    asr_tx
                        .send(AsrPipelineMessage::Job(asr_job))
                        .await
                        .map_err(|_| anyhow::anyhow!("ASR stage channel closed"))?;
                }
            }
        }
        Ok(())
    })
}

fn spawn_asr_worker(
    pipeline: Arc<OfflinePipeline>,
    asr_options: AsrOptions,
    max_batch_size: usize,
    mut asr_rx: mpsc::Receiver<AsrPipelineMessage>,
    result_tx: mpsc::Sender<StageResult>,
) -> JoinHandle<Result<()>> {
    tokio::spawn(async move {
        let mut segment_state = SegmentAsrState::default();
        let mut pending = VecDeque::new();
        while let Some(message) = recv_asr_message(&mut pending, &mut asr_rx).await {
            match message {
                AsrPipelineMessage::Job(job) => {
                    let jobs = collect_job_microbatch(
                        job,
                        max_batch_size.max(1),
                        &mut asr_rx,
                        &mut pending,
                    )
                    .await;
                    if jobs.is_empty() {
                        continue;
                    }
                    let results = tokio::task::spawn_blocking({
                        let pipeline = Arc::clone(&pipeline);
                        let asr_options = asr_options.clone();
                        move || run_asr_jobs_batch(&pipeline, jobs, &asr_options)
                    })
                    .await
                    .unwrap_or_else(|error| {
                        vec![StageResult::Failed {
                            index: 0,
                            error: anyhow::anyhow!("ASR worker join error: {error}"),
                        }]
                    });
                    for result in results {
                        result_tx
                            .send(result)
                            .await
                            .map_err(|_| anyhow::anyhow!("result channel closed"))?;
                    }
                }
                AsrPipelineMessage::Segment(segment) => {
                    let segments = collect_segment_microbatch(
                        segment,
                        max_batch_size.max(1),
                        &mut asr_rx,
                        &mut pending,
                    )
                    .await;
                    prepare_segment_job_state(&mut segment_state, &segments);
                    let batch = tokio::task::spawn_blocking({
                        let pipeline = Arc::clone(&pipeline);
                        let asr_options = asr_options.clone();
                        let segments = segments.clone();
                        move || run_segment_asr_inference(&pipeline, &segments, &asr_options)
                    })
                    .await
                    .unwrap_or_else(|error| Err(anyhow::anyhow!("ASR worker join error: {error}")));
                    let results = finish_segment_asr_batch(&mut segment_state, &segments, batch);
                    for result in results {
                        result_tx
                            .send(result)
                            .await
                            .map_err(|_| anyhow::anyhow!("result channel closed"))?;
                    }
                }
            }
        }
        Ok(())
    })
}

async fn recv_asr_message(
    pending: &mut VecDeque<AsrPipelineMessage>,
    asr_rx: &mut mpsc::Receiver<AsrPipelineMessage>,
) -> Option<AsrPipelineMessage> {
    match pending.pop_front() {
        Some(message) => Some(message),
        None => asr_rx.recv().await,
    }
}

async fn collect_job_microbatch(
    first: AsrJob,
    max_batch_size: usize,
    asr_rx: &mut mpsc::Receiver<AsrPipelineMessage>,
    pending: &mut VecDeque<AsrPipelineMessage>,
) -> Vec<AsrJob> {
    let mut jobs = vec![first];
    if max_batch_size <= 1 {
        return jobs;
    }
    while jobs.len() < max_batch_size {
        match asr_rx.try_recv() {
            Ok(AsrPipelineMessage::Job(job)) => jobs.push(job),
            Ok(message) => {
                pending.push_front(message);
                break;
            }
            Err(mpsc::error::TryRecvError::Empty) => break,
            Err(mpsc::error::TryRecvError::Disconnected) => return jobs,
        }
    }
    if jobs.len() >= max_batch_size {
        return jobs;
    }
    while jobs.len() < max_batch_size {
        match asr_rx.recv().await {
            Some(AsrPipelineMessage::Job(job)) => jobs.push(job),
            Some(message) => {
                pending.push_front(message);
                break;
            }
            None => break,
        }
    }
    jobs
}

async fn collect_segment_microbatch(
    first: AsrSegmentJob,
    max_batch_size: usize,
    asr_rx: &mut mpsc::Receiver<AsrPipelineMessage>,
    pending: &mut VecDeque<AsrPipelineMessage>,
) -> Vec<AsrSegmentJob> {
    let mut segments = vec![first];
    if max_batch_size <= 1 {
        return segments;
    }
    while segments.len() < max_batch_size {
        match asr_rx.try_recv() {
            Ok(AsrPipelineMessage::Segment(segment)) => segments.push(segment),
            Ok(message) => {
                pending.push_front(message);
                break;
            }
            Err(mpsc::error::TryRecvError::Empty) => break,
            Err(mpsc::error::TryRecvError::Disconnected) => return segments,
        }
    }
    if segments.len() >= max_batch_size {
        return segments;
    }
    while segments.len() < max_batch_size {
        match asr_rx.recv().await {
            Some(AsrPipelineMessage::Segment(segment)) => segments.push(segment),
            Some(message) => {
                pending.push_front(message);
                break;
            }
            None => break,
        }
    }
    segments
}

fn prepare_segment_job_state(state: &mut SegmentAsrState, segments: &[AsrSegmentJob]) {
    for segment in segments {
        let entry = state
            .jobs
            .entry(segment.job_index)
            .or_insert_with(|| SegmentJobState {
                index: segment.job_index,
                speech_annotations: segment.speech_annotations.clone(),
                timeline: Timeline::new("offline_audio"),
                segment_count: segment.segment_count,
                segments_done: 0,
            });
        if !segment.speech_annotations.is_empty() {
            entry.speech_annotations = segment.speech_annotations.clone();
        }
        if entry.timeline.annotations.is_empty() && !entry.speech_annotations.is_empty() {
            entry
                .timeline
                .annotations
                .extend(entry.speech_annotations.clone());
        }
        entry.segment_count = segment.segment_count;
    }
}

fn run_segment_asr_inference(
    pipeline: &OfflinePipeline,
    segments: &[AsrSegmentJob],
    asr_options: &AsrOptions,
) -> Result<Vec<Timeline>> {
    let slices: Vec<Waveform> = segments
        .iter()
        .map(|segment| segment.waveform.clone())
        .collect();
    pipeline
        .asr
        .transcribe_batch(slices.as_slice(), asr_options)
}

fn finish_segment_asr_batch(
    state: &mut SegmentAsrState,
    segments: &[AsrSegmentJob],
    batch: Result<Vec<Timeline>>,
) -> Vec<StageResult> {
    if segments.is_empty() {
        return Vec::new();
    }

    let mut completed = Vec::new();
    match batch {
        Ok(timelines) if timelines.len() == segments.len() => {
            for (segment, asr_timeline) in segments.iter().zip(timelines) {
                if let Some(job) = state.jobs.get_mut(&segment.job_index) {
                    job.timeline
                        .annotations
                        .extend(offset_annotations(asr_timeline.annotations, segment.offset));
                    job.segments_done = job.segments_done.saturating_add(1);
                    if job.segments_done >= job.segment_count {
                        if let Some(job) = state.jobs.remove(&segment.job_index) {
                            completed.push(StageResult::Ready {
                                index: job.index,
                                timeline: job.timeline,
                            });
                        }
                    }
                }
            }
        }
        Ok(timelines) => {
            let err = anyhow::anyhow!(
                "ASR segment batch returned {} timelines for {} segments",
                timelines.len(),
                segments.len()
            );
            let affected: HashSet<usize> =
                segments.iter().map(|segment| segment.job_index).collect();
            for job_index in affected {
                if let Some(job) = state.jobs.remove(&job_index) {
                    completed.push(StageResult::Failed {
                        index: job.index,
                        error: anyhow::anyhow!("{err}"),
                    });
                }
            }
        }
        Err(error) => {
            let affected: HashSet<usize> =
                segments.iter().map(|segment| segment.job_index).collect();
            for job_index in affected {
                if let Some(job) = state.jobs.remove(&job_index) {
                    completed.push(StageResult::Failed {
                        index: job.index,
                        error: anyhow::anyhow!("{error}"),
                    });
                }
            }
        }
    }

    completed
}

fn run_asr_jobs_batch(
    pipeline: &OfflinePipeline,
    jobs: Vec<AsrJob>,
    asr_options: &AsrOptions,
) -> Vec<StageResult> {
    let mut results = Vec::with_capacity(jobs.len());
    let mut raw_jobs = Vec::new();
    let mut prepared_jobs = Vec::new();
    let mut full_speech_annotations = Vec::new();

    for job in jobs {
        match &job.prepared {
            VadPreparation::Speech(_) => prepared_jobs.push(job),
            VadPreparation::NoSpeech => results.push(StageResult::Ready {
                index: job.index,
                timeline: Timeline::new("offline_audio"),
            }),
            VadPreparation::FullSpeech { speech_annotations } => {
                full_speech_annotations.push((job.index, speech_annotations.clone()));
                raw_jobs.push(job);
            }
            VadPreparation::Disabled => raw_jobs.push(job),
        }
    }

    if !raw_jobs.is_empty() {
        let mut raw_results = run_raw_asr_jobs_batch(pipeline, raw_jobs, asr_options);
        for result in &mut raw_results {
            if let StageResult::Ready { index, timeline } = result {
                if let Some((_, annotations)) = full_speech_annotations
                    .iter()
                    .find(|(annotated_index, _)| annotated_index == index)
                {
                    let mut annotations = annotations.clone();
                    annotations.extend(std::mem::take(&mut timeline.annotations));
                    timeline.annotations = annotations;
                }
            }
        }
        results.extend(raw_results);
    }

    if !prepared_jobs.is_empty() {
        let mut segment_state = SegmentAsrState::default();
        let mut segment_jobs = Vec::new();
        for job in prepared_jobs {
            match speech_job_to_segments(job) {
                Ok(mut segments) => segment_jobs.append(&mut segments),
                Err(error) => results.push(StageResult::Failed { index: 0, error }),
            }
        }
        prepare_segment_job_state(&mut segment_state, &segment_jobs);
        let batch = run_segment_asr_inference(pipeline, &segment_jobs, asr_options);
        results.extend(finish_segment_asr_batch(
            &mut segment_state,
            &segment_jobs,
            batch,
        ));
    }

    results.sort_unstable_by_key(|result| match result {
        StageResult::Ready { index, .. } | StageResult::Failed { index, .. } => *index,
    });
    results
}

fn run_raw_asr_jobs_batch(
    pipeline: &OfflinePipeline,
    jobs: Vec<AsrJob>,
    asr_options: &AsrOptions,
) -> Vec<StageResult> {
    let indices: Vec<usize> = jobs.iter().map(|job| job.index).collect();
    let job_count = indices.len();
    let waveforms: Vec<Waveform> = jobs.into_iter().map(|job| job.waveform).collect();
    match pipeline.asr.transcribe_batch(&waveforms, asr_options) {
        Ok(timelines) if timelines.len() == indices.len() => indices
            .into_iter()
            .zip(timelines)
            .map(|(index, timeline)| StageResult::Ready { index, timeline })
            .collect(),
        Ok(timelines) => indices
            .into_iter()
            .map(|index| StageResult::Failed {
                index,
                error: anyhow::anyhow!(
                    "ASR batch returned {} timelines for {} jobs",
                    timelines.len(),
                    job_count
                ),
            })
            .collect(),
        Err(error) => indices
            .into_iter()
            .map(|index| StageResult::Failed {
                index,
                error: anyhow::anyhow!("{error}"),
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vasr_data::{
        Annotation, AnnotationPayload, AnnotationSource, AnnotationStatus, TextSpan, TimeRange,
    };

    use super::*;
    use crate::model::{AsrModel, StreamingAsrModel};

    struct BatchFakeAsr;

    impl AsrModel for BatchFakeAsr {
        fn transcribe(&self, waveform: &Waveform, options: &AsrOptions) -> Result<Timeline> {
            let mut timelines = self.transcribe_batch(std::slice::from_ref(waveform), options)?;
            timelines
                .pop()
                .ok_or_else(|| anyhow::anyhow!("fake ASR returned no timeline"))
        }

        fn transcribe_batch(
            &self,
            waveforms: &[Waveform],
            _options: &AsrOptions,
        ) -> Result<Vec<Timeline>> {
            Ok(waveforms
                .iter()
                .map(|_| {
                    let mut timeline = Timeline::new("fake_batch_asr");
                    timeline.push(Annotation::new(
                        TimeRange::default(),
                        AnnotationPayload::Transcription(TextSpan::new("ok")),
                        AnnotationSource::Model("fake_batch_asr".to_string()),
                        AnnotationStatus::Final,
                    ));
                    timeline
                })
                .collect())
        }

        fn start_stream(&self, _options: &AsrOptions) -> Result<Box<dyn StreamingAsrModel>> {
            anyhow::bail!("not implemented")
        }
    }

    #[tokio::test]
    async fn asr_worker_does_not_drop_job_after_segment_microbatch() -> Result<()> {
        let pipeline = Arc::new(OfflinePipeline {
            vad: None,
            asr: Arc::new(BatchFakeAsr),
        });
        let (asr_tx, asr_rx) = mpsc::channel(4);
        let (result_tx, mut result_rx) = mpsc::channel(4);
        let handle = spawn_asr_worker(pipeline, AsrOptions::default(), 4, asr_rx, result_tx);

        asr_tx
            .send(AsrPipelineMessage::Segment(AsrSegmentJob {
                job_index: 0,
                speech_annotations: Vec::new(),
                segment_index: 0,
                segment_count: 1,
                offset: DurationMs(0),
                waveform: Waveform::new(vec![0.1; 160], 16_000),
            }))
            .await?;
        tokio::task::yield_now().await;
        asr_tx
            .send(AsrPipelineMessage::Job(AsrJob {
                index: 1,
                prepared: VadPreparation::Disabled,
                waveform: Waveform::new(vec![0.2; 160], 16_000),
            }))
            .await?;
        drop(asr_tx);

        let mut results = Vec::new();
        while let Some(result) = result_rx.recv().await {
            results.push(result);
        }
        handle.await??;
        results.sort_unstable_by_key(|result| match result {
            StageResult::Ready { index, .. } | StageResult::Failed { index, .. } => *index,
        });

        assert_eq!(
            results
                .iter()
                .map(|result| match result {
                    StageResult::Ready { index, .. } | StageResult::Failed { index, .. } => *index,
                })
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert!(
            results
                .iter()
                .all(|result| matches!(result, StageResult::Ready { .. }))
        );
        Ok(())
    }

    #[tokio::test]
    async fn asr_worker_does_not_drop_segment_after_job_microbatch() -> Result<()> {
        let pipeline = Arc::new(OfflinePipeline {
            vad: None,
            asr: Arc::new(BatchFakeAsr),
        });
        let (asr_tx, asr_rx) = mpsc::channel(4);
        let (result_tx, mut result_rx) = mpsc::channel(4);
        let handle = spawn_asr_worker(pipeline, AsrOptions::default(), 4, asr_rx, result_tx);

        asr_tx
            .send(AsrPipelineMessage::Job(AsrJob {
                index: 0,
                prepared: VadPreparation::Disabled,
                waveform: Waveform::new(vec![0.2; 160], 16_000),
            }))
            .await?;
        tokio::task::yield_now().await;
        asr_tx
            .send(AsrPipelineMessage::Segment(AsrSegmentJob {
                job_index: 1,
                speech_annotations: Vec::new(),
                segment_index: 0,
                segment_count: 1,
                offset: DurationMs(0),
                waveform: Waveform::new(vec![0.1; 160], 16_000),
            }))
            .await?;
        drop(asr_tx);

        let mut results = Vec::new();
        while let Some(result) = result_rx.recv().await {
            results.push(result);
        }
        handle.await??;
        results.sort_unstable_by_key(|result| match result {
            StageResult::Ready { index, .. } | StageResult::Failed { index, .. } => *index,
        });

        assert_eq!(
            results
                .iter()
                .map(|result| match result {
                    StageResult::Ready { index, .. } | StageResult::Failed { index, .. } => *index,
                })
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert!(
            results
                .iter()
                .all(|result| matches!(result, StageResult::Ready { .. }))
        );
        Ok(())
    }
}
