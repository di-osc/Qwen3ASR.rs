//! Async parallel transcribe: loader, VAD, and ASR stages overlap via channels.

use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Result, bail};
use tokio::sync::mpsc;
use tokio::task::{JoinHandle, JoinSet};
use vasr_audio::{AudioLoadOptions, AudioLoader};
use vasr_data::{Annotation, AudioSource, DurationMs, Timeline, Waveform};
use vasr_runtime::pipeline::{OfflinePipeline, VadPreparation, VadPrepared, offset_annotations};
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

    pub fn with_vad_options(mut self, vad_options: VadOptions) -> Self {
        self.parallel_options.vad_options = vad_options;
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
        let pipeline_start = Instant::now();
        let job_count = inputs.len();
        let DeduplicatedTranscribeInputs {
            unique_inputs,
            duplicate_indices,
        } = deduplicate_transcribe_inputs(inputs);
        if unique_inputs.len() == 1 {
            let input = unique_inputs
                .into_iter()
                .next()
                .expect("checked single input above");
            let outcomes = expand_duplicate_outcomes(
                vec![self.transcribe_single(input).await],
                &duplicate_indices,
            );
            log_pipeline(job_count, &outcomes, pipeline_start.elapsed().as_secs_f64());
            return outcomes;
        }

        let buffer = self.stage_buffer;
        let (loaded_tx, loaded_rx) = mpsc::channel(buffer);
        let (asr_tx, asr_rx) = mpsc::channel(buffer);
        let (result_tx, mut result_rx) = mpsc::channel(unique_inputs.len());

        let loader = self.loader.clone();
        let load_options = self.load_options.clone();
        let loader_handle =
            spawn_loader_worker(unique_inputs, loader, load_options, buffer, loaded_tx);

        let offline = Arc::clone(&self.offline);
        let vad_options = self.parallel_options.vad_options.clone();
        let vad_handle = spawn_vad_worker(offline, vad_options, buffer, loaded_rx, asr_tx);

        let offline = Arc::clone(&self.offline);
        let asr_options = self.parallel_options.asr_options.clone();
        let asr_handle = spawn_asr_worker(offline, asr_options, buffer, asr_rx, result_tx);

        let mut outcomes = Vec::with_capacity(job_count);
        while let Some(outcome) = result_rx.recv().await {
            outcomes.push(outcome);
        }

        let _ = loader_handle.await;
        let _ = vad_handle.await;
        let _ = asr_handle.await;

        let outcomes = expand_duplicate_outcomes(outcomes, &duplicate_indices);
        log_pipeline(job_count, &outcomes, pipeline_start.elapsed().as_secs_f64());
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
struct DeduplicatedTranscribeInputs {
    unique_inputs: Vec<TranscribeInput>,
    duplicate_indices: Vec<(usize, usize)>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum AudioSourceDedupKey {
    Path(PathBuf),
    Url(String),
    Base64(String),
    Bytes(Vec<u8>),
    LocalFileContent { len: u64, hash: u64 },
}

fn deduplicate_transcribe_inputs(inputs: Vec<TranscribeInput>) -> DeduplicatedTranscribeInputs {
    let mut seen = HashMap::with_capacity(inputs.len());
    let mut unique_inputs = Vec::with_capacity(inputs.len());
    let mut duplicate_indices = Vec::new();

    for input in inputs {
        if let Some(key) = audio_source_dedup_key(&input.source) {
            if let Some(&canonical_index) = seen.get(&key) {
                duplicate_indices.push((input.index, canonical_index));
                continue;
            }
            seen.insert(key, input.index);
        }
        unique_inputs.push(input);
    }

    DeduplicatedTranscribeInputs {
        unique_inputs,
        duplicate_indices,
    }
}

fn audio_source_dedup_key(source: &AudioSource) -> Option<AudioSourceDedupKey> {
    match source {
        AudioSource::Path(path) => Some(
            local_file_content_key(path).unwrap_or_else(|| AudioSourceDedupKey::Path(path.clone())),
        ),
        AudioSource::Url(url) => {
            if let Some(path) = local_path_from_url(url) {
                return Some(
                    local_file_content_key(&path)
                        .unwrap_or_else(|| AudioSourceDedupKey::Url(url.clone())),
                );
            }
            Some(AudioSourceDedupKey::Url(url.clone()))
        }
        AudioSource::Base64(encoded) => Some(AudioSourceDedupKey::Base64(encoded.clone())),
        AudioSource::Bytes(bytes) => Some(AudioSourceDedupKey::Bytes(bytes.clone())),
        AudioSource::Waveform(_) => None,
    }
}

fn local_file_content_key(path: &Path) -> Option<AudioSourceDedupKey> {
    let bytes = fs::read(path).ok()?;
    Some(bytes_content_key(&bytes))
}

fn bytes_content_key(bytes: &[u8]) -> AudioSourceDedupKey {
    let mut hasher = DefaultHasher::new();
    bytes.hash(&mut hasher);
    AudioSourceDedupKey::LocalFileContent {
        len: bytes.len() as u64,
        hash: hasher.finish(),
    }
}

fn local_path_from_url(value: &str) -> Option<PathBuf> {
    value
        .strip_prefix("file://")
        .map(|rest| PathBuf::from(percent_decode_path(rest)))
}

fn percent_decode_path(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let (Some(hi), Some(lo)) = (hex_value(bytes[i + 1]), hex_value(bytes[i + 2])) {
                out.push((hi << 4) | lo);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn expand_duplicate_outcomes(
    mut outcomes: Vec<TranscribeItemOutcome>,
    duplicate_indices: &[(usize, usize)],
) -> Vec<TranscribeItemOutcome> {
    if duplicate_indices.is_empty() {
        outcomes.sort_unstable_by_key(|outcome| outcome.index);
        return outcomes;
    }

    let duplicate_outcomes = duplicate_indices
        .iter()
        .filter_map(|&(duplicate_index, canonical_index)| {
            outcomes
                .iter()
                .find(|outcome| outcome.index == canonical_index)
                .map(|outcome| duplicate_outcome_from(outcome, duplicate_index))
        })
        .collect::<Vec<_>>();
    outcomes.extend(duplicate_outcomes);
    outcomes.sort_unstable_by_key(|outcome| outcome.index);
    outcomes
}

fn duplicate_outcome_from(
    outcome: &TranscribeItemOutcome,
    duplicate_index: usize,
) -> TranscribeItemOutcome {
    let result = match &outcome.result {
        Ok(timeline) => Ok(timeline.clone()),
        Err(error) => Err(anyhow::anyhow!("{}", error)),
    };
    TranscribeItemOutcome {
        index: duplicate_index,
        result,
        audio_seconds: outcome.audio_seconds,
        bad_component: outcome.bad_component,
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
    prepared: VadPreparation,
    waveform: Waveform,
}

/// One VAD speech segment enqueued for ASR continuous micro-batching.
#[derive(Debug, Clone)]
struct AsrSegmentJob {
    job_index: usize,
    job_audio_seconds: f64,
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
struct SegmentJobState {
    index: usize,
    audio_seconds: f64,
    speech_annotations: Vec<Annotation>,
    timeline: Timeline,
    segment_count: usize,
    segments_done: usize,
}

#[derive(Default)]
struct SegmentAsrState {
    jobs: HashMap<usize, SegmentJobState>,
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
            job_audio_seconds: job.waveform.duration_seconds(),
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

fn spawn_loader_worker(
    inputs: Vec<TranscribeInput>,
    loader: AudioLoader,
    load_options: AudioLoadOptions,
    max_in_flight: usize,
    loaded_tx: mpsc::Sender<Result<LoadedJob, TranscribeItemOutcome>>,
) -> JoinHandle<Result<()>> {
    tokio::spawn(async move {
        let max_in_flight = max_in_flight.max(1);
        let mut tasks = JoinSet::new();
        for input in inputs {
            spawn_loader_task(&mut tasks, loader.clone(), load_options.clone(), input);
            if tasks.len() >= max_in_flight {
                send_next_loaded(&mut tasks, &loaded_tx).await?;
            }
        }
        while !tasks.is_empty() {
            send_next_loaded(&mut tasks, &loaded_tx).await?;
        }
        Ok(())
    })
}

fn spawn_loader_task(
    tasks: &mut JoinSet<Result<LoadedJob, TranscribeItemOutcome>>,
    loader: AudioLoader,
    load_options: AudioLoadOptions,
    input: TranscribeInput,
) {
    tasks.spawn(async move {
        let stage_start = Instant::now();
        let loaded = tokio::task::spawn_blocking({
            let source = input.source.clone();
            move || loader.load(&source, &load_options)
        })
        .await;

        match loaded {
            Ok(Ok(waveform)) => {
                log_loader(
                    input.index,
                    waveform.duration_seconds(),
                    stage_start.elapsed().as_secs_f64(),
                );
                Ok(LoadedJob {
                    index: input.index,
                    waveform,
                })
            }
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
        }
    });
}

async fn send_next_loaded(
    tasks: &mut JoinSet<Result<LoadedJob, TranscribeItemOutcome>>,
    loaded_tx: &mpsc::Sender<Result<LoadedJob, TranscribeItemOutcome>>,
) -> Result<()> {
    let message = match tasks.join_next().await {
        Some(Ok(message)) => message,
        Some(Err(error)) => Err(TranscribeItemOutcome {
            index: 0,
            result: Err(anyhow::anyhow!("loader task join error: {error}")),
            audio_seconds: 0.0,
            bad_component: Some("loader"),
        }),
        None => return Ok(()),
    };
    loaded_tx
        .send(message)
        .await
        .map_err(|_| anyhow::anyhow!("loaded channel closed"))
}

fn spawn_vad_worker(
    offline: Arc<OfflinePipeline>,
    vad_options: VadOptions,
    max_in_flight: usize,
    mut loaded_rx: mpsc::Receiver<Result<LoadedJob, TranscribeItemOutcome>>,
    asr_tx: mpsc::Sender<Result<AsrPipelineMessage, TranscribeItemOutcome>>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let max_in_flight = max_in_flight.max(1);
        let mut tasks = JoinSet::new();
        while let Some(message) = loaded_rx.recv().await {
            match message {
                Ok(job) => {
                    spawn_vad_task(
                        &mut tasks,
                        Arc::clone(&offline),
                        vad_options.clone(),
                        asr_tx.clone(),
                        job,
                    );
                    if tasks.len() >= max_in_flight {
                        let _ = send_next_vad_task(&mut tasks).await;
                    }
                }
                Err(outcome) => {
                    let _ = asr_tx.send(Err(outcome)).await;
                }
            }
        }
        while !tasks.is_empty() {
            let _ = send_next_vad_task(&mut tasks).await;
        }
    })
}

fn spawn_vad_task(
    tasks: &mut JoinSet<Result<(), TranscribeItemOutcome>>,
    offline: Arc<OfflinePipeline>,
    vad_options: VadOptions,
    asr_tx: mpsc::Sender<Result<AsrPipelineMessage, TranscribeItemOutcome>>,
    job: LoadedJob,
) {
    tasks.spawn(async move {
        let stage_start = Instant::now();
        let job_index = job.index;
        let audio_seconds = job.waveform.duration_seconds();
        let prepared = tokio::task::spawn_blocking({
            let offline = Arc::clone(&offline);
            let waveform = job.waveform.clone();
            let vad_options = vad_options.clone();
            move || offline.prepare_vad_stage(&waveform, &vad_options)
        })
        .await;

        let send_message = |message: Result<AsrPipelineMessage, TranscribeItemOutcome>| {
            let asr_tx = asr_tx.clone();
            async move {
                asr_tx
                    .send(message)
                    .await
                    .map_err(|_| anyhow::anyhow!("ASR stage channel closed"))
            }
        };

        match prepared {
            Ok(Ok(prepared)) => {
                let (segments, speech_seconds) = vad_preparation_stats(&prepared);
                log_vad(
                    job_index,
                    audio_seconds,
                    segments,
                    speech_seconds,
                    stage_start.elapsed().as_secs_f64(),
                );
                let asr_job = AsrJob {
                    index: job_index,
                    prepared,
                    waveform: job.waveform,
                };
                match asr_job.prepared {
                    VadPreparation::Speech(_) => {
                        let segments = match speech_job_to_segments(asr_job) {
                            Ok(segments) => segments,
                            Err(error) => {
                                send_message(Err(TranscribeItemOutcome {
                                    index: job_index,
                                    result: Err(error),
                                    audio_seconds,
                                    bad_component: Some("vad"),
                                }))
                                .await
                                .map_err(|error| {
                                    TranscribeItemOutcome {
                                        index: job_index,
                                        result: Err(error),
                                        audio_seconds,
                                        bad_component: Some("vad"),
                                    }
                                })?;
                                return Ok(());
                            }
                        };
                        for segment in segments {
                            send_message(Ok(AsrPipelineMessage::Segment(segment)))
                                .await
                                .map_err(|error| TranscribeItemOutcome {
                                    index: job_index,
                                    result: Err(error),
                                    audio_seconds,
                                    bad_component: Some("asr"),
                                })?;
                        }
                    }
                    _ => {
                        send_message(Ok(AsrPipelineMessage::Job(asr_job)))
                            .await
                            .map_err(|error| TranscribeItemOutcome {
                                index: job_index,
                                result: Err(error),
                                audio_seconds,
                                bad_component: Some("asr"),
                            })?;
                    }
                }
                Ok(())
            }
            Ok(Err(error)) => {
                send_message(Err(TranscribeItemOutcome {
                    index: job_index,
                    result: Err(error),
                    audio_seconds,
                    bad_component: Some("vad"),
                }))
                .await
                .map_err(|error| TranscribeItemOutcome {
                    index: job_index,
                    result: Err(error),
                    audio_seconds,
                    bad_component: Some("vad"),
                })?;
                Ok(())
            }
            Err(error) => {
                send_message(Err(TranscribeItemOutcome {
                    index: job_index,
                    result: Err(anyhow::anyhow!("VAD join error: {error}")),
                    audio_seconds,
                    bad_component: Some("vad"),
                }))
                .await
                .map_err(|error| TranscribeItemOutcome {
                    index: job_index,
                    result: Err(error),
                    audio_seconds,
                    bad_component: Some("vad"),
                })?;
                Ok(())
            }
        }
    });
}

async fn send_next_vad_task(
    tasks: &mut JoinSet<Result<(), TranscribeItemOutcome>>,
) -> Result<(), TranscribeItemOutcome> {
    match tasks.join_next().await {
        Some(Ok(result)) => result,
        Some(Err(error)) => Err(TranscribeItemOutcome {
            index: 0,
            result: Err(anyhow::anyhow!("VAD task join error: {error}")),
            audio_seconds: 0.0,
            bad_component: Some("vad"),
        }),
        None => Ok(()),
    }
}

fn spawn_asr_worker(
    offline: Arc<OfflinePipeline>,
    asr_options: AsrOptions,
    max_batch_size: usize,
    mut asr_rx: mpsc::Receiver<Result<AsrPipelineMessage, TranscribeItemOutcome>>,
    result_tx: mpsc::Sender<TranscribeItemOutcome>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut segment_state = SegmentAsrState::default();
        while let Some(message) = asr_rx.recv().await {
            match message {
                Err(outcome) => {
                    let _ = result_tx.send(outcome).await;
                }
                Ok(AsrPipelineMessage::Job(job)) => {
                    let messages =
                        collect_job_microbatch(job, max_batch_size.max(1), &mut asr_rx).await;
                    let mut jobs = Vec::new();
                    for message in messages {
                        match message {
                            Err(outcome) => {
                                let _ = result_tx.send(outcome).await;
                            }
                            Ok(AsrPipelineMessage::Job(job)) => jobs.push(job),
                            Ok(AsrPipelineMessage::Segment(_)) => {
                                tracing::warn!(
                                    "ASR worker received segment while collecting job micro-batch"
                                );
                            }
                        }
                    }
                    if jobs.is_empty() {
                        continue;
                    }
                    let outcomes = tokio::task::spawn_blocking({
                        let offline = Arc::clone(&offline);
                        let asr_options = asr_options.clone();
                        move || run_asr_jobs_batch(&offline, jobs, &asr_options)
                    })
                    .await
                    .unwrap_or_else(|error| {
                        vec![TranscribeItemOutcome {
                            index: 0,
                            result: Err(anyhow::anyhow!("ASR join error: {error}")),
                            audio_seconds: 0.0,
                            bad_component: Some("recognizer"),
                        }]
                    });
                    for outcome in outcomes {
                        let _ = result_tx.send(outcome).await;
                    }
                }
                Ok(AsrPipelineMessage::Segment(segment)) => {
                    let segments =
                        collect_segment_microbatch(segment, max_batch_size.max(1), &mut asr_rx)
                            .await;
                    prepare_segment_job_state(&mut segment_state, &segments);
                    let stage_start = Instant::now();
                    let batch = tokio::task::spawn_blocking({
                        let offline = Arc::clone(&offline);
                        let asr_options = asr_options.clone();
                        let segments = segments.clone();
                        move || run_segment_asr_inference(&offline, &segments, &asr_options)
                    })
                    .await
                    .unwrap_or_else(|error| Err(anyhow::anyhow!("ASR join error: {error}")));
                    let outcomes =
                        finish_segment_asr_batch(&mut segment_state, &segments, stage_start, batch);
                    for outcome in outcomes {
                        let _ = result_tx.send(outcome).await;
                    }
                }
            }
        }
    })
}

async fn collect_job_microbatch(
    first: AsrJob,
    max_batch_size: usize,
    asr_rx: &mut mpsc::Receiver<Result<AsrPipelineMessage, TranscribeItemOutcome>>,
) -> Vec<Result<AsrPipelineMessage, TranscribeItemOutcome>> {
    let mut messages = vec![Ok(AsrPipelineMessage::Job(first))];
    if max_batch_size <= 1 {
        return messages;
    }
    while messages.len() < max_batch_size {
        match asr_rx.try_recv() {
            Ok(message) => messages.push(message),
            Err(mpsc::error::TryRecvError::Empty) => break,
            Err(mpsc::error::TryRecvError::Disconnected) => return messages,
        }
    }
    if messages.len() >= max_batch_size {
        return messages;
    }
    while messages.len() < max_batch_size {
        match asr_rx.recv().await {
            Some(message) => messages.push(message),
            None => break,
        }
    }
    messages
}

async fn collect_segment_microbatch(
    first: AsrSegmentJob,
    max_batch_size: usize,
    asr_rx: &mut mpsc::Receiver<Result<AsrPipelineMessage, TranscribeItemOutcome>>,
) -> Vec<AsrSegmentJob> {
    let mut segments = vec![first];
    if max_batch_size <= 1 {
        return segments;
    }
    while segments.len() < max_batch_size {
        match asr_rx.try_recv() {
            Ok(Ok(AsrPipelineMessage::Segment(segment))) => segments.push(segment),
            Ok(Ok(AsrPipelineMessage::Job(_))) => break,
            Ok(Err(_)) => break,
            Err(mpsc::error::TryRecvError::Empty) => break,
            Err(mpsc::error::TryRecvError::Disconnected) => return segments,
        }
    }
    if segments.len() >= max_batch_size {
        return segments;
    }
    while segments.len() < max_batch_size {
        match asr_rx.recv().await {
            Some(Ok(AsrPipelineMessage::Segment(segment))) => segments.push(segment),
            Some(Ok(AsrPipelineMessage::Job(_))) | Some(Err(_)) | None => break,
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
                audio_seconds: segment.job_audio_seconds,
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

#[derive(Debug)]
struct SegmentAsrInference {
    targets: Vec<(usize, DurationMs)>,
    segment_audio_seconds: f64,
    original_audio_seconds: f64,
    active_job_count: usize,
}

fn run_segment_asr_inference(
    offline: &OfflinePipeline,
    segments: &[AsrSegmentJob],
    asr_options: &AsrOptions,
) -> Result<(SegmentAsrInference, Vec<Timeline>)> {
    let slices: Vec<Waveform> = segments
        .iter()
        .map(|segment| segment.waveform.clone())
        .collect();
    let targets: Vec<(usize, DurationMs)> = segments
        .iter()
        .map(|segment| (segment.job_index, segment.offset))
        .collect();
    let mut seen_jobs = HashSet::new();
    let original_audio_seconds: f64 = segments
        .iter()
        .filter_map(|segment| {
            seen_jobs
                .insert(segment.job_index)
                .then_some(segment.job_audio_seconds)
        })
        .sum();
    let segment_audio_seconds: f64 = slices.iter().map(Waveform::duration_seconds).sum();
    let active_job_count = seen_jobs.len();
    let timelines = offline
        .asr
        .transcribe_batch(slices.as_slice(), asr_options)?;
    Ok((
        SegmentAsrInference {
            targets,
            segment_audio_seconds,
            original_audio_seconds,
            active_job_count,
        },
        timelines,
    ))
}

fn finish_segment_asr_batch(
    state: &mut SegmentAsrState,
    segments: &[AsrSegmentJob],
    stage_start: Instant,
    batch: Result<(SegmentAsrInference, Vec<Timeline>)>,
) -> Vec<TranscribeItemOutcome> {
    if segments.is_empty() {
        return Vec::new();
    }

    let mut completed = Vec::new();
    match batch {
        Ok((inference, timelines)) => {
            log_asr(
                "vad-segment",
                inference.active_job_count,
                segments.len(),
                inference.segment_audio_seconds,
                inference.original_audio_seconds,
                stage_start.elapsed().as_secs_f64(),
            );
            if timelines.len() == inference.targets.len() {
                for ((job_index, offset), asr_timeline) in
                    inference.targets.into_iter().zip(timelines)
                {
                    if let Some(job) = state.jobs.get_mut(&job_index) {
                        job.timeline
                            .annotations
                            .extend(offset_annotations(asr_timeline.annotations, offset));
                        job.segments_done = job.segments_done.saturating_add(1);
                        if job.segments_done >= job.segment_count {
                            if let Some(job) = state.jobs.remove(&job_index) {
                                completed.push(TranscribeItemOutcome {
                                    index: job.index,
                                    result: Ok(job.timeline),
                                    audio_seconds: job.audio_seconds,
                                    bad_component: None,
                                });
                            }
                        }
                    }
                }
            } else {
                let err = format!(
                    "ASR segment batch returned {} timelines for {} segments",
                    timelines.len(),
                    segments.len()
                );
                let affected: HashSet<usize> =
                    segments.iter().map(|segment| segment.job_index).collect();
                for job_index in affected {
                    if let Some(job) = state.jobs.remove(&job_index) {
                        completed.push(outcome_from_result(
                            job.index,
                            job.audio_seconds,
                            Err(anyhow::anyhow!(err.clone())),
                        ));
                    }
                }
            }
        }
        Err(error) => {
            let reason = error.to_string();
            let affected: HashSet<usize> =
                segments.iter().map(|segment| segment.job_index).collect();
            for job_index in affected {
                if let Some(job) = state.jobs.remove(&job_index) {
                    completed.push(outcome_from_result(
                        job.index,
                        job.audio_seconds,
                        Err(anyhow::anyhow!(reason.clone())),
                    ));
                }
            }
        }
    }

    completed
}

fn run_segment_microbatch(
    offline: &OfflinePipeline,
    segments: Vec<AsrSegmentJob>,
    asr_options: &AsrOptions,
    state: &mut SegmentAsrState,
) -> Vec<TranscribeItemOutcome> {
    if segments.is_empty() {
        return Vec::new();
    }

    prepare_segment_job_state(state, &segments);
    let stage_start = Instant::now();
    let batch = run_segment_asr_inference(offline, &segments, asr_options);
    finish_segment_asr_batch(state, &segments, stage_start, batch)
}

fn run_asr_jobs_batch(
    offline: &OfflinePipeline,
    jobs: Vec<AsrJob>,
    asr_options: &AsrOptions,
) -> Vec<TranscribeItemOutcome> {
    let mut outcomes = Vec::with_capacity(jobs.len());
    let mut raw_jobs = Vec::new();
    let mut prepared_jobs = Vec::new();

    for job in jobs {
        match job.prepared {
            VadPreparation::Speech(_) => prepared_jobs.push(job),
            VadPreparation::NoSpeech => outcomes.push(TranscribeItemOutcome {
                index: job.index,
                result: Ok(Timeline::new("offline_audio")),
                audio_seconds: job.waveform.duration_seconds(),
                bad_component: None,
            }),
            VadPreparation::Disabled => raw_jobs.push(job),
        }
    }

    if !raw_jobs.is_empty() {
        let indices: Vec<usize> = raw_jobs.iter().map(|job| job.index).collect();
        let audio_seconds: Vec<f64> = raw_jobs
            .iter()
            .map(|job| job.waveform.duration_seconds())
            .collect();
        let total_audio_seconds: f64 = audio_seconds.iter().sum();
        let waveforms: Vec<Waveform> = raw_jobs.into_iter().map(|job| job.waveform).collect();
        let stage_start = Instant::now();
        let result = offline.asr.transcribe_batch(&waveforms, asr_options);
        log_asr(
            "raw",
            indices.len(),
            waveforms.len(),
            total_audio_seconds,
            total_audio_seconds,
            stage_start.elapsed().as_secs_f64(),
        );
        match result {
            Ok(timelines) if timelines.len() == indices.len() => {
                for ((index, audio_seconds), timeline) in
                    indices.into_iter().zip(audio_seconds).zip(timelines)
                {
                    outcomes.push(TranscribeItemOutcome {
                        index,
                        result: Ok(timeline),
                        audio_seconds,
                        bad_component: None,
                    });
                }
            }
            Ok(timelines) => {
                let err = anyhow::anyhow!(
                    "ASR batch returned {} timelines for {} jobs",
                    timelines.len(),
                    indices.len()
                );
                for (index, audio_seconds) in indices.into_iter().zip(audio_seconds) {
                    outcomes.push(outcome_from_result(
                        index,
                        audio_seconds,
                        Err(anyhow::anyhow!("{err}")),
                    ));
                }
            }
            Err(error) => {
                let reason = error.to_string();
                for (index, audio_seconds) in indices.into_iter().zip(audio_seconds) {
                    outcomes.push(outcome_from_result(
                        index,
                        audio_seconds,
                        Err(anyhow::anyhow!("{reason}")),
                    ));
                }
            }
        }
    }

    if !prepared_jobs.is_empty() {
        let mut segment_state = SegmentAsrState::default();
        let mut segment_jobs = Vec::new();
        for job in prepared_jobs {
            match speech_job_to_segments(job) {
                Ok(mut segments) => segment_jobs.append(&mut segments),
                Err(error) => outcomes.push(TranscribeItemOutcome {
                    index: 0,
                    result: Err(error),
                    audio_seconds: 0.0,
                    bad_component: Some("vad"),
                }),
            }
        }
        outcomes.extend(run_segment_microbatch(
            offline,
            segment_jobs,
            asr_options,
            &mut segment_state,
        ));
    }

    outcomes.sort_unstable_by_key(|outcome| outcome.index);
    outcomes
}

#[derive(Debug, Clone, Copy)]
struct StageMetrics {
    speedup: f64,
    rtf: f64,
}

fn stage_metrics(audio_seconds: f64, wall_seconds: f64) -> StageMetrics {
    if audio_seconds <= 0.0 || wall_seconds <= 0.0 {
        return StageMetrics {
            speedup: 0.0,
            rtf: 0.0,
        };
    }
    StageMetrics {
        speedup: audio_seconds / wall_seconds,
        rtf: wall_seconds / audio_seconds,
    }
}

fn vad_preparation_stats(prepared: &VadPreparation) -> (usize, f64) {
    match prepared {
        VadPreparation::Speech(prepared) => {
            let speech_seconds = prepared
                .segments
                .iter()
                .map(|segment| segment.range.duration().0 as f64 / 1000.0)
                .sum();
            (prepared.segments.len(), speech_seconds)
        }
        VadPreparation::Disabled | VadPreparation::NoSpeech => (0, 0.0),
    }
}

fn log_loader(index: usize, audio_seconds: f64, wall_seconds: f64) {
    let metrics = stage_metrics(audio_seconds, wall_seconds);
    tracing::info!(
        target: "vasr_server::async_transcribe",
        "loader | job={} | audio={:.2}s | spent={:.2}s | speed={:.2}x | rtf={:.3}",
        index,
        audio_seconds,
        wall_seconds,
        metrics.speedup,
        metrics.rtf
    );
}

fn log_vad(
    index: usize,
    audio_seconds: f64,
    segments: usize,
    speech_seconds: f64,
    wall_seconds: f64,
) {
    let metrics = stage_metrics(audio_seconds, wall_seconds);
    tracing::info!(
        target: "vasr_server::async_transcribe",
        "vad | job={} | audio={:.2}s | segments={} | speech={:.2}s | spent={:.2}s | speed={:.2}x | rtf={:.3}",
        index,
        audio_seconds,
        segments,
        speech_seconds,
        wall_seconds,
        metrics.speedup,
        metrics.rtf
    );
}

fn log_asr(
    kind: &'static str,
    batch_size: usize,
    segments: usize,
    segment_audio_seconds: f64,
    original_audio_seconds: f64,
    wall_seconds: f64,
) {
    let metrics = stage_metrics(segment_audio_seconds, wall_seconds);
    tracing::info!(
        target: "vasr_server::async_transcribe",
        "asr | kind={} | batch={} | segments={} | segment_audio={:.2}s | original_audio={:.2}s | spent={:.2}s | speed={:.2}x | rtf={:.3}",
        kind,
        batch_size,
        segments,
        segment_audio_seconds,
        original_audio_seconds,
        wall_seconds,
        metrics.speedup,
        metrics.rtf
    );
}

fn log_pipeline(job_count: usize, outcomes: &[TranscribeItemOutcome], wall_seconds: f64) {
    let audio_seconds: f64 = outcomes.iter().map(|outcome| outcome.audio_seconds).sum();
    let bad = outcomes
        .iter()
        .filter(|outcome| outcome.result.is_err())
        .count();
    let metrics = stage_metrics(audio_seconds, wall_seconds);
    tracing::info!(
        target: "vasr_server::async_transcribe",
        "pipeline | batch={} | returned={} | audio={:.2}s | spent={:.2}s | speed={:.2}x | rtf={:.3} | bad={}",
        job_count,
        outcomes.len(),
        audio_seconds,
        wall_seconds,
        metrics.speedup,
        metrics.rtf,
        bad
    );
}

fn outcome_from_result(
    index: usize,
    audio_seconds: f64,
    result: Result<Timeline>,
) -> TranscribeItemOutcome {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use vasr_data::{
        Annotation, AnnotationPayload, AnnotationSource, AnnotationStatus, DurationMs, TextSegment,
        TimeRange,
    };
    use vasr_runtime::{AsrModel, StreamingAsrModel, VadSegment};

    #[derive(Default)]
    struct BatchRecordingAsr {
        seen_batch_sizes: Mutex<Vec<usize>>,
    }

    impl AsrModel for BatchRecordingAsr {
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
            self.seen_batch_sizes
                .lock()
                .expect("seen batch sizes poisoned")
                .push(waveforms.len());
            Ok(waveforms
                .iter()
                .map(|_| {
                    let mut timeline = Timeline::new("fake_batch_asr");
                    timeline.push(Annotation::new(
                        Default::default(),
                        AnnotationPayload::Segment(TextSegment::new("ok")),
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

    #[test]
    fn no_vad_jobs_are_transcribed_as_one_batch() -> Result<()> {
        let asr = Arc::new(BatchRecordingAsr::default());
        let pipeline = OfflinePipeline {
            vad: None,
            asr: asr.clone(),
        };
        let jobs = vec![
            AsrJob {
                index: 0,
                prepared: VadPreparation::Disabled,
                waveform: Waveform::new(vec![0.1; 160], 16_000),
            },
            AsrJob {
                index: 1,
                prepared: VadPreparation::Disabled,
                waveform: Waveform::new(vec![0.2; 160], 16_000),
            },
            AsrJob {
                index: 2,
                prepared: VadPreparation::Disabled,
                waveform: Waveform::new(vec![0.3; 160], 16_000),
            },
        ];

        let outcomes = run_asr_jobs_batch(&pipeline, jobs, &AsrOptions::default());

        assert_eq!(outcomes.len(), 3);
        assert!(outcomes.iter().all(|outcome| outcome.result.is_ok()));
        assert_eq!(
            *asr.seen_batch_sizes
                .lock()
                .expect("seen batch sizes poisoned"),
            vec![3]
        );
        Ok(())
    }

    #[test]
    fn vad_segments_from_multiple_jobs_are_transcribed_as_one_batch() -> Result<()> {
        let asr = Arc::new(BatchRecordingAsr::default());
        let pipeline = OfflinePipeline {
            vad: None,
            asr: asr.clone(),
        };
        let jobs = vec![
            AsrJob {
                index: 0,
                prepared: VadPreparation::Speech(prepared_with_segments(&[(0, 100), (250, 400)])),
                waveform: Waveform::new(vec![0.1; 8_000], 16_000),
            },
            AsrJob {
                index: 1,
                prepared: VadPreparation::Speech(prepared_with_segments(&[(50, 200)])),
                waveform: Waveform::new(vec![0.2; 8_000], 16_000),
            },
        ];

        let outcomes = run_asr_jobs_batch(&pipeline, jobs, &AsrOptions::default());

        assert_eq!(outcomes.len(), 2);
        assert!(outcomes.iter().all(|outcome| outcome.result.is_ok()));
        assert_eq!(
            *asr.seen_batch_sizes
                .lock()
                .expect("seen batch sizes poisoned"),
            vec![3]
        );
        Ok(())
    }

    #[test]
    fn no_speech_jobs_are_returned_without_asr() -> Result<()> {
        let asr = Arc::new(BatchRecordingAsr::default());
        let pipeline = OfflinePipeline {
            vad: None,
            asr: asr.clone(),
        };
        let jobs = vec![AsrJob {
            index: 0,
            prepared: VadPreparation::NoSpeech,
            waveform: Waveform::new(vec![0.1; 160], 16_000),
        }];

        let outcomes = run_asr_jobs_batch(&pipeline, jobs, &AsrOptions::default());

        assert_eq!(outcomes.len(), 1);
        assert!(
            outcomes[0]
                .result
                .as_ref()
                .is_ok_and(|timeline| timeline.annotations.is_empty())
        );
        assert!(
            asr.seen_batch_sizes
                .lock()
                .expect("seen batch sizes poisoned")
                .is_empty()
        );
        Ok(())
    }

    #[tokio::test]
    async fn asr_worker_coalesces_slightly_staggered_jobs() -> Result<()> {
        let asr = Arc::new(BatchRecordingAsr::default());
        let pipeline = Arc::new(OfflinePipeline {
            vad: None,
            asr: asr.clone(),
        });
        let (asr_tx, asr_rx) = mpsc::channel(4);
        let (result_tx, mut result_rx) = mpsc::channel(4);
        let handle = spawn_asr_worker(pipeline, AsrOptions::default(), 4, asr_rx, result_tx);

        for index in 0..3usize {
            asr_tx
                .send(Ok(AsrPipelineMessage::Job(AsrJob {
                    index,
                    prepared: VadPreparation::Disabled,
                    waveform: Waveform::new(vec![index as f32; 160], 16_000),
                })))
                .await?;
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        }
        drop(asr_tx);

        let mut outcomes = Vec::new();
        while let Some(outcome) = result_rx.recv().await {
            outcomes.push(outcome);
        }
        handle.await?;

        assert_eq!(outcomes.len(), 3);
        assert_eq!(
            *asr.seen_batch_sizes
                .lock()
                .expect("seen batch sizes poisoned"),
            vec![3]
        );
        Ok(())
    }

    #[test]
    fn duplicate_transcribe_inputs_keep_one_canonical_job() {
        let inputs = vec![
            TranscribeInput {
                index: 10,
                source: AudioSource::Url("file:///same.wav".to_string()),
            },
            TranscribeInput {
                index: 11,
                source: AudioSource::Url("file:///other.wav".to_string()),
            },
            TranscribeInput {
                index: 12,
                source: AudioSource::Url("file:///same.wav".to_string()),
            },
        ];

        let deduped = deduplicate_transcribe_inputs(inputs);

        assert_eq!(
            deduped
                .unique_inputs
                .iter()
                .map(|input| input.index)
                .collect::<Vec<_>>(),
            vec![10, 11]
        );
        assert_eq!(deduped.duplicate_indices, vec![(12, 10)]);
    }

    #[test]
    fn local_file_urls_with_same_content_keep_one_canonical_job() {
        let dir =
            std::env::temp_dir().join(format!("vasr-dedup-test-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let first = dir.join("first.wav");
        let second = dir.join("second.wav");
        std::fs::write(&first, b"same audio bytes").expect("write first file");
        std::fs::write(&second, b"same audio bytes").expect("write second file");

        let inputs = vec![
            TranscribeInput {
                index: 0,
                source: AudioSource::Url(format!("file://{}", first.display())),
            },
            TranscribeInput {
                index: 1,
                source: AudioSource::Url(format!("file://{}", second.display())),
            },
        ];

        let deduped = deduplicate_transcribe_inputs(inputs);

        assert_eq!(deduped.unique_inputs.len(), 1);
        assert_eq!(deduped.unique_inputs[0].index, 0);
        assert_eq!(deduped.duplicate_indices, vec![(1, 0)]);

        std::fs::remove_dir_all(dir).expect("remove temp dir");
    }

    #[test]
    fn duplicate_transcribe_outcomes_are_expanded_to_original_indices() {
        let mut timeline = Timeline::new("canonical");
        timeline.push(Annotation::new(
            Default::default(),
            AnnotationPayload::Segment(TextSegment::new("same result")),
            AnnotationSource::Model("fake_batch_asr".to_string()),
            AnnotationStatus::Final,
        ));
        let outcomes = vec![
            TranscribeItemOutcome {
                index: 10,
                result: Ok(timeline),
                audio_seconds: 1.25,
                bad_component: None,
            },
            TranscribeItemOutcome {
                index: 11,
                result: Err(anyhow::anyhow!("loader failed")),
                audio_seconds: 0.0,
                bad_component: Some("loader"),
            },
        ];

        let expanded = expand_duplicate_outcomes(outcomes, &[(12, 10), (13, 11)]);

        assert_eq!(
            expanded
                .iter()
                .map(|outcome| outcome.index)
                .collect::<Vec<_>>(),
            vec![10, 11, 12, 13]
        );
        assert_eq!(expanded[2].audio_seconds, 1.25);
        assert!(
            expanded[2]
                .result
                .as_ref()
                .is_ok_and(|timeline| { timeline.transcript().text.trim() == "same result" })
        );
        assert_eq!(expanded[3].bad_component, Some("loader"));
        assert!(expanded[3].result.is_err());
    }

    #[test]
    fn stage_metrics_calculate_speedup_and_rtf() {
        let metrics = stage_metrics(10.0, 2.0);
        assert_eq!(metrics.speedup, 5.0);
        assert_eq!(metrics.rtf, 0.2);

        let empty_audio = stage_metrics(0.0, 2.0);
        assert_eq!(empty_audio.speedup, 0.0);
        assert_eq!(empty_audio.rtf, 0.0);
    }

    fn prepared_with_segments(ranges: &[(u64, u64)]) -> VadPrepared {
        let mut speech_annotations = Vec::new();
        let mut segments = Vec::new();
        let mut slices = Vec::new();
        for &(start, end) in ranges {
            let range = TimeRange::new(DurationMs(start), DurationMs(end));
            speech_annotations.push(Annotation::new(
                range,
                AnnotationPayload::Speech,
                AnnotationSource::Model("fake_vad".to_string()),
                AnnotationStatus::Final,
            ));
            segments.push(VadSegment {
                range,
                probability: 0.9,
            });
            let sample_count = ((end - start) as usize).max(1) * 16;
            slices.push(Waveform::new(vec![0.1; sample_count], 16_000));
        }
        VadPrepared {
            speech_annotations,
            segments,
            slices,
        }
    }
}
