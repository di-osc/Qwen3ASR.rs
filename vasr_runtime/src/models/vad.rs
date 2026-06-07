use std::ffi::CStr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear, linear_no_bias, ops::softmax_last_dim};
use hf_hub::api::sync::Api;
use kaldi_fbank_rust_kautism::{
    FbankOptions, FrameExtractionOptions, MelBanksOptions, OnlineFbank,
};
use ndarray::{Array1, Array2, s};
use vasr_data::{
    Annotation, AnnotationPayload, AnnotationSource, AnnotationStatus, AudioChunk, DurationMs,
    TimeRange, Timeline, Waveform,
};

use crate::model::{StreamingVadModel, VadModel, VadOptions, VadSegment};
use crate::models::e2e_vad::{E2EVadConfig, E2EVadModel};

pub const FSMN_VAD_CHUNK_SIZE: usize = 512;
const FSMN_VAD_FRAME_SHIFT_SAMPLES: usize = 160;
const FSMN_VAD_FRAME_LENGTH_SAMPLES: usize = 400;
const FSMN_VAD_FEAT_CHUNK_SIZE: usize = 6000;
const FSMN_VAD_REPO: &str = "funasr/fsmn-vad";
const FSMN_VAD_LAYERS: usize = 4;
const FSMN_VAD_PROJ_DIM: usize = 128;
const FSMN_VAD_CACHE_FRAMES: usize = 19;
const FSMN_VAD_SAMPLE_RATE: u32 = 16_000;
const FSMN_VAD_SOURCE: &str = "fsmn_vad";

#[derive(Debug, Clone, Copy, Default)]
pub struct FsmnVadTiming {
    pub pcm_seconds: f64,
    pub frontend_seconds: f64,
    pub forward_seconds: f64,
    pub segmenter_seconds: f64,
    pub chunks: usize,
}

impl FsmnVadTiming {
    fn add_pcm(&mut self, duration: Duration) {
        self.pcm_seconds += duration.as_secs_f64();
    }

    fn add_segmenter(&mut self, duration: Duration) {
        self.segmenter_seconds += duration.as_secs_f64();
    }
}

#[derive(Debug, Clone)]
pub struct FsmnVadDetection {
    pub segments: Vec<VadSegment>,
    pub timing: FsmnVadTiming,
}

#[derive(Debug, Clone)]
pub struct FsmnVadModel {
    weights: Arc<FsmnVadWeights>,
    model_dir: PathBuf,
}

impl FsmnVadModel {
    pub fn from_pretrained(path: Option<impl AsRef<Path>>) -> Result<Self> {
        let model_dir = match path {
            Some(path) => path.as_ref().to_path_buf(),
            None => download_fsmn_vad()?,
        };
        let weights = Arc::new(FsmnVadWeights::load(&model_dir)?);
        Ok(Self { weights, model_dir })
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    pub fn annotate(&self, waveform: &Waveform, options: &VadOptions) -> Timeline {
        let mut timeline = Timeline::new("vad_audio");
        if let Ok(segments) = self.detect(waveform, options) {
            for segment in segments {
                timeline.push(speech_annotation(
                    ms_to_samples(segment.range.start.0),
                    ms_to_samples(segment.range.end.0),
                    AnnotationStatus::Final,
                ));
            }
        }
        timeline
    }

    fn new_session(&self, options: &VadOptions) -> Result<FsmnVadSession> {
        FsmnVadSession::new(Arc::clone(&self.weights), options.clone())
    }

    pub fn detect_with_timing(
        &self,
        waveform: &Waveform,
        options: &VadOptions,
    ) -> Result<FsmnVadDetection> {
        if waveform.sample_rate != FSMN_VAD_SAMPLE_RATE {
            bail!(
                "FSMN VAD expects 16kHz mono audio, got sample_rate={}",
                waveform.sample_rate
            );
        }
        let detection = self.detect_offline(waveform, options)?;
        let segments = detection.segments;
        let timing = detection.timing;
        tracing::info!(
            target: "vasr_runtime::models::vad",
            "fsmn_vad | audio={:.2}s | chunks={} | segments={} | pcm={:.3}s | frontend={:.3}s | forward={:.3}s | segmenter={:.3}s",
            waveform.duration_seconds(),
            timing.chunks,
            segments.len(),
            timing.pcm_seconds,
            timing.frontend_seconds,
            timing.forward_seconds,
            timing.segmenter_seconds,
        );
        Ok(FsmnVadDetection { segments, timing })
    }

    fn detect_offline(
        &self,
        waveform: &Waveform,
        options: &VadOptions,
    ) -> Result<FsmnVadDetection> {
        let mut timing = FsmnVadTiming::default();
        let pcm_start = Instant::now();
        let pcm = samples_f32_to_i16(&waveform.samples);
        timing.add_pcm(pcm_start.elapsed());

        let frontend_start = Instant::now();
        let feats = self.weights.frontend.extract_features_from_pcm(&pcm)?;
        timing.frontend_seconds += frontend_start.elapsed().as_secs_f64();

        let total_frames = feats.shape()[0];
        if total_frames == 0 {
            return Ok(FsmnVadDetection {
                segments: Vec::new(),
                timing,
            });
        }

        let mut caches = self.weights.zero_caches()?;
        let mut e2e = build_e2e_model(options);
        let max_end_sil = options.min_silence_ms as i32;
        let mut ms_segments = Vec::new();
        let forward_start = Instant::now();
        let mut frame_offset = 0usize;

        while frame_offset < total_frames {
            let step = FSMN_VAD_FEAT_CHUNK_SIZE.min(total_frames - frame_offset);
            let is_final = frame_offset + step >= total_frames;
            let feat_chunk = feats
                .slice(s![frame_offset..frame_offset + step, ..])
                .to_owned();
            let frame_scores = self
                .weights
                .forward_frame_scores(&feat_chunk, &mut caches)?;
            let wave_start = frame_offset * FSMN_VAD_FRAME_SHIFT_SAMPLES;
            let wave_end = if is_final {
                waveform.samples.len()
            } else {
                ((frame_offset + step - 1) * FSMN_VAD_FRAME_SHIFT_SAMPLES
                    + FSMN_VAD_FRAME_LENGTH_SAMPLES)
                    .min(waveform.samples.len())
            };
            let segmenter_start = Instant::now();
            ms_segments.extend(e2e.detect_chunk(
                &frame_scores,
                &waveform.samples[wave_start..wave_end],
                is_final,
                max_end_sil,
            ));
            timing.add_segmenter(segmenter_start.elapsed());
            frame_offset += step;
        }

        timing.forward_seconds += forward_start.elapsed().as_secs_f64();
        timing.chunks = 1;

        let segments = ms_segments
            .into_iter()
            .map(|(start, end)| VadSegment {
                range: TimeRange::new(DurationMs(start), DurationMs(end)),
                probability: options.threshold,
            })
            .collect();

        Ok(FsmnVadDetection { segments, timing })
    }
}

fn fsmn_vad_device() -> Result<Device> {
    let requested = std::env::var("VASR_VAD_DEVICE")
        .ok()
        .map(|value| value.trim().to_ascii_lowercase());
    match requested.as_deref() {
        Some("cuda") | Some("gpu") => {
            #[cfg(any(feature = "cuda", feature = "cuda-vad"))]
            {
                return fsmn_vad_cuda_device();
            }
            #[cfg(not(any(feature = "cuda", feature = "cuda-vad")))]
            {
                bail!(
                    "VASR_VAD_DEVICE=cuda requires building with `--features cuda-vad` or `cuda`"
                );
            }
        }
        Some("cpu") => {}
        None =>
        {
            #[cfg(any(feature = "cuda", feature = "cuda-vad"))]
            match fsmn_vad_cuda_device() {
                Ok(device) => return Ok(device),
                Err(err) => {
                    tracing::warn!(
                        target: "vasr_runtime::models::vad",
                        "FSMN VAD CUDA auto-select failed; falling back to CPU: {err}"
                    );
                }
            }
        }
        Some(other) => bail!("unknown VASR_VAD_DEVICE {other:?}; expected cpu or cuda"),
    }
    Ok(Device::Cpu)
}

#[cfg(any(feature = "cuda", feature = "cuda-vad"))]
fn fsmn_vad_cuda_device() -> Result<Device> {
    let device = Device::new_cuda(0)
        .map_err(|err| anyhow::anyhow!("failed to create CUDA device 0: {err}"))?;
    tracing::info!(
        target: "vasr_runtime::models::vad",
        "FSMN VAD using CUDA device 0"
    );
    Ok(device)
}

fn build_e2e_model(options: &VadOptions) -> E2EVadModel {
    let mut config = E2EVadConfig::default();
    config.speech_noise_thres = options.threshold;
    E2EVadModel::new(config)
}

impl VadModel for FsmnVadModel {
    fn detect(&self, waveform: &Waveform, options: &VadOptions) -> Result<Vec<VadSegment>> {
        Ok(self.detect_with_timing(waveform, options)?.segments)
    }

    fn start_stream(&self, options: &VadOptions) -> Result<Box<dyn StreamingVadModel>> {
        Ok(Box::new(self.new_session(options)?))
    }
}

struct FsmnVadWeights {
    frontend: WavFrontend,
    in_linear1: Linear,
    in_linear2: Linear,
    fsmn_blocks: Vec<FsmnVadBlock>,
    out_linear1: Linear,
    out_linear2: Linear,
    device: Device,
}

impl std::fmt::Debug for FsmnVadWeights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FsmnVadWeights").finish_non_exhaustive()
    }
}

impl FsmnVadWeights {
    fn load(model_dir: &Path) -> Result<Self> {
        Self::load_on_device(model_dir, &fsmn_vad_device()?)
    }

    fn load_on_device(model_dir: &Path, device: &Device) -> Result<Self> {
        let model_path = model_dir.join("model.pt");
        let cmvn_path = model_dir.join("am.mvn");
        if !model_path.exists() {
            bail!("FSMN VAD model.pt not found in {}", model_dir.display());
        }
        if !cmvn_path.exists() {
            bail!("FSMN VAD am.mvn not found in {}", model_dir.display());
        }
        let frontend = WavFrontend::new(WavFrontendConfig {
            sample_rate: FSMN_VAD_SAMPLE_RATE as i32,
            lfr_m: 5,
            lfr_n: 1,
            cmvn_file: Some(cmvn_path),
            ..Default::default()
        })?;

        let vb = VarBuilder::from_pth(&model_path, DType::F32, &device)
            .with_context(|| format!("failed to load FSMN VAD weights {}", model_path.display()))?;
        let encoder_vb = vb.pp("encoder");
        let in_linear1 = linear(400, 140, encoder_vb.pp("in_linear1").pp("linear"))?;
        let in_linear2 = linear(140, 250, encoder_vb.pp("in_linear2").pp("linear"))?;
        let out_linear1 = linear(250, 140, encoder_vb.pp("out_linear1").pp("linear"))?;
        let out_linear2 = linear(140, 248, encoder_vb.pp("out_linear2").pp("linear"))?;

        let mut fsmn_blocks = Vec::with_capacity(FSMN_VAD_LAYERS);
        for i in 0..FSMN_VAD_LAYERS {
            fsmn_blocks.push(FsmnVadBlock::new(encoder_vb.pp("fsmn").pp(i))?);
        }

        Ok(Self {
            frontend,
            in_linear1,
            in_linear2,
            fsmn_blocks,
            out_linear1,
            out_linear2,
            device: device.clone(),
        })
    }

    fn zero_caches(&self) -> Result<Vec<Tensor>> {
        let cache = Tensor::zeros(
            (1usize, FSMN_VAD_PROJ_DIM, FSMN_VAD_CACHE_FRAMES, 1usize),
            DType::F32,
            &self.device,
        )?;
        Ok((0..FSMN_VAD_LAYERS)
            .map(|_| cache.clone())
            .collect::<Vec<_>>())
    }

    fn forward_frame_scores(
        &self,
        feats: &Array2<f32>,
        caches: &mut [Tensor],
    ) -> Result<Vec<Vec<f32>>> {
        let t = feats.shape()[0];
        let d = feats.shape()[1];
        if d != 400 {
            bail!("FSMN VAD expects feature dim 400, got {d}");
        }
        if t == 0 {
            return Ok(Vec::new());
        }

        let speech = Tensor::from_vec(
            feats.iter().copied().collect::<Vec<f32>>(),
            (1usize, t, d),
            &self.device,
        )?;

        let mut x = self.in_linear1.forward(&speech)?;
        x = self.in_linear2.forward(&x)?.relu()?;
        for (idx, block) in self.fsmn_blocks.iter().enumerate() {
            let (new_x, new_cache) = block.forward(&x, &caches[idx])?;
            x = new_x;
            caches[idx] = new_cache;
        }
        x = self.out_linear1.forward(&x)?;
        x = self.out_linear2.forward(&x)?;
        let logits = softmax_last_dim(&x)?;

        let dims = logits.dims();
        if dims.len() != 3 || dims[0] != 1 || dims[2] == 0 {
            bail!("unexpected FSMN VAD logits shape: {:?}", dims);
        }
        let frames = dims[1];
        let classes = dims[2];
        let data = logits.flatten_all()?.to_vec1::<f32>()?;
        Ok((0..frames)
            .map(|frame_idx| {
                (0..classes)
                    .map(|class_idx| data[frame_idx * classes + class_idx])
                    .collect()
            })
            .collect())
    }
}

struct FsmnVadBlock {
    linear: Linear,
    affine: Linear,
    conv_left_weight: Tensor,
}

impl FsmnVadBlock {
    fn new(vb: VarBuilder) -> Result<Self> {
        let linear_layer = linear_no_bias(250, FSMN_VAD_PROJ_DIM, vb.pp("linear").pp("linear"))?;
        let affine = linear(FSMN_VAD_PROJ_DIM, 250, vb.pp("affine").pp("linear"))?;
        let conv_left_weight = vb
            .pp("fsmn_block")
            .pp("conv_left")
            .get((FSMN_VAD_PROJ_DIM, 1, 20, 1), "weight")?;
        Ok(Self {
            linear: linear_layer,
            affine,
            conv_left_weight,
        })
    }

    fn forward(&self, input: &Tensor, cache: &Tensor) -> Result<(Tensor, Tensor)> {
        let x = self.linear.forward(input)?;
        let x_per = x.unsqueeze(1)?.permute((0, 3, 2, 1))?;
        let y_left = Tensor::cat(&[cache, &x_per], 2)?;
        let y_left_t = y_left.dim(2)?;
        let new_cache =
            y_left.narrow(2, y_left_t - FSMN_VAD_CACHE_FRAMES, FSMN_VAD_CACHE_FRAMES)?;
        let y_left = fsmn_depthwise_conv_left(&y_left, &self.conv_left_weight)?;
        let out = x_per.add(&y_left)?;
        let out = out.permute((0, 3, 2, 1))?.squeeze(1)?;
        let out = self.affine.forward(&out)?.relu()?;
        Ok((out, new_cache))
    }
}

const FSMN_VAD_CONV_KERNEL: usize = 20;

/// Depthwise time convolution for FSMN left context.
/// Implemented with `unfold` + multiply-accumulate so CPU and CUDA stay numerically aligned.
/// Candle's grouped `conv2d` is incorrect on CUDA for this kernel shape.
fn fsmn_depthwise_conv_left(y_left: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let (batch, channels, time, width) = y_left.dims4()?;
    if batch != 1 {
        bail!("FSMN depthwise conv expects batch=1, got {batch}");
    }
    if width != 1 {
        bail!("FSMN depthwise conv expects width=1, got {width}");
    }
    if channels != FSMN_VAD_PROJ_DIM {
        bail!("FSMN depthwise conv expects {FSMN_VAD_PROJ_DIM} channels, got {channels}");
    }
    if time < FSMN_VAD_CONV_KERNEL {
        bail!(
            "FSMN depthwise conv input too short: {time} < {}",
            FSMN_VAD_CONV_KERNEL
        );
    }
    let weight_dims = weight.dims();
    if weight_dims != [channels, 1, FSMN_VAD_CONV_KERNEL, 1] {
        bail!("unexpected FSMN conv weight shape: {weight_dims:?}");
    }

    let windows = y_left.unfold(2, FSMN_VAD_CONV_KERNEL, 1)?;
    let kernel = weight.reshape((1, channels, 1, 1, FSMN_VAD_CONV_KERNEL))?;
    let out = windows.broadcast_mul(&kernel)?.sum_keepdim(4)?.squeeze(4)?;
    Ok(out)
}

pub struct FsmnVadSession {
    weights: Arc<FsmnVadWeights>,
    caches: Vec<Tensor>,
    e2e: E2EVadModel,
    vad_options: VadOptions,
    pending: Vec<i16>,
    timing: FsmnVadTiming,
}

impl FsmnVadSession {
    fn new(weights: Arc<FsmnVadWeights>, options: VadOptions) -> Result<Self> {
        let caches = weights.zero_caches()?;
        Ok(Self {
            weights,
            caches,
            e2e: build_e2e_model(&options),
            vad_options: options,
            pending: Vec::with_capacity(FSMN_VAD_CHUNK_SIZE),
            timing: FsmnVadTiming::default(),
        })
    }

    fn push_samples_f32(&mut self, samples: &[f32]) -> Result<Vec<VadSegment>> {
        let pcm_start = Instant::now();
        let pcm = samples_f32_to_i16(samples);
        self.timing.add_pcm(pcm_start.elapsed());
        self.push_samples_i16(&pcm)
    }

    fn push_samples_i16(&mut self, samples: &[i16]) -> Result<Vec<VadSegment>> {
        let mut segments = Vec::new();
        let mut input = samples;

        if !self.pending.is_empty() {
            let needed = FSMN_VAD_CHUNK_SIZE - self.pending.len();
            let take = needed.min(input.len());
            self.pending.extend_from_slice(&input[..take]);
            input = &input[take..];
            if self.pending.len() == FSMN_VAD_CHUNK_SIZE {
                let mut chunk = [0i16; FSMN_VAD_CHUNK_SIZE];
                chunk.copy_from_slice(&self.pending);
                self.pending.clear();
                segments.extend(self.score_and_segment(&chunk, false)?);
            }
        }

        let (full_chunks, tail) = split_full_chunks(input);
        for chunk_slice in full_chunks.chunks_exact(FSMN_VAD_CHUNK_SIZE) {
            let mut chunk = [0i16; FSMN_VAD_CHUNK_SIZE];
            chunk.copy_from_slice(chunk_slice);
            segments.extend(self.score_and_segment(&chunk, false)?);
        }
        self.pending.extend_from_slice(tail);
        Ok(segments
            .into_iter()
            .map(|(start, end)| VadSegment {
                range: TimeRange::new(DurationMs(start), DurationMs(end)),
                probability: self.vad_options.threshold,
            })
            .collect())
    }

    fn finish_segments(&mut self) -> Result<Vec<VadSegment>> {
        let mut segments = Vec::new();
        if !self.pending.is_empty() {
            let mut chunk = [0i16; FSMN_VAD_CHUNK_SIZE];
            let len = self.pending.len();
            chunk[..len].copy_from_slice(&self.pending);
            self.pending.clear();
            segments.extend(self.score_and_segment(&chunk, true)?);
        } else {
            let segmenter_start = Instant::now();
            segments.extend(
                self.e2e
                    .finalize_streaming(self.vad_options.min_silence_ms as i32),
            );
            self.timing.add_segmenter(segmenter_start.elapsed());
        }
        Ok(segments
            .into_iter()
            .map(|(start, end)| VadSegment {
                range: TimeRange::new(DurationMs(start), DurationMs(end)),
                probability: self.vad_options.threshold,
            })
            .collect())
    }

    fn score_and_segment(
        &mut self,
        chunk: &[i16; FSMN_VAD_CHUNK_SIZE],
        is_final: bool,
    ) -> Result<Vec<(u64, u64)>> {
        let frontend_start = Instant::now();
        let feats = self.weights.frontend.extract_features(chunk)?;
        self.timing.frontend_seconds += frontend_start.elapsed().as_secs_f64();

        let forward_start = Instant::now();
        let frame_scores = self
            .weights
            .forward_frame_scores(&feats, &mut self.caches)?;
        self.timing.forward_seconds += forward_start.elapsed().as_secs_f64();
        self.timing.chunks += 1;

        let wave_chunk: Vec<f32> = chunk
            .iter()
            .map(|&sample| f32::from(sample) / f32::from(i16::MAX))
            .collect();
        let segmenter_start = Instant::now();
        let segments = self.e2e.detect_chunk(
            &frame_scores,
            &wave_chunk,
            is_final,
            self.vad_options.min_silence_ms as i32,
        );
        self.timing.add_segmenter(segmenter_start.elapsed());
        Ok(segments)
    }
}

fn split_full_chunks(samples: &[i16]) -> (&[i16], &[i16]) {
    let full_len = samples.len() / FSMN_VAD_CHUNK_SIZE * FSMN_VAD_CHUNK_SIZE;
    samples.split_at(full_len)
}

fn samples_f32_to_i16(samples: &[f32]) -> Vec<i16> {
    samples
        .iter()
        .map(|sample| {
            (sample.clamp(-1.0, 1.0) * f32::from(i16::MAX))
                .round()
                .clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16
        })
        .collect()
}

fn extract_cmvn_vector(text: &str, section: &str) -> Result<Vec<f32>> {
    let section_start = text
        .find(section)
        .ok_or_else(|| anyhow::anyhow!("missing {section} section"))?;
    let after_section = &text[section_start + section.len()..];
    let learn_rate = "<LearnRateCoef>";
    let learn_start = after_section
        .find(learn_rate)
        .ok_or_else(|| anyhow::anyhow!("missing {learn_rate} after {section}"))?;
    let after_learn = &after_section[learn_start + learn_rate.len()..];
    let bracket_start = after_learn
        .find('[')
        .ok_or_else(|| anyhow::anyhow!("missing vector start after {section}"))?;
    let after_bracket = &after_learn[bracket_start + 1..];
    let bracket_end = after_bracket
        .find(']')
        .ok_or_else(|| anyhow::anyhow!("missing vector end after {section}"))?;
    let values = after_bracket[..bracket_end]
        .split_whitespace()
        .map(|token| {
            token
                .parse::<f32>()
                .with_context(|| format!("invalid CMVN value {token:?} in {section}"))
        })
        .collect::<Result<Vec<_>>>()?;
    if values.is_empty() {
        bail!("empty CMVN vector in {section}");
    }
    Ok(values)
}

impl StreamingVadModel for FsmnVadSession {
    fn push_chunk(&mut self, chunk: &AudioChunk) -> Result<Vec<Annotation>> {
        if chunk.waveform.sample_rate != FSMN_VAD_SAMPLE_RATE {
            bail!(
                "FSMN VAD expects 16kHz mono audio, got sample_rate={}",
                chunk.waveform.sample_rate
            );
        }
        let segments = self.push_samples_f32(&chunk.waveform.samples)?;
        Ok(segments
            .into_iter()
            .map(|segment| {
                Annotation::new(
                    segment.range,
                    AnnotationPayload::Speech,
                    AnnotationSource::Model(FSMN_VAD_SOURCE.to_string()),
                    AnnotationStatus::Final,
                )
            })
            .collect())
    }

    fn finish(&mut self) -> Result<Vec<Annotation>> {
        Ok(self
            .finish_segments()?
            .into_iter()
            .map(|segment| {
                Annotation::new(
                    segment.range,
                    AnnotationPayload::Speech,
                    AnnotationSource::Model(FSMN_VAD_SOURCE.to_string()),
                    AnnotationStatus::Final,
                )
            })
            .collect())
    }
}

#[cfg(test)]
mod legacy_vad_segmenter {
    use std::collections::VecDeque;

    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum VadState {
        Waiting,
        Recording,
    }

    pub(super) struct VadSegmenter {
        options: VadOptions,
        state: VadState,
        history: VecDeque<i16>,
        speech_start: usize,
        silence_chunks: usize,
        current_chunks: usize,
        chunk_cursor: usize,
    }

    impl VadSegmenter {
        pub(super) fn new(options: VadOptions) -> Self {
            Self {
                options,
                state: VadState::Waiting,
                history: VecDeque::new(),
                speech_start: 0,
                silence_chunks: 0,
                current_chunks: 0,
                chunk_cursor: 0,
            }
        }

        pub(super) fn push_probability(&mut self, probability: f32) -> Vec<VadSegment> {
            let chunk_start = self.chunk_cursor * FSMN_VAD_CHUNK_SIZE;
            self.chunk_cursor += 1;
            match self.state {
                VadState::Waiting => {
                    self.push_history_chunk();
                    if probability >= self.options.threshold {
                        self.state = VadState::Recording;
                        self.speech_start = chunk_start.saturating_sub(self.history.len());
                        self.silence_chunks = 0;
                        self.current_chunks = 1;
                        self.history.clear();
                    }
                    Vec::new()
                }
                VadState::Recording => {
                    self.current_chunks += 1;
                    if probability >= self.options.threshold {
                        self.silence_chunks = 0;
                        if self.current_samples() >= ms_to_samples(9_000) {
                            return self.finalize(chunk_start + FSMN_VAD_CHUNK_SIZE, false);
                        }
                        return Vec::new();
                    }
                    self.silence_chunks += 1;
                    if self.silence_chunks * FSMN_VAD_CHUNK_SIZE
                        >= ms_to_samples(self.options.min_silence_ms)
                    {
                        return self.finalize(chunk_start + FSMN_VAD_CHUNK_SIZE, true);
                    }
                    Vec::new()
                }
            }
        }

        pub(super) fn finish(&mut self) -> Vec<VadSegment> {
            let end = self.chunk_cursor * FSMN_VAD_CHUNK_SIZE;
            self.finalize(end, false)
        }

        fn push_history_chunk(&mut self) {
            let rollback = ms_to_samples(200);
            self.history
                .extend(std::iter::repeat_n(0i16, FSMN_VAD_CHUNK_SIZE));
            while self.history.len() > rollback {
                self.history.pop_front();
            }
        }

        fn current_samples(&self) -> usize {
            self.current_chunks * FSMN_VAD_CHUNK_SIZE
        }

        fn finalize(&mut self, end: usize, trim_tail: bool) -> Vec<VadSegment> {
            if self.state != VadState::Recording {
                return Vec::new();
            }
            let end = if trim_tail {
                end.saturating_sub(self.silence_chunks * FSMN_VAD_CHUNK_SIZE)
            } else {
                end
            };
            let start = self.speech_start.min(end);
            let duration = end.saturating_sub(start);
            self.state = VadState::Waiting;
            self.silence_chunks = 0;
            self.current_chunks = 0;
            self.history.clear();
            if duration < ms_to_samples(self.options.min_speech_ms) {
                return Vec::new();
            }
            vec![VadSegment {
                range: sample_range(start, end),
                probability: self.options.threshold,
            }]
        }
    }
}

#[derive(Debug, Clone)]
struct WavFrontendConfig {
    sample_rate: i32,
    frame_length_ms: f32,
    frame_shift_ms: f32,
    n_mels: usize,
    lfr_m: usize,
    lfr_n: usize,
    cmvn_file: Option<PathBuf>,
}

impl Default for WavFrontendConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            frame_length_ms: 25.0,
            frame_shift_ms: 10.0,
            n_mels: 80,
            lfr_m: 7,
            lfr_n: 6,
            cmvn_file: None,
        }
    }
}

#[derive(Debug, Clone)]
struct WavFrontend {
    config: WavFrontendConfig,
    cmvn_means: Option<Array1<f32>>,
    cmvn_vars: Option<Array1<f32>>,
}

impl WavFrontend {
    fn new(config: WavFrontendConfig) -> Result<Self> {
        let (cmvn_means, cmvn_vars) = if let Some(cmvn_path) = &config.cmvn_file {
            let (means, vars) = Self::load_cmvn(cmvn_path)?;
            (Some(means), Some(vars))
        } else {
            (None, None)
        };
        Ok(Self {
            config,
            cmvn_means,
            cmvn_vars,
        })
    }

    fn load_cmvn(path: &Path) -> Result<(Array1<f32>, Array1<f32>)> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read CMVN file {}", path.display()))?;
        let means = extract_cmvn_vector(&text, "<AddShift>")
            .with_context(|| format!("failed to parse AddShift CMVN in {}", path.display()))?;
        let vars = extract_cmvn_vector(&text, "<Rescale>")
            .with_context(|| format!("failed to parse Rescale CMVN in {}", path.display()))?;
        if means.len() != vars.len() {
            bail!(
                "CMVN file {} has mismatched AddShift/Rescale dims: {} vs {}",
                path.display(),
                means.len(),
                vars.len()
            );
        }
        Ok((Array1::from_vec(means), Array1::from_vec(vars)))
    }

    fn extract_features(&self, waveform: &[i16]) -> Result<Array2<f32>> {
        self.extract_features_from_pcm(waveform)
    }

    fn extract_features_from_pcm(&self, samples: &[i16]) -> Result<Array2<f32>> {
        let waveform = samples
            .iter()
            .map(|&sample| f32::from(sample))
            .collect::<Vec<_>>();
        let fbank = self.compute_fbank_features(&waveform)?;
        let lfr = self.apply_lfr(&fbank);
        Ok(self.apply_cmvn(&lfr))
    }

    fn compute_fbank_features(&self, waveform: &[f32]) -> Result<Array2<f32>> {
        let opt = FbankOptions {
            frame_opts: FrameExtractionOptions {
                samp_freq: self.config.sample_rate as f32,
                window_type: CStr::from_bytes_with_nul(b"hamming\0")?.as_ptr(),
                dither: 0.0,
                frame_shift_ms: self.config.frame_shift_ms,
                frame_length_ms: self.config.frame_length_ms,
                snip_edges: true,
                ..Default::default()
            },
            mel_opts: MelBanksOptions {
                num_bins: self.config.n_mels as i32,
                ..Default::default()
            },
            energy_floor: 0.0,
            ..Default::default()
        };
        let mut fbank = OnlineFbank::new(opt);
        fbank.accept_waveform(self.config.sample_rate as f32, waveform);
        let frames = fbank.num_ready_frames();
        let mut out = Vec::with_capacity(frames as usize * self.config.n_mels);
        for i in 0..frames {
            let frame = fbank
                .get_frame(i)
                .ok_or_else(|| anyhow::anyhow!("missing fbank frame {i}"))?;
            out.extend_from_slice(frame);
        }
        Ok(Array2::from_shape_vec(
            (frames as usize, self.config.n_mels),
            out,
        )?)
    }

    fn apply_lfr(&self, fbank: &Array2<f32>) -> Array2<f32> {
        let t = fbank.shape()[0];
        if t == 0 {
            return Array2::zeros((0, self.config.n_mels * self.config.lfr_m));
        }
        let t_lfr = (t as f32 / self.config.lfr_n as f32).ceil() as usize;
        let left_padding_rows = (self.config.lfr_m - 1) / 2;
        let mut padded = Array2::zeros((t + left_padding_rows, fbank.shape()[1]));
        for i in 0..left_padding_rows {
            padded.slice_mut(s![i, ..]).assign(&fbank.slice(s![0, ..]));
        }
        for i in 0..t {
            padded
                .slice_mut(s![i + left_padding_rows, ..])
                .assign(&fbank.slice(s![i, ..]));
        }

        let feat_dim = self.config.n_mels * self.config.lfr_m;
        let mut lfr = Array2::zeros((t_lfr, feat_dim));
        for i in 0..t_lfr {
            let start = i * self.config.lfr_n;
            let end = (start + self.config.lfr_m).min(t + left_padding_rows);
            let frame = padded.slice(s![start..end, ..]);
            let flat = frame
                .into_shape_with_order(frame.len())
                .expect("LFR frame reshape");
            lfr.slice_mut(s![i, ..flat.len()]).assign(&flat);
            if end < start + self.config.lfr_m {
                let last_row = padded.slice(s![padded.shape()[0] - 1, ..]);
                for j in end - start..self.config.lfr_m {
                    lfr.slice_mut(s![i, j * self.config.n_mels..(j + 1) * self.config.n_mels])
                        .assign(&last_row);
                }
            }
        }
        lfr
    }

    fn apply_cmvn(&self, feats: &Array2<f32>) -> Array2<f32> {
        let (Some(means), Some(vars)) = (&self.cmvn_means, &self.cmvn_vars) else {
            return feats.to_owned();
        };
        let (frames, dim) = feats.dim();
        if means.len() != dim || vars.len() != dim {
            return feats.to_owned();
        }
        let means = means.broadcast((frames, dim)).expect("CMVN mean broadcast");
        let vars = vars.broadcast((frames, dim)).expect("CMVN var broadcast");
        (feats + &means) * vars
    }
}

fn download_fsmn_vad() -> Result<PathBuf> {
    let api = Api::new()?;
    let repo = api.model(FSMN_VAD_REPO.to_string());
    let model_path = repo.get("model.pt")?;
    repo.get("am.mvn")?;
    model_path
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| anyhow::anyhow!("downloaded FSMN VAD model path has no parent"))
}

fn sample_range(start: usize, end: usize) -> TimeRange {
    TimeRange::new(
        DurationMs(sample_to_ms(start)),
        DurationMs(sample_to_ms(end)),
    )
}

fn sample_to_ms(sample: usize) -> u64 {
    (sample as u64).saturating_mul(1000) / u64::from(FSMN_VAD_SAMPLE_RATE)
}

fn ms_to_samples(ms: u64) -> usize {
    (ms as usize).saturating_mul(FSMN_VAD_SAMPLE_RATE as usize) / 1000
}

fn speech_annotation(start: usize, end: usize, status: AnnotationStatus) -> Annotation {
    Annotation::new(
        sample_range(start, end),
        AnnotationPayload::Speech,
        AnnotationSource::Model(FSMN_VAD_SOURCE.to_string()),
        status,
    )
}

#[cfg(test)]
mod tests {
    use super::legacy_vad_segmenter::VadSegmenter;
    use super::*;

    #[test]
    fn segmenter_keeps_sessions_independent() {
        let options = VadOptions {
            threshold: 0.5,
            min_speech_ms: 1,
            min_silence_ms: 1,
            ..VadOptions::default()
        };
        let mut first = VadSegmenter::new(options.clone());
        let mut second = VadSegmenter::new(options);

        assert!(first.push_probability(0.9).is_empty());
        assert!(second.push_probability(0.1).is_empty());
        let first_segments = first.push_probability(0.1);
        let second_segments = second.finish();

        assert_eq!(first_segments.len(), 1);
        assert!(second_segments.is_empty());
    }

    #[test]
    fn segmenter_rolls_back_speech_start() {
        let options = VadOptions {
            threshold: 0.5,
            min_speech_ms: 1,
            min_silence_ms: 1,
            ..VadOptions::default()
        };
        let mut segmenter = VadSegmenter::new(options);

        segmenter.push_probability(0.1);
        segmenter.push_probability(0.1);
        segmenter.push_probability(0.9);
        let segments = segmenter.push_probability(0.1);

        assert_eq!(segments.len(), 1);
        assert!(segments[0].range.start.0 < sample_to_ms(2 * FSMN_VAD_CHUNK_SIZE));
    }

    #[test]
    #[ignore = "downloads funasr/fsmn-vad and uses local raw_audios fixture"]
    fn fsmn_vad_detects_local_raw_audio() -> Result<()> {
        let wav = std::path::Path::new("raw_audios/audio (13).wav");
        if !wav.exists() {
            return Ok(());
        }
        let waveform = load_pcm16_wav(wav)?;
        let model = FsmnVadModel::from_pretrained(None::<&str>)?;
        let segments = model.detect(&waveform, &VadOptions::default())?;
        assert!(
            !segments.is_empty(),
            "FSMN VAD should detect speech in {}",
            wav.display()
        );
        Ok(())
    }

    #[cfg(feature = "cuda-vad")]
    #[test]
    #[ignore = "requires CUDA toolkit and .cache/fsmn-vad"]
    fn forward_frame_scores_match_cpu_and_cuda() -> Result<()> {
        let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let model_dir = workspace.join(".cache/fsmn-vad");
        let wav = workspace.join("raw_audios/audio (13).wav");
        if !model_dir.exists() || !wav.exists() {
            return Ok(());
        }

        let cpu = FsmnVadWeights::load_on_device(&model_dir, &Device::Cpu)?;
        let cuda = FsmnVadWeights::load_on_device(&model_dir, &Device::new_cuda(0)?)?;
        let waveform = load_pcm16_wav(&wav)?;
        let pcm = samples_f32_to_i16(&waveform.samples);
        let feats = cpu.frontend.extract_features_from_pcm(&pcm)?;

        let mut cpu_caches = cpu.zero_caches()?;
        let mut cuda_caches = cuda.zero_caches()?;
        let cpu_scores = cpu.forward_frame_scores(&feats, &mut cpu_caches)?;
        let cuda_scores = cuda.forward_frame_scores(&feats, &mut cuda_caches)?;
        assert_eq!(cpu_scores.len(), cuda_scores.len());

        let mut max_diff = 0.0f32;
        let mut worst = (0usize, 0usize, 0.0f32, 0.0f32);
        for (frame_idx, (cpu_frame, cuda_frame)) in
            cpu_scores.iter().zip(cuda_scores.iter()).enumerate()
        {
            assert_eq!(cpu_frame.len(), cuda_frame.len());
            for (class_idx, (cpu_score, cuda_score)) in
                cpu_frame.iter().zip(cuda_frame.iter()).enumerate()
            {
                let diff = (cpu_score - cuda_score).abs();
                if diff > max_diff {
                    max_diff = diff;
                    worst = (frame_idx, class_idx, *cpu_score, *cuda_score);
                }
            }
        }
        println!(
            "forward_score_max_diff={max_diff} worst_frame={} worst_class={} cpu={} cuda={}",
            worst.0, worst.1, worst.2, worst.3
        );
        assert!(
            max_diff < 1e-4,
            "CPU/CUDA forward scores diverged: max_diff={max_diff}"
        );
        Ok(())
    }

    #[cfg(feature = "cuda-vad")]
    #[test]
    #[ignore = "requires CUDA toolkit and raw_audios fixture"]
    fn detect_segments_match_cpu_and_cuda_on_raw_audios() -> Result<()> {
        let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let model_dir = workspace.join(".cache/fsmn-vad");
        let audio_dir = workspace.join("raw_audios");
        if !model_dir.exists() || !audio_dir.is_dir() {
            return Ok(());
        }

        let mut cpu_segments = Vec::new();
        let mut cuda_segments = Vec::new();
        for device in ["cpu", "cuda"] {
            // SAFETY: this ignored test is single-threaded.
            unsafe { std::env::set_var("VASR_VAD_DEVICE", device) };
            let model = FsmnVadModel::from_pretrained(Some(model_dir.as_path()))?;
            let mut files = std::fs::read_dir(&audio_dir)?
                .map(|entry| entry.map(|entry| entry.path()))
                .collect::<std::io::Result<Vec<_>>>()?;
            files.retain(|path| path.extension().and_then(|ext| ext.to_str()) == Some("wav"));
            files.sort();
            let mut ranges = Vec::new();
            for path in files {
                let waveform = load_pcm16_wav(&path)?;
                for segment in model.detect(&waveform, &VadOptions::default())? {
                    ranges.push((segment.range.start.0, segment.range.end.0));
                }
            }
            if device == "cpu" {
                cpu_segments = ranges;
            } else {
                cuda_segments = ranges;
            }
        }
        assert_eq!(cpu_segments, cuda_segments);
        Ok(())
    }

    #[test]
    #[ignore = "benchmark FSMN VAD on raw_audios; set VASR_VAD_DEVICE=cpu|cuda"]
    fn fsmn_vad_benchmark_raw_audios() -> Result<()> {
        let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let audio_dir = workspace.join("raw_audios");
        let model_dir = workspace.join(".cache/fsmn-vad");
        if !audio_dir.is_dir() || !model_dir.exists() {
            return Ok(());
        }
        let mut files = std::fs::read_dir(audio_dir)?
            .map(|entry| entry.map(|entry| entry.path()))
            .collect::<std::io::Result<Vec<_>>>()?;
        files.retain(|path| path.extension().and_then(|ext| ext.to_str()) == Some("wav"));
        files.sort();

        let model = FsmnVadModel::from_pretrained(Some(model_dir.as_path()))?;
        let options = VadOptions::default();
        let bench_start = Instant::now();
        let mut audio_seconds = 0.0f64;
        let mut segments = 0usize;
        for path in &files {
            let waveform = load_pcm16_wav(path)?;
            audio_seconds += waveform.duration_seconds();
            segments += model.detect(&waveform, &options)?.len();
        }
        let vad_seconds = bench_start.elapsed().as_secs_f64();
        println!(
            "fsmn_vad_bench files={} audio_seconds={audio_seconds:.3} segments={segments} vad_seconds={vad_seconds:.3} speedup={:.1}x rtf={:.4}",
            files.len(),
            audio_seconds / vad_seconds,
            vad_seconds / audio_seconds
        );
        Ok(())
    }

    #[derive(serde::Deserialize)]
    struct FasrVadFixtureFile {
        name: String,
        segments: Vec<[i32; 2]>,
    }

    #[derive(serde::Deserialize)]
    struct FasrVadFixture {
        files: Vec<FasrVadFixtureFile>,
    }

    fn assert_vad_matches_fasr_fixture() -> Result<()> {
        let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let audio_dir = workspace.join("raw_audios");
        let model_dir = workspace.join(".cache/fsmn-vad");
        let fixture_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/fsmn_vad_fasr_segments.json");
        if !audio_dir.is_dir() || !model_dir.exists() || !fixture_path.exists() {
            return Ok(());
        }

        let fixture: FasrVadFixture =
            serde_json::from_str(&std::fs::read_to_string(&fixture_path)?)?;
        let model = FsmnVadModel::from_pretrained(Some(model_dir.as_path()))?;
        let options = VadOptions::default();

        for expected_file in fixture.files {
            let wav = audio_dir.join(&expected_file.name);
            if !wav.exists() {
                continue;
            }
            let waveform = load_pcm16_wav(&wav)?;
            let segments = model.detect(&waveform, &options)?;
            assert_eq!(
                segments.len(),
                expected_file.segments.len(),
                "segment count mismatch for {}",
                expected_file.name
            );
            for (segment, [start, end]) in segments.iter().zip(expected_file.segments) {
                assert_eq!(
                    segment.range.start.0,
                    u64::try_from(start).expect("negative fasr segment start"),
                    "start mismatch for {}",
                    expected_file.name
                );
                assert_eq!(
                    segment.range.end.0,
                    u64::try_from(end).expect("negative fasr segment end"),
                    "end mismatch for {}",
                    expected_file.name
                );
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "uses local FSMN VAD, raw_audios fixture, and funasr_onnx reference JSON"]
    fn fsmn_vad_matches_fasr_funasr_onnx_segments_all_raw_audios() -> Result<()> {
        assert_vad_matches_fasr_fixture()
    }

    #[cfg(feature = "cuda-vad")]
    #[test]
    #[ignore = "requires CUDA toolkit, raw_audios fixture, and funasr_onnx reference JSON"]
    fn fsmn_vad_cuda_matches_fasr_funasr_onnx_segments_all_raw_audios() -> Result<()> {
        // SAFETY: this ignored test is single-threaded.
        unsafe { std::env::set_var("VASR_VAD_DEVICE", "cuda") };
        assert_vad_matches_fasr_fixture()
    }

    #[test]
    #[ignore = "uses local FSMN VAD and raw_audios fixture"]
    fn offline_fast_path_matches_streaming_session_segments() -> Result<()> {
        let wav = std::path::Path::new("raw_audios/audio (13).wav");
        let model_dir = std::path::Path::new(".cache/fsmn-vad");
        if !wav.exists() || !model_dir.exists() {
            return Ok(());
        }
        let waveform = load_pcm16_wav(wav)?;
        let model = FsmnVadModel::from_pretrained(Some(model_dir))?;
        let options = VadOptions::default();

        let fast = model.detect_with_timing(&waveform, &options)?.segments;
        let mut session = model.new_session(&options)?;
        session.push_samples_f32(&waveform.samples)?;
        let streaming = session.finish_segments()?;

        assert_eq!(fast, streaming);
        Ok(())
    }

    #[test]
    fn split_full_chunks_keeps_unprocessed_tail() {
        let samples = (0..(FSMN_VAD_CHUNK_SIZE * 2 + 17))
            .map(|sample| sample as i16)
            .collect::<Vec<_>>();

        let (full_chunks, tail) = split_full_chunks(&samples);
        let chunks = full_chunks
            .chunks_exact(FSMN_VAD_CHUNK_SIZE)
            .collect::<Vec<_>>();

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0][0], 0);
        assert_eq!(chunks[1][0], FSMN_VAD_CHUNK_SIZE as i16);
        assert_eq!(tail.len(), 17);
        assert_eq!(tail[0], (FSMN_VAD_CHUNK_SIZE * 2) as i16);
    }

    #[test]
    fn cmvn_parser_reads_addshift_and_rescale_vectors() -> Result<()> {
        let path = std::env::temp_dir().join(format!("vasr-test-am-{}.mvn", std::process::id()));
        std::fs::write(
            &path,
            "<Nnet>\n\
             <Splice> 2 2\n\
             [ 0 ]\n\
             <AddShift> 2 2\n\
             <LearnRateCoef> 0 [ -1.5 -2.5 ]\n\
             <Rescale> 2 2\n\
             <LearnRateCoef> 0 [ 0.25 0.5 ]\n\
             </Nnet>\n",
        )?;

        let (means, vars) = WavFrontend::load_cmvn(&path)?;

        assert_eq!(means.to_vec(), vec![-1.5, -2.5]);
        assert_eq!(vars.to_vec(), vec![0.25, 0.5]);
        let _ = std::fs::remove_file(path);
        Ok(())
    }

    fn load_pcm16_wav(path: &std::path::Path) -> Result<Waveform> {
        let data = std::fs::read(path)?;
        if data.len() < 44 || &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" {
            bail!("not a RIFF/WAVE file: {}", path.display());
        }
        let mut offset = 12usize;
        let mut sample_rate = None;
        let mut channels = None;
        let mut data_range = None;
        while offset + 8 <= data.len() {
            let id = &data[offset..offset + 4];
            let size = u32::from_le_bytes(data[offset + 4..offset + 8].try_into()?) as usize;
            offset += 8;
            if offset + size > data.len() {
                break;
            }
            match id {
                b"fmt " if size >= 16 => {
                    let format = u16::from_le_bytes(data[offset..offset + 2].try_into()?);
                    channels = Some(u16::from_le_bytes(data[offset + 2..offset + 4].try_into()?));
                    sample_rate =
                        Some(u32::from_le_bytes(data[offset + 4..offset + 8].try_into()?));
                    let bits = u16::from_le_bytes(data[offset + 14..offset + 16].try_into()?);
                    if format != 1 || bits != 16 {
                        bail!("expected PCM16 WAV, got format={format} bits={bits}");
                    }
                }
                b"data" => data_range = Some(offset..offset + size),
                _ => {}
            }
            offset += size + (size % 2);
        }
        let sample_rate = sample_rate.ok_or_else(|| anyhow::anyhow!("WAV missing fmt chunk"))?;
        let channels = channels.ok_or_else(|| anyhow::anyhow!("WAV missing channel count"))?;
        let data_range = data_range.ok_or_else(|| anyhow::anyhow!("WAV missing data chunk"))?;
        let mut samples = Vec::new();
        for frame in data[data_range].chunks_exact(usize::from(channels) * 2) {
            let mut sum = 0f32;
            for ch in 0..usize::from(channels) {
                let start = ch * 2;
                sum += f32::from(i16::from_le_bytes(frame[start..start + 2].try_into()?));
            }
            samples.push(sum / f32::from(channels) / f32::from(i16::MAX));
        }
        Ok(Waveform::new(samples, sample_rate))
    }
}
