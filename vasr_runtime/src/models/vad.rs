use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{Context, Result, bail};
use ort::session::Session;
use ort::value::Tensor;
use vasr_data::{
    Annotation, AnnotationPayload, AnnotationSource, AnnotationStatus, AudioChunk, DurationMs,
    TimeRange, Timeline, Waveform,
};

use crate::model::{StreamingVadModel, VadModel, VadOptions, VadSegment};

const SILERO_SAMPLE_RATE: u32 = 16_000;
const SILERO_WINDOW_SAMPLES: usize = 512;

pub struct SileroVadModel {
    path: PathBuf,
    inner: Mutex<SileroOnnx>,
}

impl std::fmt::Debug for SileroVadModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SileroVadModel")
            .field("path", &self.path)
            .finish_non_exhaustive()
    }
}

impl SileroVadModel {
    pub fn from_onnx(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let inner = SileroOnnx::new(&path)?;
        Ok(Self {
            path,
            inner: Mutex::new(inner),
        })
    }

    pub fn from_default_model() -> Result<Self> {
        let path = default_silero_model_path().ok_or_else(|| {
            anyhow::anyhow!(
                "Silero VAD ONNX model was not found in local caches; pass --vad-model /path/to/silero_vad.onnx"
            )
        })?;
        Self::from_onnx(path)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn annotate(&self, waveform: &Waveform, options: &VadOptions) -> Timeline {
        let mut timeline = Timeline::new("vad_audio");
        if let Ok(segments) = self.detect(waveform, options) {
            for segment in segments {
                timeline.push(Annotation::new(
                    segment.range,
                    AnnotationPayload::Speech,
                    AnnotationSource::Model("silero_vad".to_string()),
                    AnnotationStatus::Final,
                ));
            }
        }
        timeline
    }
}

impl VadModel for SileroVadModel {
    fn detect(&self, waveform: &Waveform, options: &VadOptions) -> Result<Vec<VadSegment>> {
        let mut engine = self
            .inner
            .lock()
            .map_err(|_| anyhow::anyhow!("Silero VAD mutex poisoned"))?;
        engine.reset();
        detect_segments(waveform, options, |window| engine.predict(window))
    }

    fn start_stream(&self, options: &VadOptions) -> Result<Box<dyn StreamingVadModel>> {
        Ok(Box::new(SileroStreamingVadModel {
            engine: SileroOnnx::new(&self.path)?,
            options: options.clone(),
            buffer: Vec::new(),
            sample_cursor: 0,
            in_speech: false,
            speech_start_sample: 0,
            silence_samples: 0,
            last_range: None,
        }))
    }
}

pub struct SileroStreamingVadModel {
    engine: SileroOnnx,
    options: VadOptions,
    buffer: Vec<f32>,
    sample_cursor: usize,
    in_speech: bool,
    speech_start_sample: usize,
    silence_samples: usize,
    last_range: Option<TimeRange>,
}

impl StreamingVadModel for SileroStreamingVadModel {
    fn push_chunk(&mut self, chunk: &AudioChunk) -> Result<Vec<Annotation>> {
        if chunk.waveform.sample_rate != SILERO_SAMPLE_RATE {
            bail!(
                "Silero VAD expects 16kHz mono audio, got sample_rate={}",
                chunk.waveform.sample_rate
            );
        }

        self.buffer.extend_from_slice(&chunk.waveform.samples);
        self.last_range = Some(chunk.range);
        let mut annotations = Vec::new();

        while self.buffer.len() >= SILERO_WINDOW_SAMPLES {
            let window = self
                .buffer
                .drain(..SILERO_WINDOW_SAMPLES)
                .collect::<Vec<_>>();
            let window_start = self.sample_cursor;
            self.sample_cursor += SILERO_WINDOW_SAMPLES;
            let probability = self.engine.predict(&window)?;
            self.apply_probability(window_start, probability, &mut annotations);
        }

        Ok(annotations)
    }

    fn finish(&mut self) -> Result<Vec<Annotation>> {
        let mut annotations = Vec::new();
        if self.in_speech {
            let end_sample = self
                .last_range
                .map(|range| ms_to_samples(range.end.0))
                .unwrap_or(self.sample_cursor);
            annotations.push(speech_annotation(
                self.speech_start_sample,
                end_sample,
                AnnotationStatus::Final,
            ));
            self.in_speech = false;
            self.silence_samples = 0;
        }
        self.engine.reset();
        Ok(annotations)
    }
}

impl SileroStreamingVadModel {
    fn apply_probability(
        &mut self,
        window_start: usize,
        probability: f32,
        annotations: &mut Vec<Annotation>,
    ) {
        if probability >= self.options.threshold {
            self.silence_samples = 0;
            if !self.in_speech {
                self.in_speech = true;
                self.speech_start_sample = window_start;
                annotations.push(speech_annotation(
                    window_start,
                    window_start + SILERO_WINDOW_SAMPLES,
                    AnnotationStatus::Partial,
                ));
            }
            return;
        }

        if !self.in_speech {
            annotations.push(Annotation::new(
                sample_range(window_start, window_start + SILERO_WINDOW_SAMPLES),
                AnnotationPayload::Silence,
                AnnotationSource::Model("silero_vad".to_string()),
                AnnotationStatus::Partial,
            ));
            return;
        }

        self.silence_samples += SILERO_WINDOW_SAMPLES;
        let min_silence_samples = ms_to_samples(self.options.min_silence_ms);
        if self.silence_samples >= min_silence_samples {
            let end_sample = window_start + SILERO_WINDOW_SAMPLES;
            annotations.push(speech_annotation(
                self.speech_start_sample,
                end_sample,
                AnnotationStatus::Final,
            ));
            self.in_speech = false;
            self.silence_samples = 0;
        }
    }
}

struct SileroOnnx {
    session: Session,
    state: Vec<f32>,
}

impl SileroOnnx {
    fn new(path: &Path) -> Result<Self> {
        if !path.exists() {
            bail!("Silero VAD ONNX model does not exist: {}", path.display());
        }
        let session = Session::builder()
            .context("failed to create ONNX Runtime session builder")?
            .commit_from_file(path)
            .with_context(|| format!("failed to load Silero VAD ONNX model {}", path.display()))?;
        Ok(Self {
            session,
            state: vec![0.0; 2 * 128],
        })
    }

    fn reset(&mut self) {
        self.state.fill(0.0);
    }

    fn predict(&mut self, window: &[f32]) -> Result<f32> {
        if window.len() != SILERO_WINDOW_SAMPLES {
            bail!(
                "Silero VAD window must contain {SILERO_WINDOW_SAMPLES} samples, got {}",
                window.len()
            );
        }

        let frame = Tensor::from_array(([1usize, SILERO_WINDOW_SAMPLES], window.to_vec()))
            .context("failed to create Silero input tensor")?;
        let state = Tensor::from_array(([2usize, 1, 128], self.state.clone()))
            .context("failed to create Silero state tensor")?;
        let sample_rate = Tensor::from_array(([1usize], vec![i64::from(SILERO_SAMPLE_RATE)]))
            .context("failed to create Silero sample-rate tensor")?;
        let inputs = ort::inputs![frame, state, sample_rate,];
        let outputs = self.session.run(inputs)?;
        let (_, state) = outputs["stateN"].try_extract_tensor::<f32>()?;
        self.state = state.to_vec();
        let (_, raw_output) = outputs["output"].try_extract_tensor::<f32>()?;
        raw_output
            .first()
            .copied()
            .ok_or_else(|| anyhow::anyhow!("Silero VAD output tensor is empty"))
    }
}

fn detect_segments(
    waveform: &Waveform,
    options: &VadOptions,
    mut predict: impl FnMut(&[f32]) -> Result<f32>,
) -> Result<Vec<VadSegment>> {
    if waveform.sample_rate != SILERO_SAMPLE_RATE {
        bail!(
            "Silero VAD expects 16kHz mono audio, got sample_rate={}",
            waveform.sample_rate
        );
    }

    let mut segments = Vec::new();
    let mut in_speech = false;
    let mut speech_start = 0usize;
    let mut silence_samples = 0usize;
    let min_silence_samples = ms_to_samples(options.min_silence_ms);
    let min_speech_samples = ms_to_samples(options.min_speech_ms);

    for (i, window) in waveform.samples.chunks(SILERO_WINDOW_SAMPLES).enumerate() {
        if window.len() < SILERO_WINDOW_SAMPLES {
            break;
        }
        let window_start = i * SILERO_WINDOW_SAMPLES;
        let probability = predict(window)?;
        if probability >= options.threshold {
            silence_samples = 0;
            if !in_speech {
                in_speech = true;
                speech_start = window_start;
            }
        } else if in_speech {
            silence_samples += SILERO_WINDOW_SAMPLES;
            if silence_samples >= min_silence_samples {
                let end = window_start + SILERO_WINDOW_SAMPLES;
                if end.saturating_sub(speech_start) >= min_speech_samples {
                    segments.push(VadSegment {
                        range: sample_range(speech_start, end),
                        probability: options.threshold,
                    });
                }
                in_speech = false;
                silence_samples = 0;
            }
        }
    }

    if in_speech {
        let end = waveform.samples.len();
        if end.saturating_sub(speech_start) >= min_speech_samples {
            segments.push(VadSegment {
                range: sample_range(speech_start, end),
                probability: options.threshold,
            });
        }
    }

    Ok(segments)
}

fn default_silero_model_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME").map(PathBuf::from)?;
    [
        ".cache/torch/hub/snakers4_silero-vad_master/src/silero_vad/data/silero_vad_16k_op15.onnx",
        ".cache/torch/hub/snakers4_silero-vad_master/src/silero_vad/data/silero_vad.onnx",
        ".cache/modelscope/hub/pengzhendong/silero-vad/silero_vad.onnx",
        ".cache/modelscope/hub/models/pengzhendong/silero-vad/silero_vad.onnx",
        ".cache/uv/archive-v0/9bCd8qfnqS8TY0Afotnkt/livekit/plugins/silero/resources/silero_vad.onnx",
    ]
    .into_iter()
    .map(|relative| home.join(relative))
    .find(|path| path.exists())
}

fn sample_range(start: usize, end: usize) -> TimeRange {
    TimeRange::new(
        DurationMs(sample_to_ms(start)),
        DurationMs(sample_to_ms(end)),
    )
}

fn sample_to_ms(sample: usize) -> u64 {
    (sample as u64).saturating_mul(1000) / u64::from(SILERO_SAMPLE_RATE)
}

fn ms_to_samples(ms: u64) -> usize {
    (ms as usize).saturating_mul(SILERO_SAMPLE_RATE as usize) / 1000
}

fn speech_annotation(start: usize, end: usize, status: AnnotationStatus) -> Annotation {
    Annotation::new(
        sample_range(start, end),
        AnnotationPayload::Speech,
        AnnotationSource::Model("silero_vad".to_string()),
        status,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_silero_onnx_runs_when_cached() -> Result<()> {
        let Some(path) = default_silero_model_path() else {
            return Ok(());
        };

        let model = SileroVadModel::from_onnx(path)?;
        let waveform = Waveform::new(vec![0.0; SILERO_SAMPLE_RATE as usize], SILERO_SAMPLE_RATE);
        let segments = model.detect(&waveform, &VadOptions::default())?;

        assert!(segments.is_empty());
        Ok(())
    }

    #[test]
    fn streaming_silero_onnx_runs_when_cached() -> Result<()> {
        let Some(path) = default_silero_model_path() else {
            return Ok(());
        };

        let model = SileroVadModel::from_onnx(path)?;
        let mut stream = model.start_stream(&VadOptions::default())?;
        let chunk = AudioChunk {
            stream_id: "test".to_string(),
            waveform: Waveform::new(vec![0.0; SILERO_WINDOW_SAMPLES], SILERO_SAMPLE_RATE),
            is_start: true,
            is_last: false,
            range: sample_range(0, SILERO_WINDOW_SAMPLES),
        };

        let annotations = stream.push_chunk(&chunk)?;
        assert_eq!(annotations.len(), 1);
        assert!(matches!(annotations[0].payload, AnnotationPayload::Silence));
        assert!(stream.finish()?.is_empty());
        Ok(())
    }
}
