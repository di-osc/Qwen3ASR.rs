use anyhow::Result;
use vasr_data::{Annotation, AudioChunk, TimeRange, Timeline, Waveform};

#[derive(Debug, Clone)]
pub struct AsrOptions {
    pub language: Option<String>,
    pub context: String,
    pub max_new_tokens: usize,
    pub max_batch_size: usize,
    pub max_batch_audio_sec: f32,
    pub chunk_max_sec: Option<f32>,
}

impl Default for AsrOptions {
    fn default() -> Self {
        Self {
            language: None,
            context: String::new(),
            max_new_tokens: 256,
            max_batch_size: 0,
            max_batch_audio_sec: 180.0,
            chunk_max_sec: None,
        }
    }
}

pub trait AsrModel: Send + Sync {
    fn transcribe(&self, waveform: &Waveform, options: &AsrOptions) -> Result<Timeline>;

    fn transcribe_batch(
        &self,
        waveforms: &[Waveform],
        options: &AsrOptions,
    ) -> Result<Vec<Timeline>> {
        waveforms
            .iter()
            .map(|waveform| self.transcribe(waveform, options))
            .collect()
    }

    fn start_stream(&self, options: &AsrOptions) -> Result<Box<dyn StreamingAsrModel>>;
}

pub trait StreamingAsrModel: Send {
    fn push_chunk(&mut self, chunk: &AudioChunk) -> Result<Vec<Annotation>>;

    fn finish(&mut self) -> Result<Vec<Annotation>>;
}

#[derive(Debug, Clone)]
pub struct VadOptions {
    /// `speech_noise_thres` in funasr_onnx / fasr (default 0.6).
    pub threshold: f32,
    /// Kept for API compatibility; funasr E2E VAD uses window thresholds instead.
    pub min_speech_ms: u64,
    /// `max_end_silence_time` in funasr_onnx / fasr (default 500).
    pub min_silence_ms: u64,
    /// Merge adjacent offline VAD segments across short gaps before ASR.
    pub merge_max_gap_ms: u64,
    /// Maximum merged offline ASR slice duration.
    pub merge_max_segment_ms: u64,
}

impl Default for VadOptions {
    fn default() -> Self {
        Self {
            threshold: 0.6,
            min_speech_ms: 250,
            min_silence_ms: 500,
            merge_max_gap_ms: 2_000,
            merge_max_segment_ms: 30_000,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::VadOptions;

    #[test]
    fn vad_defaults_match_fasr_fsmn() {
        let opts = VadOptions::default();
        assert_eq!(opts.threshold, 0.6);
        assert_eq!(opts.min_silence_ms, 500);
        assert_eq!(opts.merge_max_gap_ms, 2_000);
        assert_eq!(opts.merge_max_segment_ms, 30_000);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct VadSegment {
    pub range: TimeRange,
    pub probability: f32,
}

pub trait VadModel: Send + Sync {
    fn detect(&self, waveform: &Waveform, options: &VadOptions) -> Result<Vec<VadSegment>>;

    fn start_stream(&self, options: &VadOptions) -> Result<Box<dyn StreamingVadModel>>;
}

pub trait StreamingVadModel: Send {
    fn push_chunk(&mut self, chunk: &AudioChunk) -> Result<Vec<Annotation>>;

    fn finish(&mut self) -> Result<Vec<Annotation>>;
}
