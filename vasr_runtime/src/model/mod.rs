use anyhow::Result;
use vasr_data::{Annotation, AudioChunk, TimeRange, Timeline, Waveform};

#[derive(Debug, Clone)]
pub struct AsrOptions {
    pub language: Option<String>,
    pub context: String,
    pub max_new_tokens: usize,
}

impl Default for AsrOptions {
    fn default() -> Self {
        Self {
            language: None,
            context: String::new(),
            max_new_tokens: 256,
        }
    }
}

pub trait AsrModel: Send + Sync {
    fn transcribe(&self, waveform: &Waveform, options: &AsrOptions) -> Result<Timeline>;

    fn start_stream(&self, options: &AsrOptions) -> Result<Box<dyn StreamingAsrModel>>;
}

pub trait StreamingAsrModel: Send {
    fn push_chunk(&mut self, chunk: &AudioChunk) -> Result<Vec<Annotation>>;

    fn finish(&mut self) -> Result<Vec<Annotation>>;
}

#[derive(Debug, Clone)]
pub struct VadOptions {
    pub threshold: f32,
    pub min_speech_ms: u64,
    pub min_silence_ms: u64,
}

impl Default for VadOptions {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            min_speech_ms: 250,
            min_silence_ms: 300,
        }
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
