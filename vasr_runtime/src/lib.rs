//! vASR runtime model traits and pipelines.

pub mod model;
pub mod models;
pub mod pipeline;

pub use model::{
    AsrModel, AsrOptions, StreamingAsrModel, StreamingVadModel, VadModel, VadOptions, VadSegment,
};
pub use models::{qwen3_asr::Qwen3AsrModel, vad::SileroVadModel};
#[cfg(feature = "async")]
pub use pipeline::r#async::{AsyncOfflinePipeline, ParallelTranscribeOptions};
pub use pipeline::{OfflinePipeline, RealtimePipeline};
