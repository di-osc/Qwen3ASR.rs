//! vASR runtime model traits and pipelines.

pub mod device;
pub mod model;
pub mod models;
pub mod pipeline;
pub mod scheduler;

pub use device::{auto_device, auto_dtype, device_label, resolve_device};
pub use model::{
    AsrModel, AsrOptions, StreamingAsrModel, StreamingVadModel, VadModel, VadOptions, VadSegment,
};
pub use models::{
    qwen3_asr::Qwen3AsrModel,
    vad::{FsmnVadDetection, FsmnVadModel, FsmnVadTiming},
};
#[cfg(feature = "async")]
pub use pipeline::r#async::{AsyncOfflinePipeline, ParallelTranscribeOptions};
pub use pipeline::{OfflinePipeline, RealtimePipeline};
pub use scheduler::{InferencePriority, InferenceScheduler, ScheduledAsrModel};
