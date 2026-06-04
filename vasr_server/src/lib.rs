//! HTTP and websocket service layer for vASR.

pub mod async_transcribe;
pub mod realtime;
pub mod scheduler;
pub mod transcribe;

pub use async_transcribe::{AsyncTranscribePipeline, TranscribeInput, TranscribeItemOutcome};
pub use realtime::{RealtimeService, RealtimeSession, realtime_router};
pub use scheduler::{InferencePriority, InferenceScheduler, ScheduledAsrModel};
pub use transcribe::{
    TranscribeService, build_transcribe_service, build_transcribe_service_from_parts,
    transcribe_router,
};
