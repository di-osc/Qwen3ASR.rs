//! HTTP and websocket service layer for vASR.

pub mod realtime;
pub mod transcribe;

pub use realtime::{RealtimeService, RealtimeSession, realtime_router};
pub use transcribe::{TranscribeService, transcribe_router};
