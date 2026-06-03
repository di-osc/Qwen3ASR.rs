//! vASR data model types shared by offline and realtime inference.

mod media;
mod segment;
mod stream;
mod time;
mod timeline;
mod token;
mod waveform;

pub use media::{AudioChannel, AudioFormat, AudioSource, MediaId};
pub use segment::{TextSegment, Transcript};
pub use stream::{AudioBytesStream, AudioChunk, AudioChunkList};
pub use time::{DurationMs, SampleIndex, TimeRange};
pub use timeline::{
    AcousticEvent, Annotation, AnnotationId, AnnotationPayload, AnnotationSource, AnnotationStatus,
    Diagnostic, HotwordMatch, LanguageTag, SpeakerId, Timeline, TimelineId,
};
pub use token::Token;
pub use waveform::{Waveform, WaveformError};
