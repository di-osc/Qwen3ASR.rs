//! vASR data model types shared by offline and realtime inference.

mod cer;
mod extract_audio;
mod fasr_audio_list;
mod media;
mod record;
mod segment;
mod stream;
mod time;
mod timeline;
mod token;
mod waveform;

pub use cer::{CerStats, compute_cer, normalize_for_cer};
pub use extract_audio::{
    ExtractAudioSummary, extract_embedded_audio, extract_embedded_audio_from_msgpack,
};
pub use fasr_audio_list::{
    FasrAudioListError, FasrAudioListSummary, convert_fasr_audio_list_file,
    convert_fasr_audio_list_to_vasr_records, inspect_fasr_audio_list,
};
pub use media::{AudioChannel, AudioFormat, AudioSource, MediaId};
pub use record::{
    AudioAsset, AudioEncoding, PersistedAudioFormat, RecordError, VasrRecord, VasrRecordList,
    WaveformCache,
};
pub use segment::{TextSegment, Transcript};
pub use stream::{AudioBytesStream, AudioChunk, AudioChunkList};
pub use time::{DurationMs, SampleIndex, TimeRange};
pub use timeline::{
    AcousticEvent, Annotation, AnnotationId, AnnotationPayload, AnnotationSource, AnnotationStatus,
    Diagnostic, HotwordMatch, LanguageTag, SpeakerId, Timeline, TimelineId,
};
pub use token::Token;
pub use waveform::{Waveform, WaveformError};
