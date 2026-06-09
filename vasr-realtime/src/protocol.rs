use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ClientRealtimeEvent {
    #[serde(rename = "session.update")]
    SessionUpdate {
        sample_rate: Option<u32>,
        input_audio_format: Option<String>,
    },
    #[serde(rename = "input_audio_buffer.append")]
    AudioAppend { audio: String },
    #[serde(rename = "input_audio_buffer.commit")]
    AudioCommit,
    #[serde(rename = "session.finish")]
    SessionFinish,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ServerRealtimeEvent {
    #[serde(rename = "session.created")]
    SessionCreated { session_id: String },
    #[serde(rename = "speech_started")]
    SpeechStarted { start_ms: u64 },
    #[serde(rename = "speech_stopped")]
    SpeechStopped { start_ms: u64, end_ms: u64 },
    #[serde(rename = "conversation.item.input_audio_transcription.text")]
    TranscriptionText { text: String },
    #[serde(rename = "conversation.item.input_audio_transcription.completed")]
    TranscriptionCompleted { text: String },
    #[serde(rename = "session.finished")]
    SessionFinished,
    #[serde(rename = "error")]
    Error { message: String },
}
