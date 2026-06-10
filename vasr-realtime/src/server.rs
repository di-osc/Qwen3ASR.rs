use crate::protocol::{ClientRealtimeEvent, ServerRealtimeEvent};
use base64::Engine;
use futures_util::StreamExt;
use vasr_data::{AnnotationPayload, AnnotationStatus, AudioBytesStream};
use vasr_runtime::RealtimePipeline;

use std::sync::Arc;

use axum::{
    Router,
    extract::{
        State,
        ws::{Message, WebSocket, WebSocketUpgrade},
    },
    response::IntoResponse,
    routing::get,
};

pub struct RealtimeService {
    pub make_session: Arc<dyn Fn() -> anyhow::Result<RealtimeSession> + Send + Sync>,
}

pub fn realtime_router(service: Arc<RealtimeService>) -> Router {
    Router::new()
        .route("/v1/realtime", get(handle_realtime))
        .route("/api-ws/v1/realtime", get(handle_realtime))
        .with_state(service)
}

async fn handle_realtime(
    State(service): State<Arc<RealtimeService>>,
    ws: WebSocketUpgrade,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| async move {
        match (service.make_session)() {
            Ok(session) => run_socket(socket, session).await,
            Err(err) => {
                let mut socket = socket;
                let event = ServerRealtimeEvent::Error {
                    message: err.to_string(),
                };
                let _ = socket
                    .send(Message::Text(
                        serde_json::to_string(&event).unwrap_or_default().into(),
                    ))
                    .await;
            }
        }
    })
}

async fn run_socket(mut socket: WebSocket, mut session: RealtimeSession) {
    let session_id = format!("sess_{}", uuid::Uuid::new_v4().simple());
    let created = ServerRealtimeEvent::SessionCreated { session_id };
    let _ = socket
        .send(Message::Text(
            serde_json::to_string(&created).unwrap_or_default().into(),
        ))
        .await;

    while let Some(message) = socket.next().await {
        let Ok(message) = message else {
            break;
        };
        match message {
            Message::Text(text) => {
                let Ok(event) = serde_json::from_str::<ClientRealtimeEvent>(&text) else {
                    let err = ServerRealtimeEvent::Error {
                        message: "invalid realtime event".to_string(),
                    };
                    let _ = socket
                        .send(Message::Text(
                            serde_json::to_string(&err).unwrap_or_default().into(),
                        ))
                        .await;
                    continue;
                };
                let events = match event {
                    ClientRealtimeEvent::SessionUpdate { .. } => Vec::new(),
                    ClientRealtimeEvent::AudioAppend { audio } => {
                        session.append_base64_audio(&audio)
                    }
                    ClientRealtimeEvent::AudioCommit => Vec::new(),
                    ClientRealtimeEvent::SessionFinish => session.finish(),
                };
                for event in events {
                    if socket
                        .send(Message::Text(
                            serde_json::to_string(&event).unwrap_or_default().into(),
                        ))
                        .await
                        .is_err()
                    {
                        return;
                    }
                }
            }
            Message::Close(_) => break,
            _ => {}
        }
    }
}

pub struct RealtimeSession {
    stream: AudioBytesStream,
    pipeline: RealtimePipeline,
}

impl RealtimeSession {
    pub fn new(sample_rate: u32, pipeline: RealtimePipeline) -> Self {
        Self {
            stream: AudioBytesStream::new(sample_rate, 1, 100),
            pipeline,
        }
    }

    pub fn append_base64_audio(&mut self, audio: &str) -> Vec<ServerRealtimeEvent> {
        let bytes = match base64::engine::general_purpose::STANDARD.decode(audio) {
            Ok(bytes) => bytes,
            Err(err) => {
                return vec![ServerRealtimeEvent::Error {
                    message: err.to_string(),
                }];
            }
        };
        let chunks = match self.stream.push(&bytes) {
            Ok(chunks) => chunks,
            Err(err) => {
                return vec![ServerRealtimeEvent::Error {
                    message: err.to_string(),
                }];
            }
        };
        let mut events = Vec::new();
        for chunk in chunks.iter() {
            match self.pipeline.push_chunk(chunk) {
                Ok(annotations) => events.extend(events_from_annotations(annotations)),
                Err(err) => events.push(ServerRealtimeEvent::Error {
                    message: err.to_string(),
                }),
            }
        }
        events
    }

    pub fn finish(&mut self) -> Vec<ServerRealtimeEvent> {
        let mut events = Vec::new();
        if let Ok(chunks) = self.stream.flush() {
            for chunk in chunks.iter() {
                if let Ok(annotations) = self.pipeline.push_chunk(chunk) {
                    events.extend(events_from_annotations(annotations));
                }
            }
        }
        if let Ok(annotations) = self.pipeline.finish() {
            events.extend(events_from_annotations(annotations));
        }
        events.push(ServerRealtimeEvent::SessionFinished);
        events
    }
}

fn events_from_annotations(annotations: Vec<vasr_data::Annotation>) -> Vec<ServerRealtimeEvent> {
    annotations
        .into_iter()
        .filter_map(|annotation| match annotation.payload {
            AnnotationPayload::Speech if annotation.status == AnnotationStatus::Partial => {
                Some(ServerRealtimeEvent::SpeechStarted {
                    start_ms: annotation.range.start.0,
                })
            }
            AnnotationPayload::Speech if annotation.status == AnnotationStatus::Final => {
                Some(ServerRealtimeEvent::SpeechStopped {
                    start_ms: annotation.range.start.0,
                    end_ms: annotation.range.end.0,
                })
            }
            AnnotationPayload::Segment(segment)
                if annotation.status == AnnotationStatus::Partial =>
            {
                Some(ServerRealtimeEvent::TranscriptionText { text: segment.text })
            }
            AnnotationPayload::Segment(segment) if annotation.status == AnnotationStatus::Final => {
                Some(ServerRealtimeEvent::TranscriptionCompleted { text: segment.text })
            }
            _ => None,
        })
        .collect()
}
