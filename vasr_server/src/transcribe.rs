use std::sync::Arc;
use std::time::Instant;

use axum::{Json, Router, routing::post};
use vasr_audio::{AudioLoadOptions, AudioLoader};
use vasr_data::AudioSource;
use vasr_protocol::{InferenceData, InferencePerformance, TranscribeRequest, TranscribeResponse};
use vasr_runtime::{AsrOptions, OfflinePipeline};

pub struct TranscribeService {
    pub pipeline: Arc<OfflinePipeline>,
    pub loader: AudioLoader,
    pub options: AsrOptions,
}

pub fn transcribe_router(service: Arc<TranscribeService>) -> Router {
    Router::new()
        .route("/transcribe", post(handle_transcribe))
        .route("/inference", post(handle_transcribe))
        .with_state(service)
}

async fn handle_transcribe(
    axum::extract::State(service): axum::extract::State<Arc<TranscribeService>>,
    Json(request): Json<TranscribeRequest>,
) -> Json<TranscribeResponse> {
    let start = Instant::now();
    let mut data = Vec::new();
    let mut total_audio_duration_seconds = 0.0;

    for input in request.inputs {
        let source = if let Some(url) = input.url {
            AudioSource::Url(url)
        } else if let Some(b64) = input.b64_str {
            AudioSource::Base64(b64)
        } else {
            data.push(InferenceData {
                service_id: String::new(),
                spent_seconds: 0.0,
                spent_details: Default::default(),
                text: Default::default(),
                sentences: Vec::new(),
                is_bad: true,
                bad_reason: Some("必须提供 url 或 b64_str 之一".to_string()),
                bad_component: Some("loader".to_string()),
            });
            continue;
        };

        match service.loader.load(&source, &AudioLoadOptions::default()) {
            Ok(waveform) => {
                total_audio_duration_seconds += waveform.duration_seconds();
                match service.pipeline.transcribe(&waveform, &service.options) {
                    Ok(timeline) => data.push(InferenceData::from_timeline("", &timeline)),
                    Err(err) => data.push(error_data("recognizer", err.to_string())),
                }
            }
            Err(err) => data.push(error_data("loader", err.to_string())),
        }
    }

    let wall = start.elapsed().as_secs_f64();
    let num_items = data.len();
    Json(TranscribeResponse {
        data,
        inference_performance: InferencePerformance {
            batch_wall_seconds: wall,
            num_items,
            throughput_items_per_second: if wall > 0.0 {
                num_items as f64 / wall
            } else {
                0.0
            },
            total_audio_duration_seconds,
            speedup: if wall > 0.0 {
                total_audio_duration_seconds / wall
            } else {
                0.0
            },
            rtf: if total_audio_duration_seconds > 0.0 {
                wall / total_audio_duration_seconds
            } else {
                0.0
            },
        },
    })
}

fn error_data(component: &str, reason: String) -> InferenceData {
    InferenceData {
        service_id: String::new(),
        spent_seconds: 0.0,
        spent_details: Default::default(),
        text: Default::default(),
        sentences: Vec::new(),
        is_bad: true,
        bad_reason: Some(reason),
        bad_component: Some(component.to_string()),
    }
}
