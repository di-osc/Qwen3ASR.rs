use std::sync::Arc;
use std::time::Instant;

use axum::{Json, Router, routing::post};
use vasr_data::AudioSource;
use vasr_runtime::{AsrOptions, VadOptions};

use crate::pipeline::{AsyncTranscribePipeline, TranscribeInput};
use crate::protocol::{InferenceData, InferencePerformance, TranscribeRequest, TranscribeResponse};

pub struct TranscribeService {
    pub pipeline: Arc<AsyncTranscribePipeline>,
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
    let mut inputs = Vec::new();
    let mut validation_errors = Vec::new();

    for (index, input) in request.inputs.into_iter().enumerate() {
        let source = if let Some(url) = input.url {
            AudioSource::Url(url)
        } else if let Some(b64) = input.b64_str {
            AudioSource::Base64(b64)
        } else {
            validation_errors.push(InferenceData {
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
        inputs.push(TranscribeInput { index, source });
    }

    let outcomes = service.pipeline.transcribe_many(inputs).await;
    let mut data = validation_errors;
    let mut total_audio_duration_seconds = 0.0;

    for outcome in outcomes {
        match outcome.result {
            Ok(timeline) => {
                total_audio_duration_seconds += outcome.audio_seconds;
                data.push(InferenceData::from_timeline("", &timeline));
            }
            Err(error) => data.push(error_data(
                outcome.bad_component.unwrap_or("recognizer"),
                error.to_string(),
            )),
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

pub fn build_transcribe_service(pipeline: Arc<AsyncTranscribePipeline>) -> Arc<TranscribeService> {
    Arc::new(TranscribeService { pipeline })
}

pub fn build_transcribe_service_from_parts(
    offline: Arc<vasr_runtime::OfflinePipeline>,
    asr_options: AsrOptions,
    vad_options: VadOptions,
    max_batch_size: usize,
) -> Arc<TranscribeService> {
    build_transcribe_service(Arc::new(
        AsyncTranscribePipeline::new(vasr_audio::AudioLoader, offline, asr_options)
            .with_vad_options(vad_options)
            .with_stage_buffer(max_batch_size),
    ))
}
