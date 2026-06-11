use serde::{Deserialize, Serialize};
use vasr_data::{AnnotationPayload, AnnotationStatus, Timeline};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioInputDto {
    pub url: Option<String>,
    pub b64_str: Option<String>,
    pub mono: Option<bool>,
    pub hotword: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscribeRequest {
    pub inputs: Vec<AudioInputDto>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TextResult {
    pub mono: String,
    pub left: String,
    pub right: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SentenceResult {
    pub text: String,
    pub start_time: u64,
    pub end_time: u64,
    pub duration: u64,
    pub channel: String,
    pub sentence_index: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpentDetails {
    pub loader: f64,
    pub detector: f64,
    pub recognizer: f64,
    pub sentencizer: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceData {
    #[serde(rename = "serviceId")]
    pub service_id: String,
    #[serde(rename = "spentSeconds")]
    pub spent_seconds: f64,
    #[serde(rename = "spentDetails")]
    pub spent_details: SpentDetails,
    pub text: TextResult,
    pub sentences: Vec<SentenceResult>,
    pub is_bad: bool,
    pub bad_reason: Option<String>,
    pub bad_component: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferencePerformance {
    pub batch_wall_seconds: f64,
    pub num_items: usize,
    pub throughput_items_per_second: f64,
    pub total_audio_duration_seconds: f64,
    pub speedup: f64,
    pub rtf: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscribeResponse {
    pub data: Vec<InferenceData>,
    pub inference_performance: InferencePerformance,
}

impl InferenceData {
    pub fn from_timeline(service_id: impl Into<String>, timeline: &Timeline) -> Self {
        let transcript = timeline.transcript();
        let mut sentences = Vec::new();
        for annotation in timeline
            .annotations
            .iter()
            .filter(|annotation| annotation.status == AnnotationStatus::Final)
        {
            match &annotation.payload {
                AnnotationPayload::Transcription(segment)
                | AnnotationPayload::Sentence(segment) => {
                    let duration = annotation.range.duration().0;
                    sentences.push(SentenceResult {
                        text: segment.text.clone(),
                        start_time: annotation.range.start.0,
                        end_time: annotation.range.end.0,
                        duration,
                        channel: "mono".to_string(),
                        sentence_index: sentences.len(),
                    });
                }
                _ => {}
            }
        }
        Self {
            service_id: service_id.into(),
            spent_seconds: 0.0,
            spent_details: SpentDetails::default(),
            text: TextResult {
                mono: transcript.text,
                left: String::new(),
                right: String::new(),
            },
            sentences,
            is_bad: false,
            bad_reason: None,
            bad_component: None,
        }
    }
}
