use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{MediaId, TextSegment, TimeRange, Token, Transcript};

pub type TimelineId = String;
pub type AnnotationId = String;
pub type SpeakerId = String;
pub type LanguageTag = String;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnnotationStatus {
    Partial,
    Final,
    Revised,
    Deleted,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnnotationSource {
    User,
    Model(String),
    Stage(String),
    System,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HotwordMatch {
    pub text: String,
    pub normalized: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AcousticEvent {
    pub label: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Diagnostic {
    pub code: String,
    pub message: String,
    pub component: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AnnotationPayload {
    Speech,
    Silence,
    Token(Token),
    Segment(TextSegment),
    Sentence(TextSegment),
    Speaker(SpeakerId),
    Language(LanguageTag),
    Hotword(HotwordMatch),
    AcousticEvent(AcousticEvent),
    Diagnostic(Diagnostic),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Annotation {
    pub id: AnnotationId,
    pub range: TimeRange,
    pub status: AnnotationStatus,
    pub source: AnnotationSource,
    pub confidence: Option<f32>,
    pub payload: AnnotationPayload,
}

impl Annotation {
    pub fn new(
        range: TimeRange,
        payload: AnnotationPayload,
        source: AnnotationSource,
        status: AnnotationStatus,
    ) -> Self {
        Self {
            id: format!("ann_{}", Uuid::new_v4().simple()),
            range,
            status,
            source,
            confidence: None,
            payload,
        }
    }

    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = Some(confidence);
        self
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Timeline {
    pub id: TimelineId,
    pub media_id: MediaId,
    pub duration: Option<crate::DurationMs>,
    #[serde(default)]
    pub annotations: Vec<Annotation>,
}

impl Timeline {
    pub fn new(media_id: impl Into<MediaId>) -> Self {
        Self {
            id: format!("tl_{}", Uuid::new_v4().simple()),
            media_id: media_id.into(),
            duration: None,
            annotations: Vec::new(),
        }
    }

    pub fn push(&mut self, annotation: Annotation) {
        self.annotations.push(annotation);
    }

    pub fn by_status(&self, status: AnnotationStatus) -> Vec<&Annotation> {
        self.annotations
            .iter()
            .filter(|annotation| annotation.status == status)
            .collect()
    }

    pub fn transcript(&self) -> Transcript {
        let mut segments = self
            .annotations
            .iter()
            .filter(|annotation| annotation.status == AnnotationStatus::Final)
            .filter_map(|annotation| match &annotation.payload {
                AnnotationPayload::Segment(segment) | AnnotationPayload::Sentence(segment) => {
                    Some((annotation.range.start, segment.clone()))
                }
                _ => None,
            })
            .collect::<Vec<_>>();

        segments.sort_by_key(|(start, _)| *start);
        let segments = segments
            .into_iter()
            .map(|(_, segment)| segment)
            .collect::<Vec<_>>();
        let text = segments
            .iter()
            .map(|segment| segment.text.trim())
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        let language = segments.iter().find_map(|segment| segment.language.clone());

        Transcript {
            text,
            language,
            segments,
        }
    }
}
