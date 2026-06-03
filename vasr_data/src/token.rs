use serde::{Deserialize, Serialize};

use crate::TimeRange;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Token {
    pub text: String,
    pub range: Option<TimeRange>,
    pub confidence: Option<f32>,
}

impl Token {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            range: None,
            confidence: None,
        }
    }

    pub fn with_range(mut self, range: TimeRange) -> Self {
        self.range = Some(range);
        self
    }

    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = Some(confidence);
        self
    }
}
