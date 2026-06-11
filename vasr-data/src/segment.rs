use serde::{Deserialize, Serialize};

use crate::Token;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextSpan {
    pub text: String,
    #[serde(default)]
    pub tokens: Vec<Token>,
    pub language: Option<String>,
}

impl TextSpan {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            tokens: Vec::new(),
            language: None,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Transcript {
    pub text: String,
    pub language: Option<String>,
    #[serde(default)]
    pub segments: Vec<TextSpan>,
}
