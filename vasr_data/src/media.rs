use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::Waveform;

pub type MediaId = String;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioChannel {
    Mono,
    Left,
    Right,
    Channel(u16),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioFormat {
    pub sample_rate: u32,
    pub channels: u16,
    pub sample_format: String,
}

impl AudioFormat {
    pub fn pcm16_mono(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            channels: 1,
            sample_format: "pcm16".to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AudioSource {
    Path(PathBuf),
    Url(String),
    Base64(String),
    Bytes(Vec<u8>),
    Waveform(Waveform),
}
