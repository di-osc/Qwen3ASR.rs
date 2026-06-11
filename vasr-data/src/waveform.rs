use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum WaveformError {
    #[error("sample rate must be greater than zero")]
    InvalidSampleRate,
    #[error("audio byte input length must be divisible by two")]
    OddPcmByteLength,
    #[error("channel count must be greater than zero")]
    InvalidChannelCount,
    #[error("chunk size must be greater than zero")]
    InvalidChunkSize,
    #[error("channel index is out of range")]
    ChannelIndexOutOfRange,
    #[error("sample count {samples} is not divisible by channel count {channels}")]
    IncompleteFrame { samples: usize, channels: u16 },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Waveform {
    pub id: Option<String>,
    pub samples: Vec<f32>,
    pub encoded_data: Option<Vec<u8>>,
    pub encoded_format: Option<String>,
    pub sample_rate: u32,
    pub channels: u16,
    pub is_normalized: bool,
}

impl Waveform {
    pub fn new(samples: Vec<f32>, sample_rate: u32) -> Self {
        Self::new_with_channels(samples, sample_rate, 1)
    }

    pub fn new_with_channels(samples: Vec<f32>, sample_rate: u32, channels: u16) -> Self {
        debug_assert!(channels == 0 || samples.len().is_multiple_of(usize::from(channels)));
        Self {
            id: None,
            samples,
            encoded_data: None,
            encoded_format: None,
            sample_rate,
            channels,
            is_normalized: true,
        }
    }

    pub fn try_new_with_channels(
        samples: Vec<f32>,
        sample_rate: u32,
        channels: u16,
    ) -> Result<Self, WaveformError> {
        if channels == 0 {
            return Err(WaveformError::InvalidChannelCount);
        }
        if !samples.len().is_multiple_of(usize::from(channels)) {
            return Err(WaveformError::IncompleteFrame {
                samples: samples.len(),
                channels,
            });
        }
        Ok(Self::new_with_channels(samples, sample_rate, channels))
    }

    pub fn frame_count(&self) -> usize {
        let channels = usize::from(self.channels);
        if channels == 0 {
            return 0;
        }
        self.samples.len() / channels
    }

    pub fn from_i16_pcm(samples: &[i16], sample_rate: u32) -> Self {
        let samples = samples
            .iter()
            .map(|sample| f32::from(*sample) / 32768.0)
            .collect();
        Self::new(samples, sample_rate)
    }

    pub fn from_i16_pcm_with_channels(samples: &[i16], sample_rate: u32, channels: u16) -> Self {
        let samples = samples
            .iter()
            .map(|sample| f32::from(*sample) / 32768.0)
            .collect();
        Self::new_with_channels(samples, sample_rate, channels)
    }

    pub fn from_i16_pcm_bytes(bytes: &[u8], sample_rate: u32) -> Result<Self, WaveformError> {
        Self::from_i16_pcm_bytes_with_channels(bytes, sample_rate, 1)
    }

    pub fn from_i16_pcm_bytes_with_channels(
        bytes: &[u8],
        sample_rate: u32,
        channels: u16,
    ) -> Result<Self, WaveformError> {
        if bytes.len() % 2 != 0 {
            return Err(WaveformError::OddPcmByteLength);
        }
        if channels == 0 {
            return Err(WaveformError::InvalidChannelCount);
        }

        let samples = bytes
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
            .collect::<Vec<_>>();
        let samples = samples
            .iter()
            .map(|sample| f32::from(*sample) / 32768.0)
            .collect();
        Self::try_new_with_channels(samples, sample_rate, channels)
    }

    pub fn duration_ms(&self) -> f64 {
        if self.sample_rate == 0 || self.channels == 0 {
            return 0.0;
        }
        self.frame_count() as f64 * 1000.0 / f64::from(self.sample_rate)
    }

    pub fn duration_seconds(&self) -> f64 {
        self.duration_ms() / 1000.0
    }

    pub fn to_i16_pcm(&self) -> Vec<i16> {
        self.samples
            .iter()
            .map(|sample| {
                let scaled = sample.clamp(-1.0, 1.0) * 32768.0;
                scaled.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16
            })
            .collect()
    }

    pub fn append(&mut self, other: &Waveform) -> Result<(), WaveformError> {
        if self.sample_rate == 0 || other.sample_rate == 0 || self.sample_rate != other.sample_rate
        {
            return Err(WaveformError::InvalidSampleRate);
        }
        if self.channels == 0 || other.channels == 0 || self.channels != other.channels {
            return Err(WaveformError::InvalidChannelCount);
        }
        self.samples.extend_from_slice(&other.samples);
        Ok(())
    }

    pub fn slice_ms(&self, start_ms: u64, end_ms: u64) -> Self {
        if end_ms <= start_ms || self.sample_rate == 0 || self.channels == 0 {
            return Self::new_with_channels(Vec::new(), self.sample_rate, self.channels);
        }

        let channels = usize::from(self.channels);
        let start_frame = (start_ms as usize).saturating_mul(self.sample_rate as usize) / 1000;
        let end_frame = (end_ms as usize)
            .saturating_mul(self.sample_rate as usize)
            .div_ceil(1000)
            .min(self.frame_count());
        let start = start_frame.saturating_mul(channels).min(self.samples.len());
        let end = end_frame.saturating_mul(channels).min(self.samples.len());
        Self::new_with_channels(
            self.samples[start.min(self.samples.len())..end].to_vec(),
            self.sample_rate,
            self.channels,
        )
    }

    pub fn channel(&self, index: u16) -> Result<Self, WaveformError> {
        if self.channels == 0 {
            return Err(WaveformError::InvalidChannelCount);
        }
        if index >= self.channels {
            return Err(WaveformError::ChannelIndexOutOfRange);
        }
        if self.channels == 1 {
            return Ok(self.clone());
        }

        let channels = usize::from(self.channels);
        let index = usize::from(index);
        let samples = self
            .samples
            .chunks_exact(channels)
            .map(|frame| frame[index])
            .collect();
        Ok(Self::new_with_channels(samples, self.sample_rate, 1))
    }

    pub fn to_mono(&self) -> Result<Self, WaveformError> {
        if self.channels == 0 {
            return Err(WaveformError::InvalidChannelCount);
        }
        if self.channels == 1 {
            return Ok(self.clone());
        }

        let channels = usize::from(self.channels);
        let samples = self
            .samples
            .chunks_exact(channels)
            .map(|frame| frame.iter().copied().sum::<f32>() / channels as f32)
            .collect();
        Ok(Self::new_with_channels(samples, self.sample_rate, 1))
    }
}

impl Default for Waveform {
    fn default() -> Self {
        Self::new(Vec::new(), 16_000)
    }
}
