use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{DurationMs, TimeRange, Waveform, WaveformError};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioChunk {
    pub stream_id: String,
    pub waveform: Waveform,
    pub is_start: bool,
    pub is_last: bool,
    pub range: TimeRange,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AudioChunkList(pub Vec<AudioChunk>);

impl AudioChunkList {
    pub fn new(chunks: Vec<AudioChunk>) -> Self {
        Self(chunks)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl std::ops::Deref for AudioChunkList {
    type Target = [AudioChunk];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl IntoIterator for AudioChunkList {
    type IntoIter = std::vec::IntoIter<AudioChunk>;
    type Item = AudioChunk;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

#[derive(Debug, Clone)]
pub struct AudioBytesStream {
    id: String,
    sample_rate: u32,
    num_channels: usize,
    chunk_size_ms: u64,
    buffer: Vec<u8>,
    emitted_samples: u64,
}

impl AudioBytesStream {
    pub fn new(sample_rate: u32, num_channels: usize, chunk_size_ms: u64) -> Self {
        Self {
            id: format!("stream_{}", Uuid::new_v4().simple()),
            sample_rate,
            num_channels,
            chunk_size_ms,
            buffer: Vec::new(),
            emitted_samples: 0,
        }
    }

    pub fn push(&mut self, bytes: &[u8]) -> Result<AudioChunkList, WaveformError> {
        self.validate()?;
        self.buffer.extend_from_slice(bytes);

        let frame_bytes = self.frame_bytes();
        let mut chunks = Vec::new();
        while self.buffer.len() >= frame_bytes {
            let chunk_bytes = self.buffer.drain(..frame_bytes).collect::<Vec<_>>();
            chunks.push(self.chunk_from_bytes(&chunk_bytes, false)?);
        }
        Ok(AudioChunkList::new(chunks))
    }

    pub fn flush(&mut self) -> Result<AudioChunkList, WaveformError> {
        self.validate()?;
        if self.buffer.is_empty() {
            return Ok(AudioChunkList::default());
        }
        let complete_len = self.buffer.len() - (self.buffer.len() % 2);
        let chunk_bytes = self.buffer.drain(..complete_len).collect::<Vec<_>>();
        self.buffer.clear();
        if chunk_bytes.is_empty() {
            return Ok(AudioChunkList::default());
        }
        Ok(AudioChunkList::new(vec![
            self.chunk_from_bytes(&chunk_bytes, true)?,
        ]))
    }

    pub fn pending_bytes(&self) -> usize {
        self.buffer.len()
    }

    fn validate(&self) -> Result<(), WaveformError> {
        if self.sample_rate == 0 {
            return Err(WaveformError::InvalidSampleRate);
        }
        if self.num_channels == 0 {
            return Err(WaveformError::InvalidChannelCount);
        }
        if self.chunk_size_ms == 0 {
            return Err(WaveformError::InvalidChunkSize);
        }
        Ok(())
    }

    fn frame_bytes(&self) -> usize {
        let samples = (u64::from(self.sample_rate) * self.chunk_size_ms / 1000) as usize;
        samples * self.num_channels * 2
    }

    fn chunk_from_bytes(
        &mut self,
        bytes: &[u8],
        is_last: bool,
    ) -> Result<AudioChunk, WaveformError> {
        let waveform = Waveform::from_i16_pcm_bytes_with_channels(
            bytes,
            self.sample_rate,
            self.num_channels as u16,
        )?;
        let samples = waveform.samples.len() as u64 / self.num_channels as u64;
        let start_sample = self.emitted_samples;
        self.emitted_samples += samples;

        Ok(AudioChunk {
            stream_id: self.id.clone(),
            waveform,
            is_start: start_sample == 0,
            is_last,
            range: TimeRange::new(
                DurationMs(sample_to_ms(start_sample, self.sample_rate)),
                DurationMs(sample_to_ms(self.emitted_samples, self.sample_rate)),
            ),
        })
    }
}

fn sample_to_ms(sample: u64, sample_rate: u32) -> u64 {
    sample.saturating_mul(1000) / u64::from(sample_rate)
}
