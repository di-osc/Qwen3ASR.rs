use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use thiserror::Error;

use crate::{DurationMs, Timeline, Waveform};

#[derive(Debug, Error)]
pub enum RecordError {
    #[error("failed to encode MessagePack record: {0}")]
    Encode(#[from] rmp_serde::encode::Error),
    #[error("failed to decode MessagePack record: {0}")]
    Decode(#[from] rmp_serde::decode::Error),
    #[error("record I/O failed: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioEncoding {
    Wav,
    Flac,
    Mp3,
    Ogg,
    PcmS16Le,
    Other(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PersistedAudioFormat {
    pub sample_rate: Option<u32>,
    pub channels: Option<u16>,
    pub encoding: AudioEncoding,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AudioAsset {
    Uri {
        uri: String,
        format: PersistedAudioFormat,
        duration: Option<DurationMs>,
        sha256: Option<String>,
    },
    Embedded {
        #[serde(with = "serde_bytes")]
        bytes: Vec<u8>,
        format: PersistedAudioFormat,
        duration: Option<DurationMs>,
        sha256: Option<String>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WaveformCache {
    pub waveform: Waveform,
    pub source: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VasrRecord {
    pub schema_version: String,
    pub media: AudioAsset,
    pub timeline: Timeline,
    #[serde(default)]
    pub metadata: BTreeMap<String, serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub waveform_cache: Option<WaveformCache>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VasrRecordList {
    pub schema_version: String,
    #[serde(default)]
    pub records: Vec<VasrRecord>,
    #[serde(default)]
    pub metadata: BTreeMap<String, serde_json::Value>,
}

impl VasrRecord {
    pub const CURRENT_SCHEMA_VERSION: &'static str = "vasr.record.v1";

    pub fn new(media: AudioAsset, timeline: Timeline) -> Self {
        Self {
            schema_version: Self::CURRENT_SCHEMA_VERSION.to_string(),
            media,
            timeline,
            metadata: BTreeMap::new(),
            waveform_cache: None,
        }
    }

    pub fn with_metadata_value(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    pub fn with_waveform_cache(mut self, waveform_cache: WaveformCache) -> Self {
        self.waveform_cache = Some(waveform_cache);
        self
    }

    /// Canonical vASR record id for a zero-based list index.
    pub fn id_for_index(index: usize) -> String {
        format!("record-{index}")
    }

    /// Record id used in benchmark reports and exported audio filenames.
    pub fn record_id(&self) -> String {
        let media_id = self.timeline.media_id.trim();
        if !media_id.is_empty() {
            return sanitize_record_id(media_id);
        }
        sanitize_record_id(&self.timeline.id)
    }

    pub fn to_msgpack(&self) -> Result<Vec<u8>, RecordError> {
        to_msgpack_bytes(self).map_err(RecordError::from)
    }

    pub fn from_msgpack(bytes: &[u8]) -> Result<Self, RecordError> {
        from_msgpack_bytes(bytes).map_err(RecordError::from)
    }

    pub fn write_msgpack(&self, path: impl AsRef<Path>) -> Result<(), RecordError> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        serialize_msgpack(&mut writer, self)?;
        Ok(())
    }

    pub fn read_msgpack(path: impl AsRef<Path>) -> Result<Self, RecordError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        from_msgpack_reader(reader).map_err(RecordError::from)
    }
}

impl VasrRecordList {
    pub const CURRENT_SCHEMA_VERSION: &'static str = "vasr.record_list.v1";

    pub fn new(records: Vec<VasrRecord>) -> Self {
        Self {
            schema_version: Self::CURRENT_SCHEMA_VERSION.to_string(),
            records,
            metadata: BTreeMap::new(),
        }
    }

    pub fn with_metadata_value(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn total_duration(&self) -> DurationMs {
        DurationMs(
            self.records
                .iter()
                .filter_map(|record| record.media.duration())
                .map(|duration| duration.0)
                .sum(),
        )
    }

    pub fn to_msgpack(&self) -> Result<Vec<u8>, RecordError> {
        to_msgpack_bytes(self).map_err(RecordError::from)
    }

    pub fn from_msgpack(bytes: &[u8]) -> Result<Self, RecordError> {
        from_msgpack_bytes(bytes).map_err(RecordError::from)
    }

    pub fn write_msgpack(&self, path: impl AsRef<Path>) -> Result<(), RecordError> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        serialize_msgpack(&mut writer, self)?;
        Ok(())
    }

    pub fn read_msgpack(path: impl AsRef<Path>) -> Result<Self, RecordError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        from_msgpack_reader(reader).map_err(RecordError::from)
    }
}

fn serialize_msgpack<W, T>(writer: W, value: &T) -> Result<(), rmp_serde::encode::Error>
where
    W: std::io::Write,
    T: Serialize + ?Sized,
{
    use rmp_serde::config::BytesMode;

    value.serialize(
        &mut rmp_serde::Serializer::new(writer)
            .with_struct_map()
            .with_binary()
            .with_bytes(BytesMode::ForceAll),
    )
}

fn to_msgpack_bytes<T>(value: &T) -> Result<Vec<u8>, rmp_serde::encode::Error>
where
    T: Serialize + ?Sized,
{
    let mut bytes = Vec::new();
    serialize_msgpack(&mut bytes, value)?;
    Ok(bytes)
}

fn from_msgpack_reader<R, T>(reader: R) -> Result<T, rmp_serde::decode::Error>
where
    R: std::io::Read,
    T: DeserializeOwned,
{
    T::deserialize(&mut rmp_serde::Deserializer::new(reader))
}

fn from_msgpack_bytes<T>(bytes: &[u8]) -> Result<T, rmp_serde::decode::Error>
where
    T: DeserializeOwned,
{
    from_msgpack_reader(bytes)
}

fn sanitize_record_id(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    out
}

impl AudioAsset {
    pub fn duration(&self) -> Option<DurationMs> {
        match self {
            Self::Uri { duration, .. } | Self::Embedded { duration, .. } => *duration,
        }
    }
}
