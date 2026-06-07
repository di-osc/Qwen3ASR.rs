use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use serde::Deserialize;
use thiserror::Error;

use crate::{
    Annotation, AnnotationPayload, AnnotationSource, AnnotationStatus, AudioAsset, AudioEncoding,
    DurationMs, PersistedAudioFormat, TextSegment, TimeRange, Timeline, VasrRecord, VasrRecordList,
};

const MAGIC: &[u8; 8] = b"FASRAL01";
const VERSION: u64 = 1;
const METADATA_LENGTH_BYTES: usize = 8;
const RECORD_LENGTH_BYTES: usize = 4;

#[derive(Debug, Error)]
pub enum FasrAudioListError {
    #[error("failed to read FASR AudioList: {0}")]
    Io(#[from] std::io::Error),
    #[error("unsupported FASR AudioList format: {0}")]
    Format(String),
    #[error("failed to decode FASR metadata JSON: {0}")]
    MetadataJson(#[from] serde_json::Error),
    #[error("failed to decode FASR audio pickle: {0}")]
    Pickle(String),
}

#[derive(Debug, Clone, Deserialize)]
struct FasrAudioListIndex {
    version: u64,
    audios: Vec<FasrAudioMetadata>,
}

#[derive(Debug, Clone, Deserialize)]
struct FasrAudioMetadata {
    id: String,
    url: String,
    text: String,
    sample_rate: u32,
    duration: f64,
    mono: bool,
    channel_count: u16,
}

#[derive(Debug, Clone)]
pub struct FasrAudioListSummary {
    pub sample_count: usize,
    pub has_reference_text: bool,
}

pub fn inspect_fasr_audio_list(
    path: impl AsRef<Path>,
) -> Result<FasrAudioListSummary, FasrAudioListError> {
    let index = read_index(path.as_ref())?;
    Ok(FasrAudioListSummary {
        sample_count: index.audios.len(),
        has_reference_text: index
            .audios
            .iter()
            .any(|audio| !audio.text.trim().is_empty()),
    })
}

pub fn convert_fasr_audio_list_to_vasr_records(
    path: impl AsRef<Path>,
    limit: Option<usize>,
) -> Result<Vec<VasrRecord>, FasrAudioListError> {
    let path = path.as_ref();
    let index = read_index(path)?;
    let offsets = read_record_offsets(path, index.audios.len())?;

    let count = limit.unwrap_or(index.audios.len()).min(index.audios.len());
    let mut records = Vec::with_capacity(count);
    let mut file = BufReader::new(File::open(path)?);

    for (index, (metadata, (offset, length))) in index
        .audios
        .into_iter()
        .zip(offsets)
        .enumerate()
        .take(count)
    {
        file.seek(SeekFrom::Start(offset))?;
        let mut pickle = vec![0u8; length];
        file.read_exact(&mut pickle)?;

        let encoded = extract_encoded_audio_from_pickle(&pickle)?;
        let duration_ms = DurationMs((metadata.duration * 1000.0).round() as u64);
        let record_id = VasrRecord::id_for_index(index);
        let mut timeline = Timeline::new(&record_id);
        timeline.duration = Some(duration_ms);
        if !metadata.text.trim().is_empty() {
            timeline.push(Annotation::new(
                TimeRange::new(DurationMs(0), duration_ms),
                AnnotationPayload::Segment(TextSegment::new(metadata.text.clone())),
                AnnotationSource::User,
                AnnotationStatus::Final,
            ));
        }

        let encoding = match encoded.format.as_str() {
            "mp3" => AudioEncoding::Mp3,
            "wav" => AudioEncoding::Wav,
            "flac" => AudioEncoding::Flac,
            "ogg" => AudioEncoding::Ogg,
            other => AudioEncoding::Other(other.to_string()),
        };

        let record = VasrRecord::new(
            AudioAsset::Embedded {
                bytes: encoded.bytes,
                format: PersistedAudioFormat {
                    sample_rate: Some(metadata.sample_rate),
                    channels: Some(metadata.channel_count),
                    encoding,
                },
                duration: Some(duration_ms),
                sha256: None,
            },
            timeline,
        );

        records.push(record);
    }

    Ok(records)
}

pub fn convert_fasr_audio_list_file(
    input: impl AsRef<Path>,
    output: impl AsRef<Path>,
    limit: Option<usize>,
) -> Result<VasrRecordList, FasrAudioListError> {
    let records = convert_fasr_audio_list_to_vasr_records(input, limit)?;
    let list = VasrRecordList::new(records);
    list.write_msgpack(output.as_ref())
        .map_err(|err| match err {
            crate::RecordError::Io(error) => FasrAudioListError::Io(error),
            other => FasrAudioListError::Format(other.to_string()),
        })?;
    Ok(list)
}

fn read_index(path: &Path) -> Result<FasrAudioListIndex, FasrAudioListError> {
    let mut file = BufReader::new(File::open(path)?);
    read_magic(&mut file)?;

    let metadata_length = read_u64_be(&mut file)? as usize;
    let mut metadata_bytes = vec![0u8; metadata_length];
    file.read_exact(&mut metadata_bytes)?;

    let index: FasrAudioListIndex = serde_json::from_slice(&metadata_bytes)?;
    if index.version != VERSION {
        return Err(FasrAudioListError::Format(format!(
            "unsupported FASR AudioList version {}; expected {VERSION}",
            index.version
        )));
    }
    Ok(index)
}

fn read_record_offsets(path: &Path, count: usize) -> Result<Vec<(u64, usize)>, FasrAudioListError> {
    let mut file = BufReader::new(File::open(path)?);
    read_magic(&mut file)?;
    let metadata_length = read_u64_be(&mut file)? as usize;
    file.seek(SeekFrom::Current(metadata_length as i64))?;

    let mut offsets = Vec::with_capacity(count);
    for _ in 0..count {
        let record_length = read_u32_be(&mut file)? as usize;
        let offset = file.stream_position()?;
        offsets.push((offset, record_length));
        file.seek(SeekFrom::Current(record_length as i64))?;
    }
    Ok(offsets)
}

fn read_magic(file: &mut impl Read) -> Result<(), FasrAudioListError> {
    let mut magic = [0u8; 8];
    file.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(FasrAudioListError::Format(
            "missing FASRAL01 magic header".to_string(),
        ));
    }
    Ok(())
}

fn read_u64_be(file: &mut impl Read) -> Result<u64, FasrAudioListError> {
    let mut buf = [0u8; METADATA_LENGTH_BYTES];
    file.read_exact(&mut buf)?;
    Ok(u64::from_be_bytes(buf))
}

fn read_u32_be(file: &mut impl Read) -> Result<u32, FasrAudioListError> {
    let mut buf = [0u8; RECORD_LENGTH_BYTES];
    file.read_exact(&mut buf)?;
    Ok(u32::from_be_bytes(buf))
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct EncodedAudio {
    bytes: Vec<u8>,
    format: String,
}

fn extract_encoded_audio_from_pickle(blob: &[u8]) -> Result<EncodedAudio, FasrAudioListError> {
    let bytes = find_binbytes_after_short_key(blob, "encoded_data")?.ok_or_else(|| {
        FasrAudioListError::Pickle("missing encoded_data in FASR audio pickle".to_string())
    })?;
    let format = find_short_binunicode_after_short_key(blob, "encoded_format")?
        .unwrap_or_else(|| "mp3".to_string());
    Ok(EncodedAudio { bytes, format })
}

fn find_short_binunicode_after_short_key(
    blob: &[u8],
    key: &str,
) -> Result<Option<String>, FasrAudioListError> {
    let Some(mut cursor) = find_short_binunicode_key(blob, key) else {
        return Ok(None);
    };
    skip_pickle_memoize(blob, &mut cursor)?;
    if cursor >= blob.len() || blob[cursor] != 0x8c {
        return Ok(None);
    }
    cursor += 1;
    Ok(Some(read_short_binunicode(blob, &mut cursor)?))
}

fn find_binbytes_after_short_key(
    blob: &[u8],
    key: &str,
) -> Result<Option<Vec<u8>>, FasrAudioListError> {
    let Some(mut cursor) = find_short_binunicode_key(blob, key) else {
        return Ok(None);
    };
    skip_pickle_memoize(blob, &mut cursor)?;
    if cursor >= blob.len() || blob[cursor] != 0x42 {
        return Err(FasrAudioListError::Pickle(format!(
            "expected BINBYTES after {key} key, got 0x{:02x}",
            blob[cursor]
        )));
    }
    cursor += 1;
    Ok(Some(read_binbytes(blob, &mut cursor)?))
}

fn find_short_binunicode_key(blob: &[u8], key: &str) -> Option<usize> {
    if key.len() > u8::MAX as usize {
        return None;
    }
    let mut pattern = Vec::with_capacity(2 + key.len());
    pattern.push(0x8c);
    pattern.push(key.len() as u8);
    pattern.extend_from_slice(key.as_bytes());
    blob.windows(pattern.len())
        .position(|window| window == pattern)
        .map(|start| start + pattern.len())
}

fn read_binbytes(blob: &[u8], cursor: &mut usize) -> Result<Vec<u8>, FasrAudioListError> {
    let length = read_u32_le(blob, cursor)? as usize;
    read_bytes(blob, cursor, length)
}

fn read_short_binunicode(blob: &[u8], cursor: &mut usize) -> Result<String, FasrAudioListError> {
    let length = read_u8(blob, cursor)? as usize;
    read_utf8(blob, cursor, length)
}

fn skip_pickle_memoize(blob: &[u8], cursor: &mut usize) -> Result<(), FasrAudioListError> {
    if *cursor < blob.len() && blob[*cursor] == 0x94 {
        *cursor += 1;
    }
    Ok(())
}

fn read_u8(blob: &[u8], cursor: &mut usize) -> Result<u8, FasrAudioListError> {
    if *cursor >= blob.len() {
        return Err(FasrAudioListError::Pickle("unexpected EOF".to_string()));
    }
    let value = blob[*cursor];
    *cursor += 1;
    Ok(value)
}

fn read_bytes(
    blob: &[u8],
    cursor: &mut usize,
    length: usize,
) -> Result<Vec<u8>, FasrAudioListError> {
    if *cursor + length > blob.len() {
        return Err(FasrAudioListError::Pickle("unexpected EOF".to_string()));
    }
    let value = blob[*cursor..*cursor + length].to_vec();
    *cursor += length;
    Ok(value)
}

fn read_utf8(blob: &[u8], cursor: &mut usize, length: usize) -> Result<String, FasrAudioListError> {
    let bytes = read_bytes(blob, cursor, length)?;
    String::from_utf8(bytes).map_err(|err| FasrAudioListError::Pickle(err.to_string()))
}

fn read_u32_le(blob: &[u8], cursor: &mut usize) -> Result<u32, FasrAudioListError> {
    let bytes = read_bytes(blob, cursor, 4)?;
    Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
}

#[cfg(test)]
mod tests {
    use super::{convert_fasr_audio_list_to_vasr_records, inspect_fasr_audio_list};
    use std::path::PathBuf;

    fn fixture_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("lbg_400call-200.bin")
    }

    #[test]
    fn inspect_fasr_audio_list_reads_metadata_only() {
        let path = fixture_path();
        if !path.exists() {
            return;
        }
        let summary = inspect_fasr_audio_list(&path).expect("inspect fasr list");
        assert_eq!(summary.sample_count, 202);
        assert!(summary.has_reference_text);
    }

    #[test]
    fn convert_fasr_audio_list_extracts_reference_text_and_embedded_audio() {
        let path = fixture_path();
        if !path.exists() {
            return;
        }
        let records = convert_fasr_audio_list_to_vasr_records(&path, Some(2)).expect("convert");
        assert_eq!(records.len(), 2);
        assert!(records[0].timeline.transcript().text.contains("投诉"));
        match &records[0].media {
            crate::AudioAsset::Embedded { bytes, format, .. } => {
                assert_eq!(format.encoding, crate::AudioEncoding::Mp3);
                assert!(bytes.len() > 1000);
            }
            other => panic!("expected embedded media, got {other:?}"),
        }
    }
}
