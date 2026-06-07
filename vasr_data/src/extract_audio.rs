use std::fs;
use std::path::Path;

use crate::{AudioAsset, AudioEncoding, VasrRecord, VasrRecordList};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExtractAudioSummary {
    pub extracted: usize,
    pub skipped: usize,
}

pub fn extract_embedded_audio(
    list: &VasrRecordList,
    dir: impl AsRef<Path>,
) -> Result<ExtractAudioSummary, std::io::Error> {
    let dir = dir.as_ref();
    fs::create_dir_all(dir)?;

    let mut extracted = 0usize;
    let mut skipped = 0usize;
    let mut used_names = std::collections::HashSet::new();

    for record in &list.records {
        let Some((bytes, extension)) = embedded_audio_bytes(record) else {
            skipped += 1;
            continue;
        };

        let stem = record.record_id();
        let mut filename = format!("{stem}.{extension}");
        let mut suffix = 1usize;
        while used_names.contains(&filename) {
            filename = format!("{stem}_{suffix}.{extension}");
            suffix += 1;
        }
        used_names.insert(filename.clone());

        let path = dir.join(filename);
        fs::write(path, bytes)?;
        extracted += 1;
    }

    Ok(ExtractAudioSummary { extracted, skipped })
}

fn embedded_audio_bytes(record: &VasrRecord) -> Option<(&[u8], &'static str)> {
    let AudioAsset::Embedded { bytes, format, .. } = &record.media else {
        return None;
    };
    Some((bytes.as_slice(), encoding_extension(&format.encoding)))
}

fn encoding_extension(encoding: &AudioEncoding) -> &'static str {
    match encoding {
        AudioEncoding::Mp3 => "mp3",
        AudioEncoding::Wav => "wav",
        AudioEncoding::Flac => "flac",
        AudioEncoding::Ogg => "ogg",
        AudioEncoding::PcmS16Le => "pcm",
        AudioEncoding::Other(_) => "audio",
    }
}

pub fn extract_embedded_audio_from_msgpack(
    input: impl AsRef<Path>,
    dir: impl AsRef<Path>,
) -> Result<ExtractAudioSummary, crate::RecordError> {
    let list = VasrRecordList::read_msgpack(input)?;
    extract_embedded_audio(&list, dir).map_err(crate::RecordError::Io)
}

#[cfg(test)]
mod tests {
    use crate::{AudioAsset, AudioEncoding, PersistedAudioFormat, Timeline, VasrRecord};

    #[test]
    fn exported_filename_uses_record_id() {
        let record = VasrRecord::new(
            AudioAsset::Embedded {
                bytes: vec![1, 2, 3],
                format: PersistedAudioFormat {
                    sample_rate: Some(16_000),
                    channels: Some(1),
                    encoding: AudioEncoding::Mp3,
                },
                duration: None,
                sha256: None,
            },
            Timeline::new("record-12"),
        );
        assert_eq!(record.record_id(), "record-12");
    }
}
