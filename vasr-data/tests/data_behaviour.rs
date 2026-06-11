use vasr_data::{
    Annotation, AnnotationPayload, AnnotationSource, AnnotationStatus, AudioAsset,
    AudioBytesStream, AudioEncoding, DurationMs, PersistedAudioFormat, TextSpan, TimeRange,
    Timeline, Token, VasrRecord, VasrRecordList, Waveform, WaveformCache, WaveformError,
};

#[test]
fn waveform_round_trips_pcm16_samples() {
    let waveform = Waveform::from_i16_pcm(&[0, 16_384, -16_384, 32_767], 16_000);

    assert_eq!(waveform.sample_rate, 16_000);
    assert_eq!(waveform.channels, 1);
    assert_eq!(waveform.duration_ms(), 0.25);
    assert_eq!(waveform.to_i16_pcm(), vec![0, 16_384, -16_384, 32_767]);
}

#[test]
fn stereo_waveform_uses_interleaved_frames_for_duration_and_slicing() {
    let waveform = Waveform::new_with_channels(
        vec![
            1.0, 10.0, // frame 0
            2.0, 20.0, // frame 1
            3.0, 30.0, // frame 2
            4.0, 40.0, // frame 3
        ],
        1_000,
        2,
    );

    assert_eq!(waveform.frame_count(), 4);
    assert_eq!(waveform.duration_ms(), 4.0);

    let slice = waveform.slice_ms(1, 3);
    assert_eq!(slice.channels, 2);
    assert_eq!(slice.sample_rate, 1_000);
    assert_eq!(slice.samples, vec![2.0, 20.0, 3.0, 30.0]);
}

#[test]
fn stereo_waveform_rejects_incomplete_interleaved_frames() {
    assert_eq!(
        Waveform::try_new_with_channels(vec![1.0, 2.0, 3.0], 16_000, 2),
        Err(WaveformError::IncompleteFrame {
            samples: 3,
            channels: 2,
        })
    );
}

#[test]
fn stereo_pcm_bytes_reject_incomplete_interleaved_frames() {
    let mut bytes = Vec::new();
    for sample in [1_i16, 2_i16, 3_i16] {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }

    assert_eq!(
        Waveform::from_i16_pcm_bytes_with_channels(&bytes, 16_000, 2),
        Err(WaveformError::IncompleteFrame {
            samples: 3,
            channels: 2,
        })
    );
}

#[test]
fn stereo_waveform_can_extract_channels_and_downmix_to_mono() -> Result<(), WaveformError> {
    let waveform = Waveform::new_with_channels(
        vec![
            1.0, 3.0, // frame 0
            2.0, 4.0, // frame 1
        ],
        16_000,
        2,
    );

    let left = waveform.channel(0)?;
    assert_eq!(left.channels, 1);
    assert_eq!(left.samples, vec![1.0, 2.0]);

    let right = waveform.channel(1)?;
    assert_eq!(right.channels, 1);
    assert_eq!(right.samples, vec![3.0, 4.0]);

    let mono = waveform.to_mono()?;
    assert_eq!(mono.channels, 1);
    assert_eq!(mono.samples, vec![2.0, 3.0]);
    Ok(())
}

#[test]
fn append_rejects_waveforms_with_different_channel_counts() {
    let mut mono = Waveform::new(vec![0.0, 1.0], 16_000);
    let stereo = Waveform::new_with_channels(vec![0.0, 0.0, 1.0, 1.0], 16_000, 2);

    assert_eq!(
        mono.append(&stereo),
        Err(WaveformError::InvalidChannelCount)
    );
}

#[test]
fn time_range_reports_duration_and_overlap() {
    let range = TimeRange::new(DurationMs(100), DurationMs(240));

    assert_eq!(range.duration(), DurationMs(140));
    assert!(range.overlaps(&TimeRange::new(DurationMs(120), DurationMs(190))));
    assert!(!range.overlaps(&TimeRange::new(DurationMs(30), DurationMs(90))));
}

#[test]
fn timeline_derives_transcript_from_final_text_annotations_only() {
    let mut timeline = Timeline::new("audio_1");
    timeline.push(Annotation::new(
        TimeRange::new(DurationMs(0), DurationMs(100)),
        AnnotationPayload::Transcription(TextSpan {
            text: "partial".to_string(),
            tokens: vec![],
            language: None,
        }),
        AnnotationSource::Model("asr".to_string()),
        AnnotationStatus::Partial,
    ));
    timeline.push(Annotation::new(
        TimeRange::new(DurationMs(0), DurationMs(100)),
        AnnotationPayload::Transcription(TextSpan {
            text: "hello".to_string(),
            tokens: vec![
                Token::new("hello").with_range(TimeRange::new(DurationMs(0), DurationMs(40))),
            ],
            language: Some("English".to_string()),
        }),
        AnnotationSource::Model("asr".to_string()),
        AnnotationStatus::Final,
    ));
    timeline.push(Annotation::new(
        TimeRange::new(DurationMs(100), DurationMs(130)),
        AnnotationPayload::Sentence(TextSpan {
            text: "world".to_string(),
            tokens: vec![],
            language: None,
        }),
        AnnotationSource::Stage("sentencizer".to_string()),
        AnnotationStatus::Final,
    ));

    let transcript = timeline.transcript();

    assert_eq!(transcript.text, "hello world");
    assert_eq!(transcript.language.as_deref(), Some("English"));
    assert_eq!(transcript.segments.len(), 2);
    assert_eq!(timeline.by_status(AnnotationStatus::Final).len(), 2);
}

#[test]
fn audio_bytes_stream_emits_fixed_pcm_chunks_and_flushes_tail() -> Result<(), WaveformError> {
    let mut stream = AudioBytesStream::new(16_000, 1, 100);
    let mut frame = Vec::new();
    for _ in 0..1600 {
        frame.extend_from_slice(&0_i16.to_le_bytes());
    }
    frame.extend_from_slice(&1000_i16.to_le_bytes());
    frame.extend_from_slice(&(-1000_i16).to_le_bytes());

    let chunks = stream.push(&frame)?;
    assert_eq!(chunks.len(), 1);
    assert!(chunks[0].is_start);
    assert!(!chunks[0].is_last);
    assert_eq!(
        chunks[0].range,
        TimeRange::new(DurationMs(0), DurationMs(100))
    );
    assert_eq!(chunks[0].waveform.samples.len(), 1600);

    let tail = stream.flush()?;
    assert_eq!(tail.len(), 1);
    assert!(tail[0].is_last);
    assert_eq!(tail[0].range.start, DurationMs(100));
    assert_eq!(tail[0].waveform.samples.len(), 2);
    Ok(())
}

#[test]
fn audio_bytes_stream_preserves_stereo_channel_count() -> Result<(), WaveformError> {
    let mut stream = AudioBytesStream::new(1_000, 2, 2);
    let samples = [100_i16, 1000_i16, 200_i16, 2000_i16];
    let mut bytes = Vec::new();
    for sample in samples {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }

    let chunks = stream.push(&bytes)?;

    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].waveform.channels, 2);
    assert_eq!(chunks[0].waveform.samples.len(), 4);
    assert_eq!(
        chunks[0].range,
        TimeRange::new(DurationMs(0), DurationMs(2))
    );
    Ok(())
}

#[test]
fn vasr_record_round_trips_messagepack_with_embedded_audio_bytes() {
    let mut timeline = Timeline::new("audio_1");
    timeline.push(Annotation::new(
        TimeRange::new(DurationMs(0), DurationMs(100)),
        AnnotationPayload::Transcription(TextSpan::new("hello")),
        AnnotationSource::Model("asr".to_string()),
        AnnotationStatus::Final,
    ));
    let record = VasrRecord::new(
        AudioAsset::Embedded {
            bytes: vec![1, 2, 3, 4],
            format: PersistedAudioFormat {
                sample_rate: Some(16_000),
                channels: Some(1),
                encoding: AudioEncoding::Wav,
            },
            duration: Some(DurationMs(100)),
            sha256: Some("sha".to_string()),
        },
        timeline,
    )
    .with_metadata_value("model", serde_json::json!("qwen3-asr"));

    let bytes = record.to_msgpack().expect("serialize msgpack");
    let decoded = VasrRecord::from_msgpack(&bytes).expect("deserialize msgpack");

    assert_eq!(decoded.schema_version, VasrRecord::CURRENT_SCHEMA_VERSION);
    assert_eq!(decoded.timeline.transcript().text, "hello");
    assert_eq!(decoded.metadata["model"], serde_json::json!("qwen3-asr"));
    assert!(decoded.waveform_cache.is_none());
    assert!(matches!(
        decoded.media,
        AudioAsset::Embedded {
            bytes,
            format: PersistedAudioFormat {
                encoding: AudioEncoding::Wav,
                ..
            },
            ..
        } if bytes == vec![1, 2, 3, 4]
    ));
}

#[test]
fn vasr_record_waveform_cache_is_explicit_opt_in() {
    let timeline = Timeline::new("audio_1");
    let record = VasrRecord::new(
        AudioAsset::Uri {
            uri: "s3://bucket/audio.wav".to_string(),
            format: PersistedAudioFormat {
                sample_rate: Some(16_000),
                channels: Some(1),
                encoding: AudioEncoding::Wav,
            },
            duration: Some(DurationMs(250)),
            sha256: None,
        },
        timeline,
    );

    assert!(record.waveform_cache.is_none());

    let record = record.with_waveform_cache(WaveformCache {
        waveform: Waveform::from_i16_pcm(&[0, 1000, -1000], 16_000),
        source: "normalized_pcm_cache".to_string(),
    });
    let decoded =
        VasrRecord::from_msgpack(&record.to_msgpack().expect("serialize msgpack")).unwrap();

    assert_eq!(
        decoded
            .waveform_cache
            .expect("waveform cache")
            .waveform
            .to_i16_pcm(),
        vec![0, 1000, -1000]
    );
}

#[test]
fn vasr_record_reads_and_writes_messagepack_files() {
    let record = VasrRecord::new(
        AudioAsset::Uri {
            uri: "file:///tmp/audio.wav".to_string(),
            format: PersistedAudioFormat {
                sample_rate: Some(16_000),
                channels: Some(1),
                encoding: AudioEncoding::Wav,
            },
            duration: Some(DurationMs(10)),
            sha256: None,
        },
        Timeline::new("audio_1"),
    );
    let path = std::env::temp_dir().join(format!(
        "vasr-record-{}.msgpack",
        uuid::Uuid::new_v4().simple()
    ));

    record.write_msgpack(&path).expect("write msgpack");
    let decoded = VasrRecord::read_msgpack(&path).expect("read msgpack");
    std::fs::remove_file(&path).ok();

    assert_eq!(decoded.media, record.media);
    assert_eq!(decoded.schema_version, VasrRecord::CURRENT_SCHEMA_VERSION);
}

#[test]
fn vasr_record_list_round_trips_messagepack_files() {
    let first = VasrRecord::new(
        AudioAsset::Uri {
            uri: "file:///tmp/first.wav".to_string(),
            format: PersistedAudioFormat {
                sample_rate: Some(16_000),
                channels: Some(1),
                encoding: AudioEncoding::Wav,
            },
            duration: Some(DurationMs(100)),
            sha256: None,
        },
        Timeline::new("first"),
    );
    let second = VasrRecord::new(
        AudioAsset::Embedded {
            bytes: vec![5, 6, 7],
            format: PersistedAudioFormat {
                sample_rate: Some(16_000),
                channels: Some(1),
                encoding: AudioEncoding::Wav,
            },
            duration: Some(DurationMs(250)),
            sha256: None,
        },
        Timeline::new("second"),
    );
    let list = VasrRecordList::new(vec![first, second])
        .with_metadata_value("split", serde_json::json!("train"));
    let path = std::env::temp_dir().join(format!(
        "vasr-record-list-{}.msgpack",
        uuid::Uuid::new_v4().simple()
    ));

    list.write_msgpack(&path).expect("write msgpack list");
    let decoded = VasrRecordList::read_msgpack(&path).expect("read msgpack list");
    std::fs::remove_file(&path).ok();

    assert_eq!(
        decoded.schema_version,
        VasrRecordList::CURRENT_SCHEMA_VERSION
    );
    assert_eq!(decoded.len(), 2);
    assert!(!decoded.is_empty());
    assert_eq!(decoded.total_duration(), DurationMs(350));
    assert_eq!(decoded.metadata["split"], serde_json::json!("train"));
}
