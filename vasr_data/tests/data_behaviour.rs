use vasr_data::{
    Annotation, AnnotationPayload, AnnotationSource, AnnotationStatus, AudioBytesStream,
    DurationMs, TextSegment, TimeRange, Timeline, Token, Waveform, WaveformError,
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
        AnnotationPayload::Segment(TextSegment {
            text: "partial".to_string(),
            tokens: vec![],
            language: None,
        }),
        AnnotationSource::Model("asr".to_string()),
        AnnotationStatus::Partial,
    ));
    timeline.push(Annotation::new(
        TimeRange::new(DurationMs(0), DurationMs(100)),
        AnnotationPayload::Segment(TextSegment {
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
        AnnotationPayload::Sentence(TextSegment {
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
