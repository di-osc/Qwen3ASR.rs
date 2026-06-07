use vasr_data::{
    Annotation, AnnotationPayload, AnnotationSource, AnnotationStatus, DurationMs, TextSegment,
    TimeRange, Timeline,
};
use vasr_protocol::InferenceData;

#[test]
fn inference_data_derives_fasr_text_and_sentences_from_timeline() {
    let mut timeline = Timeline::new("audio_1");
    timeline.push(Annotation::new(
        TimeRange::new(DurationMs(10), DurationMs(180)),
        AnnotationPayload::Segment(TextSegment::new("hello")),
        AnnotationSource::Model("asr".to_string()),
        AnnotationStatus::Final,
    ));
    timeline.push(Annotation::new(
        TimeRange::new(DurationMs(30), DurationMs(40)),
        AnnotationPayload::Segment(TextSegment::new("world")),
        AnnotationSource::Model("asr".to_string()),
        AnnotationStatus::Final,
    ));

    let data = InferenceData::from_timeline("svc", &timeline);

    assert_eq!(data.service_id, "svc");
    assert_eq!(data.text.mono, "hello world");
    assert_eq!(data.sentences.len(), 2);
    assert_eq!(data.sentences[1].sentence_index, 1);

    let json = serde_json::to_value(&data).expect("serialize inference data");
    assert_eq!(json["serviceId"], "svc");
    assert!(json.get("is_bad").is_some());
    assert!(json.get("bad_reason").is_some());
    assert!(json.get("bad_componet").is_some());
    assert!(json.get("isBad").is_none());
    assert!(json.get("badReason").is_none());
}
