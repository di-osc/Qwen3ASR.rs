use anyhow::Result;
use std::sync::{Arc, Mutex};
use vasr_data::{
    Annotation, AnnotationPayload, AnnotationSource, AnnotationStatus, AudioChunk, DurationMs,
    TextSegment, TimeRange, Timeline, Waveform,
};
use vasr_runtime::{
    AsrModel, AsrOptions, OfflinePipeline, StreamingAsrModel, StreamingVadModel, VadModel,
    VadOptions, VadSegment,
};

#[derive(Default)]
struct FakeAsr {
    seen_durations_ms: Mutex<Vec<u64>>,
    seen_batch_durations_ms: Mutex<Vec<Vec<u64>>>,
}

impl AsrModel for FakeAsr {
    fn transcribe(&self, waveform: &Waveform, _options: &AsrOptions) -> Result<Timeline> {
        self.seen_durations_ms
            .lock()
            .expect("seen durations poisoned")
            .push(waveform.duration_ms().round() as u64);
        let mut timeline = Timeline::new("fake_audio");
        timeline.push(Annotation::new(
            TimeRange::new(
                DurationMs(0),
                DurationMs(waveform.duration_ms().round() as u64),
            ),
            AnnotationPayload::Segment(TextSegment::new("hello world")),
            AnnotationSource::Model("fake_asr".to_string()),
            AnnotationStatus::Final,
        ));
        Ok(timeline)
    }

    fn transcribe_batch(
        &self,
        waveforms: &[Waveform],
        _options: &AsrOptions,
    ) -> Result<Vec<Timeline>> {
        self.seen_batch_durations_ms
            .lock()
            .expect("seen batch durations poisoned")
            .push(
                waveforms
                    .iter()
                    .map(|waveform| waveform.duration_ms().round() as u64)
                    .collect(),
            );
        Ok(waveforms
            .iter()
            .map(|waveform| {
                let mut timeline = Timeline::new("fake_audio_batch");
                timeline.push(Annotation::new(
                    TimeRange::new(
                        DurationMs(0),
                        DurationMs(waveform.duration_ms().round() as u64),
                    ),
                    AnnotationPayload::Segment(TextSegment::new("hello batch")),
                    AnnotationSource::Model("fake_asr".to_string()),
                    AnnotationStatus::Final,
                ));
                timeline
            })
            .collect())
    }

    fn start_stream(&self, _options: &AsrOptions) -> Result<Box<dyn StreamingAsrModel>> {
        Ok(Box::new(FakeStream))
    }
}

struct FakeStream;

impl StreamingAsrModel for FakeStream {
    fn push_chunk(&mut self, chunk: &AudioChunk) -> Result<Vec<Annotation>> {
        Ok(vec![Annotation::new(
            chunk.range,
            AnnotationPayload::Segment(TextSegment::new("partial")),
            AnnotationSource::Model("fake_asr".to_string()),
            AnnotationStatus::Partial,
        )])
    }

    fn finish(&mut self) -> Result<Vec<Annotation>> {
        Ok(vec![Annotation::new(
            TimeRange::default(),
            AnnotationPayload::Segment(TextSegment::new("final")),
            AnnotationSource::Model("fake_asr".to_string()),
            AnnotationStatus::Final,
        )])
    }
}

struct FakeVad {
    segments: Vec<VadSegment>,
}

impl VadModel for FakeVad {
    fn detect(&self, _waveform: &Waveform, _options: &VadOptions) -> Result<Vec<VadSegment>> {
        Ok(self.segments.clone())
    }

    fn start_stream(&self, _options: &VadOptions) -> Result<Box<dyn StreamingVadModel>> {
        Ok(Box::new(FakeVadStream))
    }
}

struct FakeVadStream;

impl StreamingVadModel for FakeVadStream {
    fn push_chunk(&mut self, _chunk: &AudioChunk) -> Result<Vec<Annotation>> {
        Ok(Vec::new())
    }

    fn finish(&mut self) -> Result<Vec<Annotation>> {
        Ok(Vec::new())
    }
}

#[test]
fn offline_pipeline_merges_vad_and_asr_annotations() -> Result<()> {
    let waveform = Waveform::new(vec![0.05; 160_000], 16_000);
    let asr = Arc::new(FakeAsr::default());
    let pipeline = OfflinePipeline {
        vad: Some(Arc::new(FakeVad {
            segments: vec![VadSegment {
                range: TimeRange::new(
                    DurationMs(0),
                    DurationMs(waveform.duration_ms().round() as u64),
                ),
                probability: 0.9,
            }],
        })),
        asr: asr.clone(),
    };

    let timeline = pipeline.transcribe(&waveform, &AsrOptions::default())?;

    assert!(timeline.annotations.len() >= 2);
    assert_eq!(timeline.transcript().text, "hello batch");
    Ok(())
}

#[test]
fn offline_pipeline_feeds_vad_segments_to_asr_and_offsets_annotations() -> Result<()> {
    let waveform = Waveform::new(vec![0.05; 160_000], 16_000);
    let asr = Arc::new(FakeAsr::default());
    let pipeline = OfflinePipeline {
        vad: Some(Arc::new(FakeVad {
            segments: vec![
                VadSegment {
                    range: TimeRange::new(DurationMs(1_000), DurationMs(2_500)),
                    probability: 0.9,
                },
                VadSegment {
                    range: TimeRange::new(DurationMs(4_000), DurationMs(5_000)),
                    probability: 0.8,
                },
            ],
        })),
        asr: asr.clone(),
    };

    let timeline = pipeline.transcribe(&waveform, &AsrOptions::default())?;

    assert_eq!(
        asr.seen_batch_durations_ms
            .lock()
            .expect("seen batch durations poisoned")
            .as_slice(),
        &[vec![1_500, 1_000]]
    );
    assert!(
        asr.seen_durations_ms
            .lock()
            .expect("seen durations poisoned")
            .is_empty()
    );
    let final_segments = timeline
        .annotations
        .iter()
        .filter(|annotation| matches!(annotation.payload, AnnotationPayload::Segment(_)))
        .collect::<Vec<_>>();
    assert_eq!(final_segments.len(), 2);
    assert_eq!(final_segments[0].range.start, DurationMs(1_000));
    assert_eq!(final_segments[0].range.end, DurationMs(2_500));
    assert_eq!(final_segments[1].range.start, DurationMs(4_000));
    assert_eq!(final_segments[1].range.end, DurationMs(5_000));
    assert_eq!(timeline.transcript().text, "hello batch hello batch");
    Ok(())
}
