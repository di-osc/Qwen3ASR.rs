use anyhow::Result;
use std::sync::Arc;
use vasr_data::{
    Annotation, AnnotationPayload, AnnotationSource, AnnotationStatus, AudioChunk, DurationMs,
    TextSegment, TimeRange, Timeline, Waveform,
};
use vasr_runtime::{
    AsrModel, AsrOptions, OfflinePipeline, StreamingAsrModel, StreamingVadModel, VadModel,
    VadOptions, VadSegment,
};

struct FakeAsr;

impl AsrModel for FakeAsr {
    fn transcribe(&self, waveform: &Waveform, _options: &AsrOptions) -> Result<Timeline> {
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

struct FakeVad;

impl VadModel for FakeVad {
    fn detect(&self, waveform: &Waveform, _options: &VadOptions) -> Result<Vec<VadSegment>> {
        Ok(vec![VadSegment {
            range: TimeRange::new(
                DurationMs(0),
                DurationMs(waveform.duration_ms().round() as u64),
            ),
            probability: 0.9,
        }])
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
    let pipeline = OfflinePipeline {
        vad: Some(Box::new(FakeVad)),
        asr: Arc::new(FakeAsr),
    };

    let timeline = pipeline.transcribe(&waveform, &AsrOptions::default())?;

    assert!(timeline.annotations.len() >= 2);
    assert_eq!(timeline.transcript().text, "hello world");
    Ok(())
}
