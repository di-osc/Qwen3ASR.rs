use std::sync::Arc;

use anyhow::Result;
use vasr_data::{Annotation, DurationMs, Timeline, Waveform};

use crate::model::{
    AsrModel, AsrOptions, StreamingAsrModel, StreamingVadModel, VadModel, VadOptions,
};

pub struct OfflinePipeline {
    pub vad: Option<Box<dyn VadModel>>,
    pub asr: Arc<dyn AsrModel>,
}

impl OfflinePipeline {
    pub fn transcribe(&self, waveform: &Waveform, options: &AsrOptions) -> Result<Timeline> {
        let mut timeline = Timeline::new("offline_audio");
        if let Some(vad) = &self.vad {
            let segments = vad.detect(waveform, &VadOptions::default())?;
            for segment in &segments {
                timeline.push(vasr_data::Annotation::new(
                    segment.range,
                    vasr_data::AnnotationPayload::Speech,
                    vasr_data::AnnotationSource::Model("vad".to_string()),
                    vasr_data::AnnotationStatus::Final,
                ));
            }
            if !segments.is_empty() {
                let slices = segments
                    .iter()
                    .map(|segment| waveform.slice_ms(segment.range.start.0, segment.range.end.0))
                    .collect::<Vec<_>>();
                let asr_timelines = self.asr.transcribe_batch(&slices, options)?;
                for (segment, asr_timeline) in segments.into_iter().zip(asr_timelines) {
                    timeline.annotations.extend(offset_annotations(
                        asr_timeline.annotations,
                        segment.range.start,
                    ));
                }
                return Ok(timeline);
            }
        }
        let asr_timeline = self.asr.transcribe(waveform, options)?;
        timeline.annotations.extend(asr_timeline.annotations);
        Ok(timeline)
    }
}

fn offset_annotations(annotations: Vec<Annotation>, offset: DurationMs) -> Vec<Annotation> {
    annotations
        .into_iter()
        .map(|mut annotation| {
            annotation.range.start.0 = annotation.range.start.0.saturating_add(offset.0);
            annotation.range.end.0 = annotation.range.end.0.saturating_add(offset.0);
            annotation
        })
        .collect()
}

pub struct RealtimePipeline {
    pub vad: Box<dyn StreamingVadModel>,
    pub asr: Box<dyn StreamingAsrModel>,
}

impl RealtimePipeline {
    pub fn push_chunk(&mut self, chunk: &vasr_data::AudioChunk) -> Result<Vec<Annotation>> {
        let mut annotations = self.vad.push_chunk(chunk)?;
        annotations.extend(self.asr.push_chunk(chunk)?);
        Ok(annotations)
    }

    pub fn finish(&mut self) -> Result<Vec<Annotation>> {
        let mut annotations = self.vad.finish()?;
        annotations.extend(self.asr.finish()?);
        Ok(annotations)
    }
}
