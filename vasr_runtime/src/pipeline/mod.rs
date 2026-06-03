use std::sync::Arc;

use anyhow::Result;
use vasr_data::{Annotation, Timeline, Waveform};

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
            for segment in vad.detect(waveform, &VadOptions::default())? {
                timeline.push(vasr_data::Annotation::new(
                    segment.range,
                    vasr_data::AnnotationPayload::Speech,
                    vasr_data::AnnotationSource::Model("vad".to_string()),
                    vasr_data::AnnotationStatus::Final,
                ));
            }
        }
        let asr_timeline = self.asr.transcribe(waveform, options)?;
        timeline.annotations.extend(asr_timeline.annotations);
        Ok(timeline)
    }
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
