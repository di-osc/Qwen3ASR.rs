use std::sync::Arc;

use anyhow::Result;
use vasr_data::{Annotation, DurationMs, Timeline, Waveform};

use crate::model::{
    AsrModel, AsrOptions, StreamingAsrModel, StreamingVadModel, VadModel, VadOptions, VadSegment,
};

pub struct OfflinePipeline {
    pub vad: Option<Arc<dyn VadModel>>,
    pub asr: Arc<dyn AsrModel>,
}

/// Output of the VAD stage, ready for batched ASR.
#[derive(Debug, Clone)]
pub struct VadPrepared {
    pub speech_annotations: Vec<Annotation>,
    pub segments: Vec<VadSegment>,
    pub slices: Vec<Waveform>,
}

/// Result of the VAD stage. `Disabled` is the only case that should fall back
/// to raw ASR; `NoSpeech` means VAD ran and found nothing to recognize.
#[derive(Debug, Clone)]
pub enum VadPreparation {
    Disabled,
    NoSpeech,
    Speech(VadPrepared),
}

impl OfflinePipeline {
    pub fn transcribe(&self, waveform: &Waveform, options: &AsrOptions) -> Result<Timeline> {
        self.transcribe_with_vad_options(waveform, options, &VadOptions::default())
    }

    pub fn transcribe_with_vad_options(
        &self,
        waveform: &Waveform,
        options: &AsrOptions,
        vad_options: &VadOptions,
    ) -> Result<Timeline> {
        let mut timeline = Timeline::new("offline_audio");
        match self.prepare_vad_stage(waveform, vad_options)? {
            VadPreparation::Speech(prepared) => {
                timeline
                    .annotations
                    .extend(self.transcribe_prepared(prepared, options)?);
            }
            VadPreparation::NoSpeech => {}
            VadPreparation::Disabled => {
                let asr_timeline = self.asr.transcribe(waveform, options)?;
                timeline.annotations.extend(asr_timeline.annotations);
            }
        }
        Ok(timeline)
    }

    /// Run VAD and slice the waveform, preserving whether VAD was disabled or
    /// ran and found no speech.
    pub fn prepare_vad_stage(
        &self,
        waveform: &Waveform,
        vad_options: &VadOptions,
    ) -> Result<VadPreparation> {
        let Some(vad) = &self.vad else {
            return Ok(VadPreparation::Disabled);
        };
        let segments = vad.detect(waveform, vad_options)?;
        if segments.is_empty() {
            return Ok(VadPreparation::NoSpeech);
        }
        let speech_annotations = segments
            .iter()
            .map(|segment| {
                vasr_data::Annotation::new(
                    segment.range,
                    vasr_data::AnnotationPayload::Speech,
                    vasr_data::AnnotationSource::Model("vad".to_string()),
                    vasr_data::AnnotationStatus::Final,
                )
            })
            .collect();
        let asr_segments = merge_vad_segments_for_asr(
            segments.as_slice(),
            vad_options.merge_max_gap_ms,
            vad_options.merge_max_segment_ms,
        );
        let slices = asr_segments
            .iter()
            .map(|segment| waveform.slice_ms(segment.range.start.0, segment.range.end.0))
            .collect();
        Ok(VadPreparation::Speech(VadPrepared {
            speech_annotations,
            segments: asr_segments,
            slices,
        }))
    }

    /// Run VAD and slice the waveform. Returns `None` when VAD is disabled or finds no speech.
    pub fn prepare_vad(
        &self,
        waveform: &Waveform,
        vad_options: &VadOptions,
    ) -> Result<Option<VadPrepared>> {
        match self.prepare_vad_stage(waveform, vad_options)? {
            VadPreparation::Speech(prepared) => Ok(Some(prepared)),
            VadPreparation::Disabled | VadPreparation::NoSpeech => Ok(None),
        }
    }

    /// Run ASR on VAD slices and offset segment annotations back to the original timeline.
    pub fn transcribe_prepared(
        &self,
        prepared: VadPrepared,
        options: &AsrOptions,
    ) -> Result<Vec<Annotation>> {
        let VadPrepared {
            speech_annotations,
            segments,
            slices,
        } = prepared;
        let asr_timelines = self.asr.transcribe_batch(&slices, options)?;
        let mut annotations = speech_annotations;
        for (segment, asr_timeline) in segments.into_iter().zip(asr_timelines) {
            annotations.extend(offset_annotations(
                asr_timeline.annotations,
                segment.range.start,
            ));
        }
        Ok(annotations)
    }
}

fn merge_vad_segments_for_asr(
    segments: &[VadSegment],
    max_gap_ms: u64,
    max_segment_ms: u64,
) -> Vec<VadSegment> {
    if segments.is_empty() || max_gap_ms == 0 || max_segment_ms == 0 {
        return segments.to_vec();
    }

    let mut sorted = segments.to_vec();
    sorted.sort_by_key(|segment| (segment.range.start, segment.range.end));
    let mut merged: Vec<VadSegment> = Vec::with_capacity(sorted.len());

    for segment in sorted {
        let Some(last) = merged.last_mut() else {
            merged.push(segment);
            continue;
        };
        let gap = segment.range.start.0.saturating_sub(last.range.end.0);
        let merged_duration = segment.range.end.0.saturating_sub(last.range.start.0);
        if gap <= max_gap_ms && merged_duration <= max_segment_ms {
            last.range.end = last.range.end.max(segment.range.end);
            last.probability = last.probability.max(segment.probability);
        } else {
            merged.push(segment);
        }
    }

    merged
}

pub fn offset_annotations(annotations: Vec<Annotation>, offset: DurationMs) -> Vec<Annotation> {
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

#[cfg(feature = "async")]
pub mod r#async;
