use anyhow::{Context, Result};
use candle_core::Device;
use vasr_data::{
    Annotation, AnnotationPayload, AnnotationSource, AnnotationStatus, TextSegment, TimeRange,
    Timeline, Token, Waveform,
};
use vasr_models::qwen3_asr::{
    AudioInput, Batch, LoadOptions, Qwen3Asr, StreamOptions as RuntimeStreamOptions,
    TranscribeOptions as RuntimeTranscribeOptions,
};

use crate::model::{AsrModel, AsrOptions, StreamingAsrModel};

pub struct Qwen3AsrModel {
    inner: Qwen3Asr,
}

impl Qwen3AsrModel {
    pub fn from_pretrained(
        model_id_or_path: &str,
        device: &Device,
        load_options: &LoadOptions,
    ) -> Result<Self> {
        let inner = Qwen3Asr::from_pretrained(model_id_or_path, device, load_options)
            .with_context(|| format!("failed to load Qwen3-ASR model from {model_id_or_path:?}"))?;
        Ok(Self { inner })
    }

    pub fn inner(&self) -> &Qwen3Asr {
        &self.inner
    }
}

impl AsrModel for Qwen3AsrModel {
    fn transcribe(&self, waveform: &Waveform, options: &AsrOptions) -> Result<Timeline> {
        let mut timelines = self.transcribe_batch(std::slice::from_ref(waveform), options)?;
        timelines
            .pop()
            .ok_or_else(|| anyhow::anyhow!("Qwen3-ASR returned no transcription"))
    }

    fn transcribe_batch(
        &self,
        waveforms: &[Waveform],
        options: &AsrOptions,
    ) -> Result<Vec<Timeline>> {
        let runtime_options = RuntimeTranscribeOptions {
            context: Batch::one(options.context.clone()),
            language: Batch::one(options.language.clone()),
            return_timestamps: false,
            max_new_tokens: options.max_new_tokens,
            max_batch_size: RuntimeTranscribeOptions::default().max_batch_size,
            chunk_max_sec: None,
            bucket_by_length: false,
        };
        let audio = waveforms
            .iter()
            .map(|waveform| AudioInput::Waveform {
                samples: &waveform.samples,
                sample_rate: waveform.sample_rate,
            })
            .collect::<Vec<_>>();
        let outputs = self.inner.transcribe(audio, runtime_options)?;
        if outputs.len() != waveforms.len() {
            anyhow::bail!(
                "Qwen3-ASR returned {} transcriptions for {} inputs",
                outputs.len(),
                waveforms.len()
            );
        }
        Ok(outputs
            .into_iter()
            .zip(waveforms)
            .map(|(output, waveform)| {
                let mut timeline = Timeline::new("qwen3_asr_audio");
                let range = TimeRange::new(
                    vasr_data::DurationMs(0),
                    vasr_data::DurationMs(waveform.duration_ms().round() as u64),
                );
                if !output.language.is_empty() {
                    timeline.push(Annotation::new(
                        range,
                        AnnotationPayload::Language(output.language.clone()),
                        AnnotationSource::Model("qwen3_asr".to_string()),
                        AnnotationStatus::Final,
                    ));
                }
                timeline.push(Annotation::new(
                    range,
                    AnnotationPayload::Segment(TextSegment {
                        text: output.text,
                        tokens: Vec::new(),
                        language: Some(output.language),
                    }),
                    AnnotationSource::Model("qwen3_asr".to_string()),
                    AnnotationStatus::Final,
                ));
                timeline
            })
            .collect())
    }

    fn start_stream(&self, options: &AsrOptions) -> Result<Box<dyn StreamingAsrModel>> {
        let stream = self.inner.start_stream(RuntimeStreamOptions {
            context: options.context.clone(),
            language: options.language.clone(),
            max_new_tokens: options.max_new_tokens,
            ..RuntimeStreamOptions::default()
        })?;
        Ok(Box::new(Qwen3AsrStreamModel {
            inner: Some(stream),
        }))
    }
}

struct Qwen3AsrStreamModel {
    inner: Option<vasr_models::qwen3_asr::AsrStream>,
}

impl StreamingAsrModel for Qwen3AsrStreamModel {
    fn push_chunk(&mut self, chunk: &vasr_data::AudioChunk) -> Result<Vec<Annotation>> {
        let stream = self
            .inner
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Qwen3-ASR stream is already finished"))?;
        let output = stream.push_audio_chunk(&AudioInput::Waveform {
            samples: &chunk.waveform.samples,
            sample_rate: chunk.waveform.sample_rate,
        })?;
        Ok(output
            .map(|output| {
                vec![Annotation::new(
                    chunk.range,
                    AnnotationPayload::Segment(TextSegment {
                        text: output.text,
                        tokens: Vec::<Token>::new(),
                        language: Some(output.language),
                    }),
                    AnnotationSource::Model("qwen3_asr".to_string()),
                    AnnotationStatus::Partial,
                )]
            })
            .unwrap_or_default())
    }

    fn finish(&mut self) -> Result<Vec<Annotation>> {
        let stream = self
            .inner
            .take()
            .ok_or_else(|| anyhow::anyhow!("Qwen3-ASR stream is already finished"))?;
        let output = stream.finish()?;
        Ok(vec![Annotation::new(
            TimeRange::default(),
            AnnotationPayload::Segment(TextSegment {
                text: output.text,
                tokens: Vec::new(),
                language: Some(output.language),
            }),
            AnnotationSource::Model("qwen3_asr".to_string()),
            AnnotationStatus::Final,
        )])
    }
}
