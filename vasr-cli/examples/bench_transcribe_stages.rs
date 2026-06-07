use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device};
use vasr_audio::{AudioLoadOptions, AudioLoader};
use vasr_data::{AnnotationPayload, AudioChunk, AudioSource, DurationMs, TimeRange};
use vasr_models::qwen3_asr::LoadOptions;
use vasr_runtime::{
    AsrModel, AsrOptions, FsmnVadModel, FsmnVadTiming, Qwen3AsrModel, VadModel, VadOptions,
};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model = args.next().context(
        "usage: bench_transcribe_stages MODEL_DIR AUDIO_DIR [max_new_tokens] [dtype] [isq] [limit] [mode]",
    )?;
    let audio_dir = args.next().context(
        "usage: bench_transcribe_stages MODEL_DIR AUDIO_DIR [max_new_tokens] [dtype] [isq] [limit] [mode]",
    )?;
    let max_new_tokens = args
        .next()
        .as_deref()
        .unwrap_or("128")
        .parse::<usize>()
        .context("failed to parse max_new_tokens")?;
    let dtype = parse_dtype(args.next().as_deref().unwrap_or("bf16"))?;
    let isq = parse_optional(args.next());
    let limit = args
        .next()
        .map(|value| value.parse::<usize>())
        .transpose()
        .context("failed to parse limit")?;
    let mode = args.next().unwrap_or_else(|| "all".to_string());
    let run_asr = match mode.as_str() {
        "all" => true,
        "vad-only" | "vad-streaming" => false,
        other => bail!("unknown mode {other:?}; expected all, vad-only, or vad-streaming"),
    };
    let vad_model = args.next();
    let vad_threshold = args
        .next()
        .map(|value| value.parse::<f32>())
        .transpose()
        .context("failed to parse vad threshold")?
        .unwrap_or_else(|| VadOptions::default().threshold);
    let vad_options = VadOptions {
        threshold: vad_threshold,
        ..VadOptions::default()
    };

    let mut files = audio_files(&audio_dir)?;
    if let Some(limit) = limit {
        files.truncate(limit);
    }
    if files.is_empty() {
        bail!("no audio files found in {audio_dir:?}");
    }

    let loader = AudioLoader;
    let vad = FsmnVadModel::from_pretrained(vad_model.as_deref())?;
    let asr = if run_asr {
        let device = default_device()?;
        let load_start = Instant::now();
        let asr: Arc<dyn AsrModel> = Arc::new(Qwen3AsrModel::from_pretrained(
            &model,
            &device,
            &LoadOptions {
                dtype,
                use_flash_attn: false,
                isq,
                #[cfg(any(feature = "cuda", feature = "cuda-paged-attn", feature = "metal"))]
                paged_cache: None,
            },
        )?);
        println!(
            "model_load_seconds={:.3}",
            load_start.elapsed().as_secs_f64()
        );
        Some(asr)
    } else {
        None
    };
    let options = AsrOptions {
        max_new_tokens,
        ..AsrOptions::default()
    };

    let mut summary = StageSummary::default();
    for path in files {
        let item_start = Instant::now();
        let load_start = Instant::now();
        let waveform = loader.load(
            &AudioSource::Path(path.clone()),
            &AudioLoadOptions::default(),
        )?;
        let load_seconds = load_start.elapsed().as_secs_f64();

        let vad_start = Instant::now();
        let (segments, vad_timing) = if mode == "vad-streaming" {
            (detect_streaming(&vad, &waveform, &vad_options)?, None)
        } else {
            let detection = vad.detect_with_timing(&waveform, &vad_options)?;
            (detection.segments, Some(detection.timing))
        };
        let vad_seconds = vad_start.elapsed().as_secs_f64();
        let slices = segments
            .iter()
            .map(|segment| waveform.slice_ms(segment.range.start.0, segment.range.end.0))
            .collect::<Vec<_>>();
        let speech_seconds = slices
            .iter()
            .map(|slice| slice.duration_seconds())
            .sum::<f64>();

        let asr_start = Instant::now();
        let output_count = if let Some(asr) = &asr {
            asr.transcribe_batch(&slices, &options)?.len()
        } else {
            0
        };
        let asr_seconds = asr_start.elapsed().as_secs_f64();
        let wall_seconds = item_start.elapsed().as_secs_f64();
        let audio_seconds = waveform.duration_seconds();

        summary.files += 1;
        summary.audio_seconds += audio_seconds;
        summary.speech_seconds += speech_seconds;
        summary.load_seconds += load_seconds;
        summary.vad_seconds += vad_seconds;
        summary.asr_seconds += asr_seconds;
        summary.wall_seconds += wall_seconds;
        summary.segments += segments.len();
        if let Some(timing) = vad_timing {
            summary.vad_internal.pcm_seconds += timing.pcm_seconds;
            summary.vad_internal.frontend_seconds += timing.frontend_seconds;
            summary.vad_internal.forward_seconds += timing.forward_seconds;
            summary.vad_internal.segmenter_seconds += timing.segmenter_seconds;
            summary.vad_internal.chunks += timing.chunks;
        }

        println!(
            "file={} audio_seconds={:.3} speech_seconds={:.3} segments={} load_seconds={:.3} vad_seconds={:.3} asr_seconds={:.3} wall_seconds={:.3} asr_speedup_on_audio={:.3} asr_speedup_on_speech={:.3} output_count={}",
            path.display(),
            audio_seconds,
            speech_seconds,
            segments.len(),
            load_seconds,
            vad_seconds,
            asr_seconds,
            wall_seconds,
            audio_seconds / asr_seconds.max(f64::EPSILON),
            speech_seconds / asr_seconds.max(f64::EPSILON),
            output_count
        );
    }
    println!(
        "summary files={} audio_seconds={:.3} speech_seconds={:.3} segments={} load_seconds={:.3} vad_seconds={:.3} asr_seconds={:.3} wall_seconds={:.3} speedup={:.3} rtf={:.4} asr_audio_speedup={:.3} asr_speech_speedup={:.3} vad_chunks={} vad_pcm_seconds={:.3} vad_frontend_seconds={:.3} vad_forward_seconds={:.3} vad_segmenter_seconds={:.3}",
        summary.files,
        summary.audio_seconds,
        summary.speech_seconds,
        summary.segments,
        summary.load_seconds,
        summary.vad_seconds,
        summary.asr_seconds,
        summary.wall_seconds,
        summary.audio_seconds / summary.wall_seconds.max(f64::EPSILON),
        summary.wall_seconds / summary.audio_seconds.max(f64::EPSILON),
        summary.audio_seconds / summary.asr_seconds.max(f64::EPSILON),
        summary.speech_seconds / summary.asr_seconds.max(f64::EPSILON),
        summary.vad_internal.chunks,
        summary.vad_internal.pcm_seconds,
        summary.vad_internal.frontend_seconds,
        summary.vad_internal.forward_seconds,
        summary.vad_internal.segmenter_seconds,
    );
    Ok(())
}

#[derive(Default)]
struct StageSummary {
    files: usize,
    segments: usize,
    audio_seconds: f64,
    speech_seconds: f64,
    load_seconds: f64,
    vad_seconds: f64,
    asr_seconds: f64,
    wall_seconds: f64,
    vad_internal: FsmnVadTiming,
}

fn audio_files(dir: impl AsRef<Path>) -> Result<Vec<PathBuf>> {
    let mut files = fs::read_dir(dir)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    files.retain(|path| {
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| {
                matches!(
                    ext.to_ascii_lowercase().as_str(),
                    "wav" | "mp3" | "flac" | "ogg"
                )
            })
            .unwrap_or(false)
    });
    files.sort();
    Ok(files)
}

fn parse_optional(value: Option<String>) -> Option<String> {
    value.and_then(|value| {
        if value == "none" || value == "-" {
            None
        } else {
            Some(value)
        }
    })
}

fn parse_dtype(value: &str) -> Result<DType> {
    match value.trim().to_ascii_lowercase().as_str() {
        "f32" => Ok(DType::F32),
        "f16" => Ok(DType::F16),
        "bf16" => Ok(DType::BF16),
        other => bail!("unknown dtype {other:?}; expected f32, f16, or bf16"),
    }
}

fn default_device() -> Result<Device> {
    #[cfg(feature = "metal")]
    {
        return Device::new_metal(0)
            .map_err(|err| anyhow::anyhow!("failed to create Metal device 0: {err}"));
    }
    #[cfg(not(feature = "metal"))]
    {
        Ok(Device::Cpu)
    }
}

fn detect_streaming(
    vad: &FsmnVadModel,
    waveform: &vasr_data::Waveform,
    options: &VadOptions,
) -> Result<Vec<vasr_runtime::VadSegment>> {
    let mut stream = vad.start_stream(options)?;
    let mut annotations = Vec::new();
    for (idx, samples) in waveform.samples.chunks(512).enumerate() {
        let start = idx * 512;
        let end = start + samples.len();
        let chunk = AudioChunk {
            stream_id: "bench".to_string(),
            waveform: vasr_data::Waveform::new(samples.to_vec(), waveform.sample_rate),
            is_start: idx == 0,
            is_last: end == waveform.samples.len(),
            range: TimeRange::new(
                sample_to_ms(start, waveform.sample_rate),
                sample_to_ms(end, waveform.sample_rate),
            ),
        };
        annotations.extend(stream.push_chunk(&chunk)?);
    }
    annotations.extend(stream.finish()?);
    Ok(annotations
        .into_iter()
        .filter_map(|annotation| match annotation.payload {
            AnnotationPayload::Speech => Some(vasr_runtime::VadSegment {
                range: annotation.range,
                probability: 0.9,
            }),
            _ => None,
        })
        .collect())
}

fn sample_to_ms(sample: usize, sample_rate: u32) -> DurationMs {
    DurationMs((sample as u64).saturating_mul(1000) / u64::from(sample_rate))
}
