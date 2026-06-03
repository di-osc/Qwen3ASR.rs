use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device};
use vasr_audio::{AudioLoadOptions, AudioLoader};
use vasr_data::AudioSource;
use vasr_models::qwen3_asr::LoadOptions;
use vasr_runtime::{AsrModel, AsrOptions, Qwen3AsrModel, SileroVadModel, VadModel, VadOptions};

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
        "vad-only" => false,
        other => bail!("unknown mode {other:?}; expected all or vad-only"),
    };

    let mut files = audio_files(&audio_dir)?;
    if let Some(limit) = limit {
        files.truncate(limit);
    }
    if files.is_empty() {
        bail!("no audio files found in {audio_dir:?}");
    }

    let loader = AudioLoader;
    let vad = SileroVadModel::from_default_model()?;
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
                #[cfg(any(feature = "metal-paged-attn", feature = "cuda-paged-attn"))]
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
        let segments = vad.detect(&waveform, &VadOptions::default())?;
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
        "summary files={} audio_seconds={:.3} speech_seconds={:.3} segments={} load_seconds={:.3} vad_seconds={:.3} asr_seconds={:.3} wall_seconds={:.3} speedup={:.3} rtf={:.4} asr_audio_speedup={:.3} asr_speech_speedup={:.3}",
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
