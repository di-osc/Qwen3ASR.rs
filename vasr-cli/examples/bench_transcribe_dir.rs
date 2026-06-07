use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device};
use vasr_audio::AudioLoader;
use vasr_data::AudioSource;
use vasr_models::qwen3_asr::LoadOptions;
use vasr_runtime::{
    AsrModel, AsrOptions, FsmnVadModel, OfflinePipeline, Qwen3AsrModel, VadModel, VadOptions,
};
use vasr_server::{AsyncTranscribePipeline, TranscribeInput};

#[tokio::main]
async fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let usage = "usage: bench_transcribe_dir MODEL_DIR AUDIO_DIR [max_new_tokens] [dtype] [isq] [limit] [language] [device] [max_batch_size] [max_batch_audio_sec] [vad_model]";
    let model = args.next().context(usage)?;
    let audio_dir = args.next().context(usage)?;
    let max_new_tokens = args
        .next()
        .as_deref()
        .unwrap_or("256")
        .parse::<usize>()
        .context("failed to parse max_new_tokens")?;
    let dtype = parse_dtype(args.next().as_deref().unwrap_or("bf16"))?;
    let isq = args.next();
    let limit = args
        .next()
        .map(|value| value.parse::<usize>())
        .transpose()
        .context("failed to parse limit")?;
    let language = args
        .next()
        .filter(|value| !value.trim().is_empty() && !value.eq_ignore_ascii_case("none"));
    let device = resolve_device(args.next().as_deref().unwrap_or("cuda"))?;
    let max_batch_size = args
        .next()
        .as_deref()
        .unwrap_or("20")
        .parse::<usize>()
        .context("failed to parse max_batch_size")?;
    let max_batch_audio_sec = args
        .next()
        .as_deref()
        .unwrap_or("60")
        .parse::<f32>()
        .context("failed to parse max_batch_audio_sec")?;
    let vad_model = args.next();

    let mut files = audio_files(&audio_dir)?;
    if let Some(limit) = limit {
        files.truncate(limit);
    }
    if files.is_empty() {
        bail!("no audio files found in {audio_dir:?}");
    }

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
    let load_seconds = load_start.elapsed().as_secs_f64();
    let vad = FsmnVadModel::from_pretrained(vad_model.as_deref())?;
    let offline = Arc::new(OfflinePipeline {
        vad: Some(Arc::new(vad) as Arc<dyn VadModel>),
        asr,
    });
    let vad_options = VadOptions::default();
    let options = AsrOptions {
        language,
        max_new_tokens,
        max_batch_size,
        max_batch_audio_sec,
        ..AsrOptions::default()
    };
    let pipeline = AsyncTranscribePipeline::new(AudioLoader, offline, options.clone())
        .with_vad_options(vad_options)
        .with_stage_buffer(max_batch_size);

    let inputs = files
        .iter()
        .enumerate()
        .map(|(index, path)| TranscribeInput {
            index,
            source: AudioSource::Path(path.clone()),
        })
        .collect::<Vec<_>>();

    let batch_start = Instant::now();
    let outcomes = pipeline.transcribe_many(inputs).await;

    let mut total_audio_seconds = 0.0;
    let mut total_items = 0usize;
    let mut total_annotations = 0usize;

    for (path, outcome) in files.iter().zip(outcomes.iter()) {
        let audio_seconds = outcome.audio_seconds;
        total_audio_seconds += audio_seconds;
        total_items += 1;
        match &outcome.result {
            Ok(timeline) => {
                total_annotations += timeline.annotations.len();
                let text = timeline.transcript().text;
                println!(
                    "file={} audio_seconds={:.3} annotations={} text_chars={} text={:?}",
                    path.display(),
                    audio_seconds,
                    timeline.annotations.len(),
                    text.chars().count(),
                    text
                );
            }
            Err(error) => {
                println!(
                    "file={} audio_seconds={:.3} error={error} component={:?}",
                    path.display(),
                    audio_seconds,
                    outcome.bad_component
                );
            }
        }
    }

    let wall = batch_start.elapsed().as_secs_f64();
    let bad = outcomes
        .iter()
        .filter(|outcome| outcome.result.is_err())
        .count();
    println!(
        "summary device={} files={} model_load_seconds={:.3} audio_seconds={:.3} wall_seconds={:.3} speedup={:.3} rtf={:.4} throughput_items_per_second={:.3} annotations={} bad={} max_batch_size={} max_batch_audio_sec={}",
        device_label(&device),
        total_items,
        load_seconds,
        total_audio_seconds,
        wall,
        total_audio_seconds / wall.max(f64::EPSILON),
        wall / total_audio_seconds.max(f64::EPSILON),
        total_items as f64 / wall.max(f64::EPSILON),
        total_annotations,
        bad,
        max_batch_size,
        max_batch_audio_sec
    );
    Ok(())
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

fn parse_dtype(value: &str) -> Result<DType> {
    match value.trim().to_ascii_lowercase().as_str() {
        "f32" => Ok(DType::F32),
        "f16" => Ok(DType::F16),
        "bf16" => Ok(DType::BF16),
        other => bail!("unknown dtype {other:?}; expected f32, f16, or bf16"),
    }
}

fn resolve_device(value: &str) -> Result<Device> {
    match value.trim().to_ascii_lowercase().as_str() {
        "cpu" => Ok(Device::Cpu),
        "metal" => {
            #[cfg(feature = "metal")]
            {
                Device::new_metal(0)
                    .map_err(|err| anyhow::anyhow!("failed to create Metal device 0: {err}"))
            }
            #[cfg(not(feature = "metal"))]
            {
                bail!("metal requested but vasr-cli was built without the metal feature")
            }
        }
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                Device::new_cuda(0)
                    .map_err(|err| anyhow::anyhow!("failed to create CUDA device 0: {err}"))
            }
            #[cfg(not(feature = "cuda"))]
            {
                bail!("cuda requested but vasr-cli was built without the cuda feature")
            }
        }
        other => bail!("unknown device {other:?}; expected cpu, metal, or cuda"),
    }
}

fn device_label(device: &Device) -> &'static str {
    if device.is_cpu() {
        "cpu"
    } else if device.is_metal() {
        "metal"
    } else if device.is_cuda() {
        "cuda"
    } else {
        "unknown"
    }
}
