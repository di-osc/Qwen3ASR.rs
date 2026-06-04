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
    AsrModel, AsrOptions, OfflinePipeline, Qwen3AsrModel, SileroVadModel, VadModel,
};
use vasr_server::{AsyncTranscribePipeline, TranscribeInput};

#[tokio::main]
async fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let usage = "usage: bench_transcribe_dir MODEL_DIR AUDIO_DIR [max_new_tokens] [dtype] [isq] [limit] [language]";
    let model = args.next().context(usage)?;
    let audio_dir = args.next().context(usage)?;
    let max_new_tokens = args
        .next()
        .as_deref()
        .unwrap_or("128")
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

    let mut files = audio_files(&audio_dir)?;
    if let Some(limit) = limit {
        files.truncate(limit);
    }
    if files.is_empty() {
        bail!("no audio files found in {audio_dir:?}");
    }

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
    let load_seconds = load_start.elapsed().as_secs_f64();
    let vad = SileroVadModel::from_default_model()?;
    let offline = Arc::new(OfflinePipeline {
        vad: Some(Arc::new(vad) as Arc<dyn VadModel>),
        asr,
    });
    let options = AsrOptions {
        language,
        max_new_tokens,
        ..AsrOptions::default()
    };
    let pipeline = AsyncTranscribePipeline::new(AudioLoader, offline, options);

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
    println!(
        "summary files={} model_load_seconds={:.3} audio_seconds={:.3} wall_seconds={:.3} speedup={:.3} rtf={:.4} annotations={}",
        total_items,
        load_seconds,
        total_audio_seconds,
        wall,
        total_audio_seconds / wall.max(f64::EPSILON),
        wall / total_audio_seconds.max(f64::EPSILON),
        total_annotations
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
