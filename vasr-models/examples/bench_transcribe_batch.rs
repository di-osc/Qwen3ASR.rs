use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use vasr_models::qwen3_asr::model::isq_linear::{isq_quantize_time_us, reset_isq_quantize_time};
use vasr_models::qwen3_asr::{AudioInput, Batch, LoadOptions, Qwen3Asr, TranscribeOptions};
#[cfg(feature = "paged-attn")]
use vasr_paged_attn::{PagedCacheConfig, PagedCacheMemory};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model = args
        .next()
        .unwrap_or_else(|| "Qwen/Qwen3-ASR-0.6B".to_string());
    let audio = args
        .next()
        .unwrap_or_else(|| "fixtures/audio/asr_en_16k.wav".to_string());
    let batch_size = args
        .next()
        .as_deref()
        .unwrap_or("1")
        .parse::<usize>()
        .context("failed to parse batch_size")?;
    let repeats = args
        .next()
        .as_deref()
        .unwrap_or("3")
        .parse::<usize>()
        .context("failed to parse repeats")?;
    let max_new_tokens = args
        .next()
        .as_deref()
        .unwrap_or("128")
        .parse::<usize>()
        .context("failed to parse max_new_tokens")?;
    let dtype = args
        .next()
        .as_deref()
        .map(parse_dtype)
        .transpose()?
        .unwrap_or(DType::BF16);
    let isq = args.next();

    if batch_size == 0 {
        anyhow::bail!("batch_size must be greater than zero");
    }
    let audio_list: Vec<String> = audio
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToOwned::to_owned)
        .collect();
    if audio_list.is_empty() {
        anyhow::bail!("audio list is empty");
    }

    let device = default_gpu_device()?;
    let use_flash_attn = cfg!(feature = "flash-attn") && device.is_cuda();
    reset_isq_quantize_time();
    let start_load = Instant::now();
    let model = Qwen3Asr::from_pretrained(
        &model,
        &device,
        &LoadOptions {
            dtype,
            use_flash_attn,
            isq,
            #[cfg(feature = "paged-attn")]
            paged_cache: default_paged_cache_config(&device),
        },
    )
    .context("failed to load model")?;
    let load_ms = start_load.elapsed().as_secs_f64() * 1000.0;
    let isq_quant_ms = isq_quantize_time_us() as f64 / 1000.0;
    println!("load_ms={load_ms:.3} isq_quant_ms={isq_quant_ms:.3}");

    let opts = TranscribeOptions {
        context: Batch::one(String::new()),
        language: Batch::one(Some("English".to_string())),
        return_timestamps: false,
        max_new_tokens,
        max_batch_size: batch_size,
        max_batch_audio_sec: 60.0,
        chunk_max_sec: None,
        bucket_by_length: false,
    };

    let audio_path = Path::new(&audio);
    for run in 0..repeats {
        let inputs: Vec<AudioInput<'_>> = if audio_list.len() == 1 {
            (0..batch_size)
                .map(|_| AudioInput::Path(audio_path))
                .collect()
        } else {
            if audio_list.len() != batch_size {
                anyhow::bail!(
                    "audio list length {} must match batch_size {} when using multiple files",
                    audio_list.len(),
                    batch_size
                );
            }
            audio_list
                .iter()
                .map(|path| AudioInput::Path(Path::new(path.as_str())))
                .collect()
        };
        let start = Instant::now();
        let (out, timings) = model.transcribe_timed(inputs, opts.clone())?;
        let wall = start.elapsed();
        let decode_s = timings.generation.decode_us as f64 / 1_000_000.0;
        let text_decode_tokens = out
            .iter()
            .map(|o| {
                model
                    .processor()
                    .tokenizer
                    .encode(o.text.as_str())
                    .map(|ids| ids.len())
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .sum::<usize>();
        let timed_tokens = if timings.generation.tokens_generated > 0 {
            timings.generation.tokens_generated
        } else {
            text_decode_tokens
        };
        let batch_tokens_per_s = if decode_s > 0.0 {
            timed_tokens as f64 / decode_s
        } else {
            let decode_ms = wall.as_secs_f64() * 1000.0
                - timings.audio_encoder_us as f64 / 1000.0
                - timings.generation.prefill_us as f64 / 1000.0;
            if decode_ms <= 0.0 {
                0.0
            } else {
                text_decode_tokens as f64 / (decode_ms / 1000.0)
            }
        };
        let per_sequence_tokens_per_s = batch_tokens_per_s / batch_size as f64;
        let first_text = out.first().map(|o| o.text.as_str()).unwrap_or("");
        let texts: Vec<&str> = out.iter().map(|o| o.text.as_str()).collect();
        println!(
            "run={} batch_size={} wall_ms={:.3} timed_total_ms={:.3} audio_encoder_ms={:.3} prefill_ms={:.3} decode_ms={:.3} steps={} tokens={} batch_decode_tokens_per_s={:.3} per_sequence_decode_tokens_per_s={:.3} first_text={:?} texts={:?}",
            run + 1,
            batch_size,
            wall.as_secs_f64() * 1000.0,
            timings.total_us as f64 / 1000.0,
            timings.audio_encoder_us as f64 / 1000.0,
            timings.generation.prefill_us as f64 / 1000.0,
            timings.generation.decode_us as f64 / 1000.0,
            timings.generation.steps,
            timed_tokens,
            batch_tokens_per_s,
            per_sequence_tokens_per_s,
            first_text,
            texts
        );
    }

    Ok(())
}

fn parse_dtype(value: &str) -> Result<DType> {
    match value.trim().to_ascii_lowercase().as_str() {
        "f32" => Ok(DType::F32),
        "f16" => Ok(DType::F16),
        "bf16" => Ok(DType::BF16),
        other => anyhow::bail!("unknown dtype {other:?}; expected f32, f16, or bf16"),
    }
}

#[cfg(feature = "paged-attn")]
fn default_paged_cache_config(device: &Device) -> Option<PagedCacheConfig> {
    Some(PagedCacheConfig {
        block_size: 32,
        memory: if device.is_cuda() {
            PagedCacheMemory::GpuMemoryFraction(0.8)
        } else {
            PagedCacheMemory::ContextSize(100_000)
        },
    })
}

fn default_gpu_device() -> Result<Device> {
    #[cfg(feature = "cuda")]
    {
        return Device::new_cuda_with_stream(0).context("failed to create CUDA device 0");
    }

    #[cfg(all(not(feature = "cuda"), feature = "metal"))]
    {
        return Device::new_metal(0).context("failed to create Metal device 0");
    }

    #[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
    {
        anyhow::bail!("bench_transcribe_batch requires a GPU build; enable `cuda` or `metal`");
    }
}
