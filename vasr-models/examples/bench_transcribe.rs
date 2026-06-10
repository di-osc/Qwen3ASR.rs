use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use vasr_models::qwen3_asr::audio::normalize::normalize_audio_input;
use vasr_models::qwen3_asr::model::isq_linear::{isq_quantize_time_us, reset_isq_quantize_time};
use vasr_models::qwen3_asr::{AudioInput, Batch, LoadOptions, Qwen3Asr, TranscribeOptions};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model = args
        .next()
        .unwrap_or_else(|| "Qwen/Qwen3-ASR-0.6B".to_string());
    let audio = args
        .next()
        .unwrap_or_else(|| "fixtures/audio/asr_en_16k.wav".to_string());
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
    let isq = args
        .next()
        .filter(|value| !matches!(value.as_str(), "-" | "none" | "None" | "NONE"));
    let language = args
        .next()
        .filter(|value| !matches!(value.as_str(), "-" | "none" | "None" | "NONE"));
    let source_mode = args.next().unwrap_or_else(|| "path".to_string());
    let slice_start_ms = args
        .next()
        .map(|value| {
            value
                .parse::<u64>()
                .context("failed to parse slice_start_ms")
        })
        .transpose()?;
    let slice_end_ms = args
        .next()
        .map(|value| value.parse::<u64>().context("failed to parse slice_end_ms"))
        .transpose()?;

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
            paged_cache: None,
        },
    )
    .context("failed to load model")?;
    let load_ms = start_load.elapsed().as_secs_f64() * 1000.0;
    let isq_quant_ms = isq_quantize_time_us() as f64 / 1000.0;
    println!("load_ms={load_ms:.3} isq_quant_ms={isq_quant_ms:.3}");

    let opts = TranscribeOptions {
        context: Batch::one(String::new()),
        language: Batch::one(language.clone()),
        return_timestamps: false,
        max_new_tokens,
        max_batch_size: 1,
        max_batch_audio_sec: 0.0,
        chunk_max_sec: None,
        bucket_by_length: false,
    };

    let audio_path = Path::new(&audio);
    let waveform_samples = if source_mode == "waveform" {
        let mut samples = normalize_audio_input(&AudioInput::Path(audio_path))?;
        if let (Some(start_ms), Some(end_ms)) = (slice_start_ms, slice_end_ms) {
            let start = (start_ms as usize).saturating_mul(16_000) / 1000;
            let end = (end_ms as usize)
                .saturating_mul(16_000)
                .div_ceil(1000)
                .min(samples.len());
            samples = samples[start.min(samples.len())..end].to_vec();
            println!(
                "slice_ms={start_ms}..{end_ms} samples={} duration_ms={:.3}",
                samples.len(),
                samples.len() as f64 * 1000.0 / 16_000.0
            );
        }
        Some(samples)
    } else {
        None
    };
    let mut totals = Vec::with_capacity(repeats);
    for run in 0..repeats {
        let start = Instant::now();
        let audio_input = if let Some(samples) = waveform_samples.as_ref() {
            AudioInput::Waveform {
                samples,
                sample_rate: 16_000,
            }
        } else {
            AudioInput::Path(audio_path)
        };
        let (out, timings) = model.transcribe_timed(vec![audio_input], opts.clone())?;
        let total = start.elapsed();
        let text = out.first().map(|o| o.text.as_str()).unwrap_or("");
        let decode_ms = timings.generation.decode_us as f64 / 1000.0;
        let decode_tokens_per_s = if timings.generation.decode_us == 0 {
            0.0
        } else {
            timings.generation.tokens_generated as f64
                / (timings.generation.decode_us as f64 / 1_000_000.0)
        };
        println!(
            "run={} wall_ms={} timed_total_ms={} audio_encoder_ms={} prefill_ms={} decode_ms={} decode_token_tensor_ms={:.3} decode_embed_ms={:.3} decode_position_ms={:.3} decode_metadata_ms={:.3} decode_graph_replay_ms={:.3} decode_forward_ms={:.3} decode_argmax_ms={:.3} prompt_len={} steps={} tokens={} decode_tokens_per_s={:.3} text={:?}",
            run + 1,
            total.as_millis(),
            timings.total_us / 1000,
            timings.audio_encoder_us / 1000,
            timings.generation.prefill_us / 1000,
            decode_ms.round() as u64,
            timings.generation.decode_token_tensor_us as f64 / 1000.0,
            timings.generation.decode_embed_us as f64 / 1000.0,
            timings.generation.decode_position_us as f64 / 1000.0,
            timings.generation.decode_metadata_us as f64 / 1000.0,
            timings.generation.decode_graph_replay_us as f64 / 1000.0,
            timings.generation.decode_forward_us as f64 / 1000.0,
            timings.generation.decode_argmax_us as f64 / 1000.0,
            timings.generation.prompt_len,
            timings.generation.steps,
            timings.generation.tokens_generated,
            decode_tokens_per_s,
            text
        );
        totals.push(total.as_secs_f64());
    }

    let avg = totals.iter().sum::<f64>() / totals.len().max(1) as f64;
    println!("avg_wall_ms={:.3}", avg * 1000.0);
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
        anyhow::bail!("bench_transcribe requires a GPU build; enable `cuda` or `metal`");
    }
}
