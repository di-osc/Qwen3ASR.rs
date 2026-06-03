use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};
use candle_core::{DType, Device};
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
    let isq = args.next();

    let device = default_gpu_device()?;
    let use_flash_attn = cfg!(feature = "flash-attn") && device.is_cuda();
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
    println!("load_ms={:.3}", start_load.elapsed().as_secs_f64() * 1000.0);

    let opts = TranscribeOptions {
        context: Batch::one(String::new()),
        language: Batch::one(Some("English".to_string())),
        return_timestamps: false,
        max_new_tokens,
        max_batch_size: 1,
        chunk_max_sec: None,
        bucket_by_length: false,
    };

    let audio_path = Path::new(&audio);
    for run in 0..repeats {
        let start = Instant::now();
        let out = model.transcribe(vec![AudioInput::Path(audio_path)], opts.clone())?;
        let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
        let text = out.first().map(|o| o.text.as_str()).unwrap_or("");
        let estimated_tokens = model.processor().tokenizer.encode(text)?.len();
        println!(
            "run={} wall_ms={:.3} estimated_text_tokens={} estimated_wall_tokens_per_s={:.3} text={:?}",
            run + 1,
            wall_ms,
            estimated_tokens,
            estimated_tokens as f64 / (wall_ms / 1000.0),
            text
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
        anyhow::bail!("bench_transcribe_wall requires a GPU build; enable `cuda` or `metal`");
    }
}
