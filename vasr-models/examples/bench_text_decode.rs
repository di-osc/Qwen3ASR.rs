use std::time::Instant;

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use vasr_models::qwen3_asr::processor::chat_template;
use vasr_models::qwen3_asr::{GenerationTimings, LoadOptions, Qwen3Asr};
#[cfg(feature = "paged-attn")]
use vasr_paged_attn::{PagedCacheConfig, PagedCacheMemory};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1).peekable();
    let mut stop_at_eos = true;
    let mut warmup_runs = 0usize;
    let mut batch_size = 1usize;
    let model = args
        .next()
        .unwrap_or_else(|| "Qwen/Qwen3-ASR-0.6B".to_string());
    let prompt = args
        .next()
        .unwrap_or_else(|| "The quick answer to this question is".to_string());
    let repeats = args
        .next()
        .as_deref()
        .unwrap_or("5")
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
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--no-eos" => stop_at_eos = false,
            "--eos" => stop_at_eos = true,
            "--warmup" => {
                let Some(v) = args.next() else {
                    anyhow::bail!("--warmup requires a numeric argument");
                };
                warmup_runs = v
                    .parse::<usize>()
                    .context("failed to parse --warmup value")?;
            }
            "--batch" => {
                let Some(v) = args.next() else {
                    anyhow::bail!("--batch requires a numeric argument");
                };
                batch_size = v
                    .parse::<usize>()
                    .context("failed to parse --batch value")?;
                if batch_size == 0 {
                    anyhow::bail!("--batch must be >= 1");
                }
            }
            other => {
                return Err(anyhow::anyhow!(
                    "unknown argument {other:?}; expected --no-eos, --batch <N>, or --warmup <N>"
                ));
            }
        }
    }

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
            paged_cache: Some(default_paged_cache_config(&device)),
        },
    )?;
    let load_ms = start_load.elapsed().as_secs_f64() * 1000.0;
    println!("load_ms={load_ms:.3}");

    // Prepare batch prompts
    let prompts: Vec<String> = (0..batch_size).map(|_| prompt.clone()).collect();
    let input_ids_list: Vec<Vec<u32>> = prompts
        .iter()
        .map(|p| {
            model
                .processor()
                .tokenizer
                .encode(p)
                .map_err(|e| anyhow::anyhow!("encode error: {e}"))
        })
        .collect::<Result<Vec<_>>>()?;
    let attention_list: Vec<Vec<u32>> = input_ids_list
        .iter()
        .map(|ids| vec![1u32; ids.len()])
        .collect();

    let eos_token_ids = if stop_at_eos {
        vec![
            model
                .processor()
                .tokenizer
                .token_to_id(chat_template::IM_END)?,
            model.processor().tokenizer.token_to_id("<|endoftext|>")?,
        ]
    } else {
        vec![u32::MAX]
    };
    let eos_token_ids = {
        let mut ids = eos_token_ids;
        ids.sort_unstable();
        ids.dedup();
        ids
    };

    for _ in 0..warmup_runs {
        let input_rows: Vec<&[u32]> = input_ids_list.iter().map(|v| v.as_slice()).collect();
        let attn_rows: Vec<&[u32]> = attention_list.iter().map(|v| v.as_slice()).collect();
        #[cfg(feature = "paged-attn")]
        {
            vasr_models::qwen3_asr::model::generation::greedy_generate_cached_batch_timed_with_paged_runtime(
                &model.inner_model().thinker,
                model.device(),
                input_rows.as_slice(),
                attn_rows.as_slice(),
                None,
                1,
                eos_token_ids.as_slice(),
                None, // Metal defaults to eager, paged is opt-in
            )?;
        }
        #[cfg(not(feature = "paged-attn"))]
        {
            vasr_models::qwen3_asr::model::generation::greedy_generate_cached_batch_timed(
                &model.inner_model().thinker,
                model.device(),
                input_rows.as_slice(),
                attn_rows.as_slice(),
                None,
                1,
                eos_token_ids.as_slice(),
            )?;
        }
    }

    let mut totals = Vec::with_capacity(repeats);
    let mut total_timing = GenerationTimings::default();
    for run in 0..repeats {
        let start = Instant::now();
        let input_rows: Vec<&[u32]> = input_ids_list.iter().map(|v| v.as_slice()).collect();
        let attn_rows: Vec<&[u32]> = attention_list.iter().map(|v| v.as_slice()).collect();
        #[cfg(feature = "paged-attn")]
        let (gen_seqs, timings) = vasr_models::qwen3_asr::model::generation::greedy_generate_cached_batch_timed_with_paged_runtime(
            &model.inner_model().thinker,
            model.device(),
            input_rows.as_slice(),
            attn_rows.as_slice(),
            None,
            max_new_tokens,
            eos_token_ids.as_slice(),
            None, // Metal defaults to eager, paged is opt-in
        )?;
        #[cfg(not(feature = "paged-attn"))]
        let (gen_seqs, timings) =
            vasr_models::qwen3_asr::model::generation::greedy_generate_cached_batch_timed(
                &model.inner_model().thinker,
                model.device(),
                input_rows.as_slice(),
                attn_rows.as_slice(),
                None,
                max_new_tokens,
                eos_token_ids.as_slice(),
            )?;
        let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
        let decode_us = timings.decode_us;
        let decode_ms = decode_us as f64 / 1000.0;
        let steps = timings.steps;
        let tokens = timings.tokens_generated;
        let batch_tokens_per_s = if decode_us > 0 {
            (tokens as f64) * 1_000_000.0 / (decode_us as f64)
        } else if wall_ms > 0.0 {
            tokens as f64 / (wall_ms / 1000.0)
        } else {
            0.0
        };
        let per_seq_tokens_per_s = batch_tokens_per_s / batch_size as f64;
        let text = gen_seqs
            .first()
            .and_then(|ids| model.processor().tokenizer.decode(ids.as_slice()).ok())
            .unwrap_or_default();
        println!(
            "run={} wall_ms={:.3} decode_ms={:.3} decode_steps={} batch={} tokens={} batch_tok_per_s={:.3} per_seq_tok_per_s={:.3} text={:?}",
            run + 1,
            wall_ms,
            decode_ms,
            steps,
            batch_size,
            tokens,
            batch_tokens_per_s,
            per_seq_tokens_per_s,
            text
        );
        totals.push(decode_ms.max(0.000_1));
        total_timing.decode_us = total_timing.decode_us.saturating_add(timings.decode_us);
        total_timing.decode_forward_us = total_timing
            .decode_forward_us
            .saturating_add(timings.decode_forward_us);
        total_timing.decode_argmax_us = total_timing
            .decode_argmax_us
            .saturating_add(timings.decode_argmax_us);
        total_timing.decode_token_tensor_us = total_timing
            .decode_token_tensor_us
            .saturating_add(timings.decode_token_tensor_us);
        total_timing.decode_embed_us = total_timing
            .decode_embed_us
            .saturating_add(timings.decode_embed_us);
        total_timing.decode_position_us = total_timing
            .decode_position_us
            .saturating_add(timings.decode_position_us);
        total_timing.decode_metadata_us = total_timing
            .decode_metadata_us
            .saturating_add(timings.decode_metadata_us);
        total_timing.decode_pre_argmax_sync_us = total_timing
            .decode_pre_argmax_sync_us
            .saturating_add(timings.decode_pre_argmax_sync_us);
        total_timing.tokens_generated = total_timing
            .tokens_generated
            .saturating_add(timings.tokens_generated);
        total_timing.steps = total_timing.steps.saturating_add(timings.steps);
        total_timing.prefill_us = total_timing.prefill_us.saturating_add(timings.prefill_us);
        total_timing.prompt_len = timings.prompt_len;
    }

    let avg_ms = totals.iter().sum::<f64>() / totals.len().max(1) as f64;
    let wall_avg_ms = avg_ms;
    let total_tokens = total_timing.tokens_generated;
    let agg_tps = if total_timing.decode_us > 0 {
        total_tokens as f64 * 1_000_000.0 / total_timing.decode_us as f64
    } else if wall_avg_ms > 0.0 {
        total_tokens as f64 / (wall_avg_ms / 1000.0)
    } else {
        0.0
    };
    println!("avg_decode_ms={:.3}", avg_ms);
    println!("stop_at_eos={stop_at_eos}");
    println!("decode_totals_us={}", total_timing.decode_us);
    println!(
        "decode_breakdown_ms={}::token_tensor={:.3} embed={:.3} position={:.3} metadata={:.3} forward={:.3} argmax={:.3} pre_argmax_sync={:.3}",
        total_timing.decode_us as f64 / 1000.0,
        total_timing.decode_token_tensor_us as f64 / 1000.0,
        total_timing.decode_embed_us as f64 / 1000.0,
        total_timing.decode_position_us as f64 / 1000.0,
        total_timing.decode_metadata_us as f64 / 1000.0,
        total_timing.decode_forward_us as f64 / 1000.0,
        total_timing.decode_argmax_us as f64 / 1000.0,
        total_timing.decode_pre_argmax_sync_us as f64 / 1000.0
    );
    println!("aggregate_decode_tokens_per_s={:.3}", agg_tps);

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
fn default_paged_cache_config(device: &Device) -> PagedCacheConfig {
    PagedCacheConfig {
        block_size: 32,
        memory: if device.is_cuda() {
            PagedCacheMemory::GpuMemoryFraction(0.8)
        } else {
            PagedCacheMemory::ContextSize(100_000)
        },
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
        anyhow::bail!("bench_text_decode requires a GPU build; enable `cuda` or `metal`");
    }
}
