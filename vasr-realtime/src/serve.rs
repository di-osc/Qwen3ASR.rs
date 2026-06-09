use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Result, bail};
use axum::{Json, Router, routing::get};
#[cfg(any(feature = "metal", feature = "cuda"))]
use candle_core::Device;
use clap::Args;
use vasr_models::qwen3_asr::LoadOptions;
use vasr_models::qwen3_asr::model::isq_linear::resolve_isq_display;
#[cfg(any(feature = "metal", feature = "cuda"))]
use vasr_models::qwen3_asr::model::paged_cache_runtime::{
    PagedCacheConfig, PagedCacheMemory, PagedCacheStats,
};
use vasr_runtime::{
    AsrModel, AsrOptions, FsmnVadModel, InferenceScheduler, Qwen3AsrModel, RealtimePipeline,
    ScheduledAsrModel, VadModel, VadOptions, auto_dtype, device_label, resolve_device,
};

use crate::server::{RealtimeService, RealtimeSession};

#[derive(Debug, Clone, Args)]
pub struct CommonModelArgs {
    /// Qwen3-ASR model: `Qwen/Qwen3-ASR-0.6B` or `Qwen/Qwen3-ASR-1.7B`, or a local directory.
    #[arg(long, default_value = "Qwen/Qwen3-ASR-0.6B", env = "VASR_MODEL")]
    pub model: String,

    /// Device: auto, cpu, metal, cuda.
    #[arg(long, default_value = "auto", env = "VASR_DEVICE")]
    pub device: String,

    /// Enable in-situ quantization, for example q8_0.
    #[arg(long, env = "VASR_ISQ")]
    pub isq: Option<String>,

    /// Maximum decode tokens per request.
    #[arg(long, default_value_t = 512, env = "VASR_MAX_NEW_TOKENS")]
    pub max_new_tokens: usize,

    /// PagedAttention KV pool as a fraction of total GPU memory (CUDA). Used when `pa_context_len` is 0.
    #[arg(
        long = "pa-gpu-memory-fraction",
        default_value_t = 0.65,
        env = "VASR_PA_GPU_MEMORY_FRACTION"
    )]
    pub pa_gpu_memory_fraction: f32,

    /// Explicit KV context tokens. Set 0 to auto-size from `--pa-gpu-memory-fraction`.
    #[arg(
        long = "pa-context-len",
        default_value_t = 0,
        env = "VASR_PA_CONTEXT_LEN"
    )]
    pub pa_context_len: usize,

    /// PagedAttention block size. Supported values: 8, 16, 32.
    #[arg(
        long = "pa-block-size",
        default_value_t = 32,
        env = "VASR_PA_BLOCK_SIZE"
    )]
    pub pa_block_size: usize,
}

#[derive(Debug, Clone, Args)]
pub struct VadCliArgs {
    /// FunASR FSMN VAD model directory. If omitted, vASR downloads/uses `funasr/fsmn-vad`.
    #[arg(long, env = "VASR_VAD_MODEL")]
    pub vad_model: Option<String>,

    /// FSMN VAD speech probability threshold (`speech_noise_thres`).
    #[arg(long, default_value_t = 0.5, env = "VASR_VAD_THRESHOLD")]
    pub vad_threshold: f32,

    /// Minimum speech duration in milliseconds before a segment is confirmed.
    #[arg(long, default_value_t = 250, env = "VASR_VAD_MIN_SPEECH_MS")]
    pub vad_min_speech_ms: u64,

    /// Minimum trailing silence in milliseconds before ending a speech segment.
    #[arg(long, default_value_t = 500, env = "VASR_VAD_MIN_SILENCE_MS")]
    pub vad_min_silence_ms: u64,

    /// Merge adjacent offline VAD segments across gaps up to this many milliseconds.
    #[arg(long, default_value_t = 2_000, env = "VASR_VAD_MERGE_MAX_GAP_MS")]
    pub vad_merge_max_gap_ms: u64,

    /// Maximum merged offline ASR slice duration in milliseconds.
    #[arg(long, default_value_t = 30_000, env = "VASR_VAD_MERGE_MAX_SEGMENT_MS")]
    pub vad_merge_max_segment_ms: u64,
}

#[derive(Debug, Clone, Args)]
pub struct RealtimeArgs {
    #[command(flatten)]
    pub model: CommonModelArgs,

    /// Bind host.
    #[arg(long, default_value = "127.0.0.1", env = "VASR_HOST")]
    pub host: String,

    /// Bind port.
    #[arg(long, default_value_t = 8000, env = "VASR_PORT")]
    pub port: u16,

    #[command(flatten)]
    pub vad: VadCliArgs,
}

pub fn init_logging(verbose: bool) {
    let default = if verbose {
        "warn,vasr_realtime=info,vasr_runtime=info"
    } else {
        "warn,vasr_realtime=info"
    };
    let filter = std::env::var("VASR_LOG").unwrap_or_else(|_| default.to_string());

    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new(filter))
        .with_timer(tracing_subscriber::fmt::time::UtcTime::rfc_3339())
        .with_target(true)
        .compact()
        .init();
}

fn vad_options_from_args(vad: &VadCliArgs) -> VadOptions {
    VadOptions {
        threshold: vad.vad_threshold,
        min_speech_ms: vad.vad_min_speech_ms,
        min_silence_ms: vad.vad_min_silence_ms,
        merge_max_gap_ms: vad.vad_merge_max_gap_ms,
        merge_max_segment_ms: vad.vad_merge_max_segment_ms,
    }
}

pub async fn run_realtime(args: RealtimeArgs) -> Result<()> {
    validate_model_args(&args.model)?;
    validate_vad_args(&args.vad)?;
    let asr = load_asr_model(&args.model, Some(1))?;

    let realtime_asr = Arc::clone(&asr);
    let max_new_tokens = args.model.max_new_tokens;
    let vad_model = args.vad.vad_model.clone();
    let vad_options = vad_options_from_args(&args.vad);
    let realtime_service = Arc::new(RealtimeService {
        make_session: Arc::new(move || {
            let vad = load_fsmn_vad(&vad_model)?.start_stream(&vad_options)?;
            let asr_stream = realtime_asr.start_stream(&AsrOptions {
                max_new_tokens,
                ..AsrOptions::default()
            })?;
            Ok(RealtimeSession::new(
                16_000,
                RealtimePipeline {
                    vad,
                    asr: asr_stream,
                },
            ))
        }),
    });

    let app = Router::new()
        .route("/health", get(health))
        .merge(crate::server::realtime_router(realtime_service));
    let addr = bind_addr(&args.host, args.port)?;
    serve_app(app, addr).await
}

fn validate_model_args(args: &CommonModelArgs) -> Result<()> {
    if args.model.trim().is_empty() {
        bail!("--model must not be empty");
    }
    if args.max_new_tokens == 0 {
        bail!("--max-new-tokens must be greater than zero");
    }
    if !(args.pa_gpu_memory_fraction > 0.0 && args.pa_gpu_memory_fraction <= 1.0) {
        bail!("--pa-gpu-memory-fraction must be in (0.0, 1.0]");
    }
    Ok(())
}

#[cfg(any(feature = "metal", feature = "cuda"))]
fn build_paged_cache_config(
    device: &Device,
    pa_context_len: usize,
    pa_gpu_memory_fraction: f32,
    pa_block_size: usize,
) -> Result<PagedCacheConfig> {
    let memory = if pa_context_len > 0 {
        PagedCacheMemory::ContextSize(pa_context_len)
    } else if device.is_cuda() {
        PagedCacheMemory::GpuMemoryFraction(pa_gpu_memory_fraction)
    } else {
        PagedCacheMemory::ContextSize(4096)
    };
    Ok(PagedCacheConfig {
        block_size: pa_block_size,
        memory,
    })
}

#[cfg(any(feature = "metal", feature = "cuda"))]
fn log_paged_cache_stats(stats: &PagedCacheStats, config: &PagedCacheConfig) {
    let sizing = match config.memory {
        PagedCacheMemory::ContextSize(tokens) => format!("context_tokens={tokens}"),
        PagedCacheMemory::Blocks(blocks) => format!("blocks={blocks}"),
        PagedCacheMemory::GpuMemoryFraction(fraction) => {
            format!("gpu_memory_fraction={fraction}")
        }
    };
    tracing::info!(
        target: "vasr_realtime::serve",
        "PagedAttention KV cache: {sizing}, block_size={}, blocks={}, free_blocks={}, max_context_tokens={}, bytes={:.2} MiB.",
        stats.block_size,
        stats.num_blocks,
        stats.free_blocks,
        stats.max_context_tokens,
        stats.bytes as f64 / (1024.0 * 1024.0)
    );
}

fn validate_vad_args(vad: &VadCliArgs) -> Result<()> {
    if !(0.0..=1.0).contains(&vad.vad_threshold) {
        bail!("--vad-threshold must be between 0.0 and 1.0");
    }
    if vad.vad_min_speech_ms == 0 {
        bail!("--vad-min-speech-ms must be greater than zero");
    }
    if vad.vad_min_silence_ms == 0 {
        bail!("--vad-min-silence-ms must be greater than zero");
    }
    if vad.vad_merge_max_gap_ms == 0 {
        bail!("--vad-merge-max-gap-ms must be greater than zero");
    }
    if vad.vad_merge_max_segment_ms == 0 {
        bail!("--vad-merge-max-segment-ms must be greater than zero");
    }
    Ok(())
}

fn load_asr_model(
    args: &CommonModelArgs,
    #[cfg_attr(not(feature = "cuda-graph"), allow(unused_variables))] cuda_graph_max_batch: Option<
        usize,
    >,
) -> Result<Arc<dyn AsrModel>> {
    let device = resolve_device(&args.device)?;
    let dtype = auto_dtype(&device)?;
    #[cfg(any(feature = "metal", feature = "cuda"))]
    let paged_cache_config = build_paged_cache_config(
        &device,
        args.pa_context_len,
        args.pa_gpu_memory_fraction,
        args.pa_block_size,
    )?;
    let load_options = LoadOptions {
        dtype,
        use_flash_attn: device.is_cuda(),
        isq: args.isq.clone(),
        #[cfg(any(feature = "metal", feature = "cuda"))]
        paged_cache: Some(paged_cache_config),
    };

    tracing::info!(
        target: "vasr_realtime::serve",
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        cfg!(target_feature = "avx"),
        cfg!(target_feature = "neon"),
        cfg!(target_feature = "simd128"),
        cfg!(target_feature = "f16c")
    );
    tracing::info!(
        target: "vasr_realtime::serve",
        "Model kind is: qwen3-asr (no adapters)"
    );
    tracing::info!(
        target: "vasr_realtime::serve",
        "Auto-selected DType {:?} for {}.",
        dtype,
        device_label(&device)
    );
    if let Some(isq) = args.isq.as_deref() {
        tracing::info!(
            target: "vasr_realtime::serve",
            "ISQ selected is {} (requested={}, backend={}).",
            resolve_isq_display(isq, &device)?,
            isq,
            device_label(&device)
        );
    }
    tracing::info!(
        target: "vasr_realtime::serve",
        "Loading Qwen3-ASR model `{}` on {} (isq={:?}).",
        args.model,
        device_label(&device),
        args.isq
    );
    let model_load_start = Instant::now();
    let qwen3_asr = Qwen3AsrModel::from_pretrained(&args.model, &device, &load_options)?;
    log_model_config(&qwen3_asr);
    #[cfg(any(feature = "metal", feature = "cuda"))]
    if let Some(stats) = qwen3_asr.inner().paged_cache_stats() {
        log_paged_cache_stats(&stats, &paged_cache_config);
    }
    #[cfg(feature = "cuda-graph")]
    if let Some(max_batch) = cuda_graph_max_batch.filter(|&n| n > 0) {
        let prewarm_start = Instant::now();
        let captured = qwen3_asr.inner().prewarm_cuda_decode_graphs(max_batch)?;
        tracing::info!(
            target: "vasr_realtime::serve",
            "CUDA decode graph prewarm: graphs={captured} max_batch={max_batch} elapsed={:.3}s.",
            prewarm_start.elapsed().as_secs_f64()
        );
    }
    tracing::info!(
        target: "vasr_realtime::serve",
        "Model loaded in {:.3}s.",
        model_load_start.elapsed().as_secs_f64()
    );
    let asr_scheduler = InferenceScheduler::start("asr");
    let base_asr: Arc<dyn AsrModel> = Arc::new(qwen3_asr);
    let asr: Arc<dyn AsrModel> = Arc::new(ScheduledAsrModel::new(
        Arc::clone(&base_asr),
        asr_scheduler.clone(),
    ));
    Ok(asr)
}

fn bind_addr(host: &str, port: u16) -> Result<SocketAddr> {
    Ok(format!("{host}:{port}").parse()?)
}

async fn serve_app(app: Router, addr: SocketAddr) -> Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(
        target: "vasr_realtime::serve",
        "WebSocket endpoints: /v1/realtime, /api-ws/v1/realtime"
    );
    tracing::info!(
        target: "vasr_realtime::serve",
        "vASR realtime service listening on http://{addr}"
    );
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

fn load_fsmn_vad(path: &Option<String>) -> Result<FsmnVadModel> {
    FsmnVadModel::from_pretrained(path.as_deref())
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({"status": "ok"}))
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
}

fn log_model_config(model: &Qwen3AsrModel) {
    let config = model.inner().config();
    let text = &config.thinker_config.text_config;
    let audio = &config.thinker_config.audio_config;
    tracing::info!(
        target: "vasr_realtime::serve",
        "Model config: model_type={:?}, thinker_type={:?}, text_layers={}, hidden_size={}, kv_heads={}, audio_layers={}, audio_dim={}",
        config.model_type,
        config.thinker_config.model_type,
        text.num_hidden_layers,
        text.hidden_size,
        text.num_key_value_heads,
        audio.encoder_layers,
        audio.d_model
    );
}
