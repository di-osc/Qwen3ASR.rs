use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Result, bail};
use axum::{Json, Router, routing::get};
use candle_core::{DType, Device};
use clap::{Args, Subcommand};
use vasr_audio::AudioLoader;
use vasr_models::qwen3_asr::LoadOptions;
use vasr_models::qwen3_asr::model::isq_linear::resolve_isq_display;
use vasr_runtime::{
    AsrModel, AsrOptions, OfflinePipeline, Qwen3AsrModel, RealtimePipeline, SileroVadModel,
    VadModel, VadOptions,
};
use vasr_server::{
    InferenceScheduler, RealtimeService, RealtimeSession, ScheduledAsrModel, TranscribeService,
};

#[derive(Debug, Clone, Args)]
pub struct ServeArgs {
    #[command(subcommand)]
    pub service: ServeService,
}

#[derive(Debug, Clone, Subcommand)]
pub enum ServeService {
    /// Start only the offline transcribe HTTP service.
    Transcribe(TranscribeServeArgs),
    /// Start only the realtime WebSocket service.
    Realtime(RealtimeServeArgs),
}

#[derive(Debug, Clone, Args)]
pub struct CommonServeArgs {
    /// Qwen3-ASR model id or local model directory.
    #[arg(long, env = "VASR_MODEL")]
    pub model: String,

    /// Bind host.
    #[arg(long, default_value = "127.0.0.1", env = "VASR_HOST")]
    pub host: String,

    /// Bind port.
    #[arg(long, default_value_t = 8000, env = "VASR_PORT")]
    pub port: u16,

    /// Device: auto, cpu, metal, cuda.
    #[arg(long, default_value = "auto", env = "VASR_DEVICE")]
    pub device: String,

    /// Weight dtype: auto, f32, f16, bf16.
    #[arg(long, default_value = "bf16", env = "VASR_DTYPE")]
    pub dtype: String,

    /// Enable Qwen3-ASR flash attention where supported.
    #[arg(long, default_value_t = false, env = "VASR_FLASH_ATTN")]
    pub flash_attn: bool,

    /// Enable in-situ quantization, for example q8_0.
    #[arg(long, env = "VASR_ISQ")]
    pub isq: Option<String>,

    /// Maximum decode tokens per request.
    #[arg(long, default_value_t = 256, env = "VASR_MAX_NEW_TOKENS")]
    pub max_new_tokens: usize,
}

#[derive(Debug, Clone, Args)]
pub struct TranscribeServeArgs {
    #[command(flatten)]
    pub common: CommonServeArgs,

    /// Disable VAD segmentation in offline transcribe.
    #[arg(long, default_value_t = false)]
    pub no_vad: bool,

    /// Silero VAD ONNX model path. If omitted, vASR searches local Silero caches.
    #[arg(long, env = "VASR_VAD_MODEL")]
    pub vad_model: Option<String>,
}

#[derive(Debug, Clone, Args)]
pub struct RealtimeServeArgs {
    #[command(flatten)]
    pub common: CommonServeArgs,

    /// Silero VAD ONNX model path. If omitted, vASR searches local Silero caches.
    #[arg(long, env = "VASR_VAD_MODEL")]
    pub vad_model: Option<String>,
}

pub async fn run(args: ServeArgs) -> Result<()> {
    match args.service {
        ServeService::Transcribe(args) => run_transcribe(args).await,
        ServeService::Realtime(args) => run_realtime(args).await,
    }
}

async fn run_transcribe(args: TranscribeServeArgs) -> Result<()> {
    validate_common(&args.common)?;
    let asr = load_asr_model(&args.common)?;

    let offline_vad = if args.no_vad {
        tracing::info!(
            target: "vasr_cli::serve",
            "Offline Silero VAD segmentation disabled."
        );
        None
    } else {
        let vad_load_start = Instant::now();
        let vad = load_silero(&args.vad_model)?;
        tracing::info!(
            target: "vasr_cli::serve",
            "Silero VAD loaded from `{}` in {:.3}s.",
            vad.path().display(),
            vad_load_start.elapsed().as_secs_f64()
        );
        Some(Box::new(vad) as Box<dyn VadModel>)
    };
    let offline = OfflinePipeline {
        vad: offline_vad,
        asr,
    };
    let transcribe_service = Arc::new(TranscribeService {
        pipeline: Arc::new(offline),
        loader: AudioLoader,
        options: AsrOptions {
            max_new_tokens: args.common.max_new_tokens,
            ..AsrOptions::default()
        },
    });

    let app = Router::new()
        .route("/health", get(health))
        .merge(vasr_server::transcribe_router(transcribe_service));
    let addr = bind_addr(&args.common)?;
    serve_app(app, addr, "transcribe").await
}

async fn run_realtime(args: RealtimeServeArgs) -> Result<()> {
    validate_common(&args.common)?;
    let asr = load_asr_model(&args.common)?;

    let realtime_asr = Arc::clone(&asr);
    let max_new_tokens = args.common.max_new_tokens;
    let vad_model = args.vad_model.clone();
    let realtime_service = Arc::new(RealtimeService {
        make_session: Arc::new(move || {
            let vad = load_silero(&vad_model)?.start_stream(&VadOptions::default())?;
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
        .merge(vasr_server::realtime_router(realtime_service));
    let addr = bind_addr(&args.common)?;
    serve_app(app, addr, "realtime").await
}

fn validate_common(args: &CommonServeArgs) -> Result<()> {
    if args.model.trim().is_empty() {
        bail!("--model must not be empty");
    }
    if args.max_new_tokens == 0 {
        bail!("--max-new-tokens must be greater than zero");
    }
    Ok(())
}

fn load_asr_model(args: &CommonServeArgs) -> Result<Arc<dyn AsrModel>> {
    let device = resolve_device(&args.device)?;
    let dtype = resolve_dtype(&args.dtype, &device)?;
    let load_options = LoadOptions {
        dtype,
        use_flash_attn: args.flash_attn,
        isq: args.isq.clone(),
    };

    tracing::info!(
        target: "vasr_cli::serve",
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        cfg!(target_feature = "avx"),
        cfg!(target_feature = "neon"),
        cfg!(target_feature = "simd128"),
        cfg!(target_feature = "f16c")
    );
    tracing::info!(
        target: "vasr_cli::serve",
        "Model kind is: qwen3-asr (no adapters)"
    );
    tracing::info!(
        target: "vasr_cli::serve",
        "DType selected is {:?}.",
        dtype
    );
    if let Some(isq) = args.isq.as_deref() {
        tracing::info!(
            target: "vasr_cli::serve",
            "ISQ selected is {} (requested={}, backend={}).",
            resolve_isq_display(isq, &device)?,
            isq,
            device_label(&device)
        );
    }
    tracing::info!(
        target: "vasr_cli::serve",
        "Loading Qwen3-ASR model `{}` on {} (flash_attn={}, isq={:?}).",
        args.model,
        device_label(&device),
        args.flash_attn,
        args.isq
    );
    let model_load_start = Instant::now();
    let qwen3_asr = Qwen3AsrModel::from_pretrained(&args.model, &device, &load_options)?;
    log_model_config(&qwen3_asr);
    tracing::info!(
        target: "vasr_cli::serve",
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

fn bind_addr(args: &CommonServeArgs) -> Result<SocketAddr> {
    Ok(format!("{}:{}", args.host, args.port).parse()?)
}

async fn serve_app(app: Router, addr: SocketAddr, service: &'static str) -> Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    match service {
        "transcribe" => tracing::info!(
            target: "vasr_cli::serve",
            "HTTP endpoints: GET /health, POST /transcribe, POST /inference"
        ),
        "realtime" => tracing::info!(
            target: "vasr_cli::serve",
            "WebSocket endpoints: /v1/realtime, /api-ws/v1/realtime"
        ),
        _ => {}
    }
    tracing::info!(
        target: "vasr_cli::serve",
        "vASR {service} service listening on http://{addr}"
    );
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

fn load_silero(path: &Option<String>) -> Result<SileroVadModel> {
    match path {
        Some(path) => SileroVadModel::from_onnx(path),
        None => SileroVadModel::from_default_model(),
    }
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({"status": "ok"}))
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
}

fn resolve_device(value: &str) -> Result<Device> {
    match value.trim().to_ascii_lowercase().as_str() {
        "auto" => auto_device(),
        "cpu" => Ok(Device::Cpu),
        "metal" => {
            #[cfg(feature = "metal")]
            {
                Device::new_metal(0)
                    .map_err(|err| anyhow::anyhow!("failed to create Metal device 0: {err}"))
            }
            #[cfg(not(feature = "metal"))]
            {
                bail!("metal requested but vasr was built without the metal feature")
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
                bail!("cuda requested but vasr was built without the cuda feature")
            }
        }
        other => bail!("unknown device {other:?}; expected auto, cpu, metal, or cuda"),
    }
}

fn auto_device() -> Result<Device> {
    #[cfg(feature = "cuda")]
    {
        return Device::new_cuda(0)
            .map_err(|err| anyhow::anyhow!("failed to create CUDA device 0: {err}"));
    }
    #[cfg(all(not(feature = "cuda"), feature = "metal"))]
    {
        return Device::new_metal(0)
            .map_err(|err| anyhow::anyhow!("failed to create Metal device 0: {err}"));
    }
    #[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
    {
        Ok(Device::Cpu)
    }
}

fn resolve_dtype(value: &str, device: &Device) -> Result<DType> {
    let dtype = match value.trim().to_ascii_lowercase().as_str() {
        "auto" if device.is_cpu() => DType::F32,
        "auto" => DType::F16,
        "f32" => DType::F32,
        "f16" => DType::F16,
        "bf16" => DType::BF16,
        other => bail!("unknown dtype {other:?}; expected auto, f32, f16, or bf16"),
    };
    Ok(dtype)
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

fn log_model_config(model: &Qwen3AsrModel) {
    let config = model.inner().config();
    let text = &config.thinker_config.text_config;
    let audio = &config.thinker_config.audio_config;
    tracing::info!(
        target: "vasr_cli::serve",
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
