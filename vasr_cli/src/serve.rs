use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Result, bail};
use axum::{Json, Router, routing::get};
use candle_core::{DType, Device};
use clap::Args;
use vasr_audio::AudioLoader;
use vasr_models::qwen3_asr::LoadOptions;
use vasr_models::qwen3_asr::model::isq_linear::resolve_isq_display;
use vasr_runtime::{
    AsrModel, AsrOptions, OfflinePipeline, Qwen3AsrModel, RealtimePipeline, SileroVadModel,
    VadModel, VadOptions,
};
use vasr_server::{RealtimeService, RealtimeSession, TranscribeService};

#[derive(Debug, Clone, Args)]
pub struct ServeArgs {
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

    /// Disable VAD annotations in offline transcribe.
    #[arg(long, default_value_t = false)]
    pub no_vad: bool,

    /// Silero VAD ONNX model path. If omitted, vASR searches local Silero caches.
    #[arg(long, env = "VASR_VAD_MODEL")]
    pub vad_model: Option<String>,

    /// Maximum decode tokens per request.
    #[arg(long, default_value_t = 256, env = "VASR_MAX_NEW_TOKENS")]
    pub max_new_tokens: usize,
}

pub async fn run(args: ServeArgs) -> Result<()> {
    if args.model.trim().is_empty() {
        bail!("--model must not be empty");
    }
    if args.max_new_tokens == 0 {
        bail!("--max-new-tokens must be greater than zero");
    }

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
    let asr: Arc<dyn AsrModel> = Arc::new(qwen3_asr);

    let offline_vad = if args.no_vad {
        tracing::info!(
            target: "vasr_cli::serve",
            "Offline Silero VAD annotations disabled."
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
        asr: Arc::clone(&asr),
    };
    let transcribe_service = Arc::new(TranscribeService {
        pipeline: Arc::new(offline),
        loader: AudioLoader,
        options: AsrOptions {
            max_new_tokens: args.max_new_tokens,
            ..AsrOptions::default()
        },
    });

    let realtime_asr = Arc::clone(&asr);
    let max_new_tokens = args.max_new_tokens;
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
        .merge(vasr_server::transcribe_router(transcribe_service))
        .merge(vasr_server::realtime_router(realtime_service));
    let addr: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(
        target: "vasr_cli::serve",
        "HTTP endpoints: GET /health, POST /transcribe, POST /inference"
    );
    tracing::info!(
        target: "vasr_cli::serve",
        "WebSocket endpoints: /v1/realtime, /api-ws/v1/realtime"
    );
    tracing::info!(
        target: "vasr_cli::serve",
        "vASR service listening on http://{addr}"
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
