//! X-infer style LinearX/QLinear layers with in-situ quantization.

use std::cell::Cell;
use std::sync::Arc;
#[cfg(feature = "timing")]
use std::sync::atomic::AtomicU64;
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(feature = "metal")]
use candle_core::quantized::k_quants;
use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder};
#[cfg(feature = "paged-attn")]
use mistralrs_quant::{
    AfqBits, AfqGroupSize, AfqLayer, GluActivationType, QuantMethod, QuantMethodConfig,
};

#[cfg(feature = "timing")]
static ISQ_QUANTIZE_US: AtomicU64 = AtomicU64::new(0);
static AFQ_FALLBACK_WARNED: AtomicBool = AtomicBool::new(false);

thread_local! {
    static LINEAR_IS_PREFILL: Cell<bool> = const { Cell::new(false) };
}

pub struct LinearPrefillGuard {
    prev: bool,
}

impl Drop for LinearPrefillGuard {
    fn drop(&mut self) {
        LINEAR_IS_PREFILL.with(|flag| flag.set(self.prev));
    }
}

pub fn set_linear_is_prefill(is_prefill: bool) -> LinearPrefillGuard {
    let prev = LINEAR_IS_PREFILL.with(|flag| {
        let prev = flag.get();
        flag.set(is_prefill);
        prev
    });
    LinearPrefillGuard { prev }
}

pub fn linear_is_prefill() -> bool {
    LINEAR_IS_PREFILL.with(|flag| flag.get())
}

#[cfg(feature = "timing")]
pub fn reset_isq_quantize_time() {
    ISQ_QUANTIZE_US.store(0, Ordering::Relaxed);
}

#[cfg(feature = "timing")]
pub fn isq_quantize_time_us() -> u64 {
    ISQ_QUANTIZE_US.load(Ordering::Relaxed)
}

#[derive(Clone)]
pub enum LinearX {
    Linear(Linear),
    QLinear(QLinear),
    #[cfg(feature = "paged-attn")]
    AfqLinear(AfqLinear),
}

pub type IsqLinear = LinearX;

/// Target bit width for automatic ISQ selection.
///
/// This mirrors mistral.rs: Metal prefers AFQ variants, while other backends
/// prefer GGML K-quants. vASR currently implements GGML ISQ only, so Metal AFQ
/// selections resolve to their GGML fallback until the AFQ backend lands.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IsqBits {
    Two,
    Three,
    Four,
    Five,
    Six,
    Eight,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResolvedIsq {
    Ggml(GgmlDType),
    #[cfg(feature = "paged-attn")]
    Afq(AfqBits),
    AfqFallback {
        requested: &'static str,
        fallback: GgmlDType,
    },
}

impl ResolvedIsq {
    pub fn dtype(self) -> GgmlDType {
        match self {
            Self::Ggml(dtype) => dtype,
            #[cfg(feature = "paged-attn")]
            Self::Afq(bits) => match bits {
                AfqBits::Two => GgmlDType::Q2K,
                AfqBits::Three => GgmlDType::Q3K,
                AfqBits::Four => GgmlDType::Q4K,
                AfqBits::Six => GgmlDType::Q6K,
                AfqBits::Eight => GgmlDType::Q8_0,
                AfqBits::Mxfp4 => GgmlDType::Q4_0,
            },
            Self::AfqFallback { fallback, .. } => fallback,
        }
    }

    pub fn display_name(self) -> String {
        match self {
            Self::Ggml(dtype) => ggml_dtype_name(dtype).to_string(),
            #[cfg(feature = "paged-attn")]
            Self::Afq(bits) => afq_bits_name(bits).to_string(),
            Self::AfqFallback {
                requested,
                fallback,
            } => {
                format!("{requested} -> {}", ggml_dtype_name(fallback))
            }
        }
    }
}

impl IsqBits {
    pub fn resolve(self, device: &Device) -> ResolvedIsq {
        match (self, device.is_metal()) {
            #[cfg(feature = "paged-attn")]
            (Self::Two, true) => ResolvedIsq::Afq(AfqBits::Two),
            #[cfg(feature = "paged-attn")]
            (Self::Three, true) => ResolvedIsq::Afq(AfqBits::Three),
            #[cfg(feature = "paged-attn")]
            (Self::Four, true) => ResolvedIsq::Afq(AfqBits::Four),
            #[cfg(feature = "paged-attn")]
            (Self::Six, true) => ResolvedIsq::Afq(AfqBits::Six),
            #[cfg(feature = "paged-attn")]
            (Self::Eight, true) => ResolvedIsq::Afq(AfqBits::Eight),
            #[cfg(not(feature = "paged-attn"))]
            (Self::Two, true) => ResolvedIsq::AfqFallback {
                requested: "afq2",
                fallback: GgmlDType::Q2K,
            },
            (Self::Two, false) => ResolvedIsq::Ggml(GgmlDType::Q2K),
            #[cfg(not(feature = "paged-attn"))]
            (Self::Three, true) => ResolvedIsq::AfqFallback {
                requested: "afq3",
                fallback: GgmlDType::Q3K,
            },
            (Self::Three, false) => ResolvedIsq::Ggml(GgmlDType::Q3K),
            #[cfg(not(feature = "paged-attn"))]
            (Self::Four, true) => ResolvedIsq::AfqFallback {
                requested: "afq4",
                fallback: GgmlDType::Q4K,
            },
            (Self::Four, false) => ResolvedIsq::Ggml(GgmlDType::Q4K),
            (Self::Five, _) => ResolvedIsq::Ggml(GgmlDType::Q5K),
            #[cfg(not(feature = "paged-attn"))]
            (Self::Six, true) => ResolvedIsq::AfqFallback {
                requested: "afq6",
                fallback: GgmlDType::Q6K,
            },
            (Self::Six, false) => ResolvedIsq::Ggml(GgmlDType::Q6K),
            #[cfg(not(feature = "paged-attn"))]
            (Self::Eight, true) => ResolvedIsq::AfqFallback {
                requested: "afq8",
                fallback: GgmlDType::Q8_0,
            },
            (Self::Eight, false) => ResolvedIsq::Ggml(GgmlDType::Q8_0),
        }
    }
}

impl TryFrom<&str> for IsqBits {
    type Error = ();

    fn try_from(value: &str) -> std::result::Result<Self, Self::Error> {
        match value {
            "2" | "auto2" | "auto_2" | "auto-2" => Ok(Self::Two),
            "3" | "auto3" | "auto_3" | "auto-3" => Ok(Self::Three),
            "4" | "auto4" | "auto_4" | "auto-4" => Ok(Self::Four),
            "5" | "auto5" | "auto_5" | "auto-5" => Ok(Self::Five),
            "6" | "auto6" | "auto_6" | "auto-6" => Ok(Self::Six),
            "8" | "auto" | "auto8" | "auto_8" | "auto-8" => Ok(Self::Eight),
            _ => Err(()),
        }
    }
}

#[derive(Clone)]
pub struct QLinear {
    matmul: Arc<QMatMul>,
    bias: Option<Tensor>,
    output_dtype: DType,
    dtype: GgmlDType,
}

#[cfg(feature = "paged-attn")]
#[derive(Clone)]
pub struct AfqLinear {
    inner: Arc<AfqLayer>,
    bias: Option<Tensor>,
    output_dtype: DType,
    bits: AfqBits,
}

impl std::fmt::Debug for LinearX {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Linear(linear) => f.debug_tuple("Linear").field(linear).finish(),
            Self::QLinear(linear) => f.debug_tuple("QLinear").field(linear).finish(),
            #[cfg(feature = "paged-attn")]
            Self::AfqLinear(linear) => f.debug_tuple("AfqLinear").field(linear).finish(),
        }
    }
}

impl std::fmt::Debug for QLinear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QLinear")
            .field("dtype", &self.dtype)
            .field("output_dtype", &self.output_dtype)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "paged-attn")]
impl std::fmt::Debug for AfqLinear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AfqLinear")
            .field("bits", &self.bits)
            .field("output_dtype", &self.output_dtype)
            .finish_non_exhaustive()
    }
}

impl LinearX {
    pub fn new(linear: Linear, isq: Option<&str>, target_device: &Device) -> Result<Self> {
        Self::new_with_module_path(linear, isq, target_device, "")
    }

    pub fn new_with_module_path(
        linear: Linear,
        isq: Option<&str>,
        target_device: &Device,
        module_path: &str,
    ) -> Result<Self> {
        let Some(isq) = isq else {
            let linear = linear_to_device(linear, target_device)?;
            return Ok(Self::Linear(linear));
        };

        let spec = parse_isq_spec(isq, target_device)?;
        if should_skip_module(module_path, &spec.modules_to_not_convert) {
            let linear = linear_to_device(linear, target_device)?;
            return Ok(Self::Linear(linear));
        }

        #[cfg(feature = "paged-attn")]
        if let Some(bits) = spec.afq {
            return Self::new_afq(linear, bits, target_device);
        }

        let requested = spec
            .dtype
            .ok_or_else(|| candle_core::Error::Msg("missing ISQ dtype".to_string()))?;
        let weight = linear.weight();
        let output_dtype = weight.dtype();
        let bias = linear.bias().cloned();
        let dtype = compatible_dtype(weight, requested);

        let Some(dtype) = dtype else {
            let linear = linear_to_device(linear, target_device)?;
            return Ok(Self::Linear(linear));
        };

        #[cfg(feature = "timing")]
        let start_quantize = std::time::Instant::now();
        let qtensor = QTensor::quantize(weight, dtype)?;
        let qtensor = qtensor_to_device(qtensor, target_device)?;
        let matmul = QMatMul::from_qtensor(qtensor)?;
        #[cfg(feature = "timing")]
        ISQ_QUANTIZE_US.fetch_add(
            start_quantize
                .elapsed()
                .as_micros()
                .min(u128::from(u64::MAX)) as u64,
            Ordering::Relaxed,
        );
        Ok(Self::QLinear(QLinear {
            matmul: Arc::new(matmul),
            bias,
            output_dtype,
            dtype,
        }))
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Linear(linear) => linear.forward(x),
            Self::QLinear(linear) => linear.forward(x),
            #[cfg(feature = "paged-attn")]
            Self::AfqLinear(linear) => linear.forward(x),
        }
    }
}

#[cfg(feature = "paged-attn")]
impl LinearX {
    fn new_afq(linear: Linear, bits: AfqBits, target_device: &Device) -> Result<Self> {
        #[cfg(feature = "timing")]
        let start_quantize = std::time::Instant::now();
        let weight = linear.weight().to_device(target_device)?;
        let bias = linear
            .bias()
            .map(|b| b.to_device(target_device))
            .transpose()?;
        let output_dtype = weight.dtype();
        let inner = AfqLayer::new(QuantMethodConfig::Afq {
            weight,
            bias: None,
            bits,
            group_size: AfqGroupSize::default(),
        })?;
        #[cfg(feature = "timing")]
        ISQ_QUANTIZE_US.fetch_add(
            start_quantize
                .elapsed()
                .as_micros()
                .min(u128::from(u64::MAX)) as u64,
            Ordering::Relaxed,
        );
        Ok(Self::AfqLinear(AfqLinear {
            inner: Arc::new(inner),
            bias,
            output_dtype,
            bits,
        }))
    }
}

impl candle_core::Module for LinearX {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        LinearX::forward(self, x)
    }
}

impl QLinear {
    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

#[cfg(feature = "paged-attn")]
impl AfqLinear {
    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

#[cfg(feature = "cuda")]
pub fn try_fused_q8_silu_gate_up(
    gate: &IsqLinear,
    up: &IsqLinear,
    x: &Tensor,
) -> Result<Option<Tensor>> {
    let (LinearX::QLinear(gate), LinearX::QLinear(up)) = (gate, up) else {
        return Ok(None);
    };
    if gate.bias.is_some() || up.bias.is_some() {
        return Ok(None);
    }
    let (QMatMul::QTensor(gate_q), QMatMul::QTensor(up_q)) =
        (gate.matmul.as_ref(), up.matmul.as_ref())
    else {
        return Ok(None);
    };
    if !crate::q8_mmvq::can_run_fused_glu(gate_q, up_q, x) {
        return Ok(None);
    }
    Ok(Some(crate::q8_mmvq::fused_glu_silu(gate_q, up_q, x)?))
}

#[cfg(feature = "metal-paged-attn")]
pub fn try_fused_afq_silu_gate_up(
    gate: &IsqLinear,
    up: &IsqLinear,
    x: &Tensor,
) -> Result<Option<Tensor>> {
    let (LinearX::AfqLinear(gate), LinearX::AfqLinear(up)) = (gate, up) else {
        return Ok(None);
    };
    if gate.bias.is_some() || up.bias.is_some() {
        return Ok(None);
    }
    mistralrs_quant::try_fused_gate_up_metal(
        x,
        gate.inner.as_ref(),
        up.inner.as_ref(),
        GluActivationType::Silu,
    )
}

impl candle_core::Module for QLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if let QMatMul::QTensor(qtensor) = self.matmul.as_ref() {
            if crate::q8_mmvq::can_run(qtensor, x) {
                let mut ys = crate::q8_mmvq::plain(qtensor, x)?;
                if let Some(bias) = &self.bias {
                    ys = ys.broadcast_add(&bias.to_dtype(ys.dtype())?.to_device(x.device())?)?;
                }
                return Ok(ys);
            }
        }

        let xs = if x.dtype() == DType::F32 {
            x.clone()
        } else {
            x.to_dtype(DType::F32)?
        };
        let mut ys = self.matmul.as_ref().forward(&xs)?;
        if let Some(bias) = &self.bias {
            ys = ys.broadcast_add(&bias.to_dtype(DType::F32)?.to_device(x.device())?)?;
        }
        if self.output_dtype == DType::F32 {
            Ok(ys)
        } else {
            ys.to_dtype(self.output_dtype)
        }
    }
}

#[cfg(feature = "paged-attn")]
impl candle_core::Module for AfqLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut ys = self.inner.forward(x)?;
        if let Some(bias) = &self.bias {
            ys = ys.broadcast_add(&bias.to_dtype(ys.dtype())?.to_device(x.device())?)?;
        }
        if ys.dtype() == self.output_dtype {
            Ok(ys)
        } else {
            ys.to_dtype(self.output_dtype)
        }
    }
}

pub fn linear_b(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    vb: VarBuilder,
    isq: Option<&str>,
) -> Result<IsqLinear> {
    let target_device = vb.device().clone();
    let module_path = vb.prefix();
    let linear = if bias {
        candle_nn::linear(in_dim, out_dim, vb)?
    } else {
        candle_nn::linear_no_bias(in_dim, out_dim, vb)?
    };
    LinearX::new_with_module_path(linear, isq, &target_device, &module_path)
}

pub fn linear_no_bias(
    in_dim: usize,
    out_dim: usize,
    vb: VarBuilder,
    isq: Option<&str>,
) -> Result<IsqLinear> {
    let target_device = vb.device().clone();
    let module_path = vb.prefix();
    let linear = candle_nn::linear_no_bias(in_dim, out_dim, vb)?;
    LinearX::new_with_module_path(linear, isq, &target_device, &module_path)
}

fn linear_to_device(linear: Linear, device: &Device) -> Result<Linear> {
    let weight = linear.weight().to_device(device)?;
    let bias = linear.bias().map(|b| b.to_device(device)).transpose()?;
    Ok(Linear::new(weight, bias))
}

fn qtensor_to_device(qtensor: QTensor, device: &Device) -> Result<QTensor> {
    match device {
        Device::Cpu => Ok(qtensor),
        #[cfg(feature = "metal")]
        Device::Metal(metal) if matches!(qtensor.device(), Device::Cpu) => {
            load_cpu_qtensor_to_metal(qtensor, metal)
        }
        _ => Ok(qtensor),
    }
}

#[cfg(feature = "metal")]
fn load_cpu_qtensor_to_metal(
    qtensor: QTensor,
    device: &candle_core::MetalDevice,
) -> Result<QTensor> {
    let dtype = qtensor.dtype();
    let shape = qtensor.shape().clone();
    let data = qtensor.data()?;

    macro_rules! load {
        ($block:ty) => {{
            let blocks = bytes_as_blocks::<$block>(data.as_ref())?;
            let storage = candle_core::quantized::metal::load_quantized(device, blocks)?;
            QTensor::new(storage, shape)
        }};
    }

    match dtype {
        GgmlDType::Q4_0 => load!(k_quants::BlockQ4_0),
        GgmlDType::Q4_1 => load!(k_quants::BlockQ4_1),
        GgmlDType::Q5_0 => load!(k_quants::BlockQ5_0),
        GgmlDType::Q5_1 => load!(k_quants::BlockQ5_1),
        GgmlDType::Q8_0 => load!(k_quants::BlockQ8_0),
        GgmlDType::Q8_1 => load!(k_quants::BlockQ8_1),
        GgmlDType::Q2K => load!(k_quants::BlockQ2K),
        GgmlDType::Q3K => load!(k_quants::BlockQ3K),
        GgmlDType::Q4K => load!(k_quants::BlockQ4K),
        GgmlDType::Q5K => load!(k_quants::BlockQ5K),
        GgmlDType::Q6K => load!(k_quants::BlockQ6K),
        GgmlDType::Q8K => load!(k_quants::BlockQ8K),
        other => candle_core::bail!("unsupported ISQ metal upload dtype: {other:?}"),
    }
}

#[cfg(feature = "metal")]
fn bytes_as_blocks<T>(bytes: &[u8]) -> Result<&[T]> {
    let block_size = std::mem::size_of::<T>();
    if block_size == 0 || bytes.len() % block_size != 0 {
        candle_core::bail!(
            "quantized byte length is not divisible by block size: bytes={} block_size={block_size}",
            bytes.len()
        );
    }

    let (prefix, blocks, suffix) = unsafe { bytes.align_to::<T>() };
    if !prefix.is_empty() || !suffix.is_empty() {
        candle_core::bail!(
            "quantized bytes are not aligned for block upload: prefix={} suffix={}",
            prefix.len(),
            suffix.len()
        );
    }
    Ok(blocks)
}

pub fn resolve_isq_dtype(value: &str, device: &Device) -> Result<ResolvedIsq> {
    let dtype = value
        .split([';', ':', ','])
        .next()
        .unwrap_or(value)
        .trim()
        .to_ascii_lowercase();

    if let Ok(bits) = IsqBits::try_from(dtype.as_str()) {
        return Ok(bits.resolve(device));
    }

    match dtype.as_str() {
        "q40" | "q4_0" => Ok(ResolvedIsq::Ggml(GgmlDType::Q4_0)),
        "q4" | "q41" | "q4_1" => Ok(ResolvedIsq::Ggml(GgmlDType::Q4_1)),
        "q50" | "q5_0" => Ok(ResolvedIsq::Ggml(GgmlDType::Q5_0)),
        "q5" | "q51" | "q5_1" => Ok(ResolvedIsq::Ggml(GgmlDType::Q5_1)),
        "q8" | "q80" | "q8_0" => Ok(ResolvedIsq::Ggml(GgmlDType::Q8_0)),
        "q2k" | "q2_k" => Ok(ResolvedIsq::Ggml(GgmlDType::Q2K)),
        "q3k" | "q3_k" => Ok(ResolvedIsq::Ggml(GgmlDType::Q3K)),
        "q4k" | "q4_k" => Ok(ResolvedIsq::Ggml(GgmlDType::Q4K)),
        "q5k" | "q5_k" => Ok(ResolvedIsq::Ggml(GgmlDType::Q5K)),
        "q6k" | "q6_k" => Ok(ResolvedIsq::Ggml(GgmlDType::Q6K)),
        #[cfg(feature = "paged-attn")]
        "afq2" => Ok(ResolvedIsq::Afq(AfqBits::Two)),
        #[cfg(feature = "paged-attn")]
        "afq3" => Ok(ResolvedIsq::Afq(AfqBits::Three)),
        #[cfg(feature = "paged-attn")]
        "afq4" => Ok(ResolvedIsq::Afq(AfqBits::Four)),
        #[cfg(feature = "paged-attn")]
        "afq6" => Ok(ResolvedIsq::Afq(AfqBits::Six)),
        #[cfg(feature = "paged-attn")]
        "afq8" => Ok(ResolvedIsq::Afq(AfqBits::Eight)),
        #[cfg(not(feature = "paged-attn"))]
        "afq2" => Ok(ResolvedIsq::AfqFallback {
            requested: "afq2",
            fallback: GgmlDType::Q2K,
        }),
        #[cfg(not(feature = "paged-attn"))]
        "afq3" => Ok(ResolvedIsq::AfqFallback {
            requested: "afq3",
            fallback: GgmlDType::Q3K,
        }),
        #[cfg(not(feature = "paged-attn"))]
        "afq4" => Ok(ResolvedIsq::AfqFallback {
            requested: "afq4",
            fallback: GgmlDType::Q4K,
        }),
        #[cfg(not(feature = "paged-attn"))]
        "afq6" => Ok(ResolvedIsq::AfqFallback {
            requested: "afq6",
            fallback: GgmlDType::Q6K,
        }),
        #[cfg(not(feature = "paged-attn"))]
        "afq8" => Ok(ResolvedIsq::AfqFallback {
            requested: "afq8",
            fallback: GgmlDType::Q8_0,
        }),
        other => candle_core::bail!(
            "unsupported isq dtype {other:?}; use auto/auto8/auto6/auto4/8/6/4/q8_0/q6_k/q5_k/q4_k/q3_k/q2_k"
        ),
    }
}

pub fn parse_isq_dtype(value: &str) -> Result<GgmlDType> {
    Ok(resolve_isq_dtype(value, &Device::Cpu)?.dtype())
}

pub fn resolve_isq_display(value: &str, device: &Device) -> Result<String> {
    Ok(resolve_isq_dtype(value, device)?.display_name())
}

#[derive(Debug, Clone)]
struct IsqSpec {
    dtype: Option<GgmlDType>,
    #[cfg(feature = "paged-attn")]
    afq: Option<AfqBits>,
    modules_to_not_convert: Vec<String>,
}

fn parse_isq_spec(value: &str, device: &Device) -> Result<IsqSpec> {
    let resolved = resolve_isq_dtype(value, device)?;
    #[cfg(not(feature = "paged-attn"))]
    let dtype = Some(resolved.dtype());
    #[cfg(feature = "paged-attn")]
    let mut dtype = Some(resolved.dtype());
    #[cfg(feature = "paged-attn")]
    let mut afq = None;
    #[cfg(feature = "paged-attn")]
    if let ResolvedIsq::Afq(bits) = resolved {
        dtype = None;
        afq = Some(bits);
    }
    if let ResolvedIsq::AfqFallback {
        requested,
        fallback,
    } = resolved
        && !AFQ_FALLBACK_WARNED.swap(true, Ordering::Relaxed)
    {
        tracing::warn!(
            "ISQ `{}` resolved by auto selection, but AFQ is not implemented in this runtime yet; falling back to {}.",
            requested,
            ggml_dtype_name(fallback)
        );
    }
    let mut modules_to_not_convert = Vec::new();

    for part in value.split([';', ',']) {
        let part = part.trim();
        let Some((key, values)) = part.split_once('=') else {
            continue;
        };
        let key = key.trim().to_ascii_lowercase();
        if matches!(
            key.as_str(),
            "skip" | "skip_modules" | "modules_to_not_convert"
        ) {
            modules_to_not_convert.extend(
                values
                    .split(['|', '+'])
                    .map(str::trim)
                    .filter(|item| !item.is_empty())
                    .map(ToOwned::to_owned),
            );
        }
    }

    modules_to_not_convert.sort();
    modules_to_not_convert.dedup();
    Ok(IsqSpec {
        dtype,
        #[cfg(feature = "paged-attn")]
        afq,
        modules_to_not_convert,
    })
}

#[cfg(feature = "paged-attn")]
fn afq_bits_name(bits: AfqBits) -> &'static str {
    match bits {
        AfqBits::Two => "afq2",
        AfqBits::Three => "afq3",
        AfqBits::Four => "afq4",
        AfqBits::Six => "afq6",
        AfqBits::Eight => "afq8",
        AfqBits::Mxfp4 => "mxfp4",
    }
}

fn ggml_dtype_name(dtype: GgmlDType) -> &'static str {
    match dtype {
        GgmlDType::Q4_0 => "q4_0",
        GgmlDType::Q4_1 => "q4_1",
        GgmlDType::Q5_0 => "q5_0",
        GgmlDType::Q5_1 => "q5_1",
        GgmlDType::Q8_0 => "q8_0",
        GgmlDType::Q8_1 => "q8_1",
        GgmlDType::Q2K => "q2_k",
        GgmlDType::Q3K => "q3_k",
        GgmlDType::Q4K => "q4_k",
        GgmlDType::Q5K => "q5_k",
        GgmlDType::Q6K => "q6_k",
        GgmlDType::Q8K => "q8_k",
        _ => "unknown",
    }
}

fn should_skip_module(module_path: &str, modules_to_not_convert: &[String]) -> bool {
    if module_path.is_empty() {
        return false;
    }

    modules_to_not_convert
        .iter()
        .any(|item| module_path_matches_not_convert(module_path, item))
}

fn module_path_matches_not_convert(module_path: &str, item: &str) -> bool {
    let module_path = module_path.trim_end_matches(".weight");
    let item = item.trim().trim_end_matches(".weight");
    !item.is_empty()
        && (module_path == item
            || module_path.ends_with(item)
            || module_path.ends_with(&format!(".{item}"))
            || item.ends_with(module_path)
            || item.ends_with(&format!(".{module_path}")))
}

fn compatible_dtype(weight: &Tensor, requested: GgmlDType) -> Option<GgmlDType> {
    let last_dim = weight.dim(candle_core::D::Minus1).ok()?;
    if last_dim % requested.block_size() == 0 {
        Some(requested)
    } else if last_dim % GgmlDType::Q8_0.block_size() == 0 {
        Some(GgmlDType::Q8_0)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "cuda")]
    use super::try_fused_q8_silu_gate_up;
    use super::{
        IsqBits, IsqLinear, LinearX, ResolvedIsq, linear_is_prefill,
        module_path_matches_not_convert, parse_isq_dtype, resolve_isq_dtype, set_linear_is_prefill,
    };
    use candle_core::quantized::GgmlDType;

    #[test]
    fn test_parse_isq_dtype_aliases() -> anyhow::Result<()> {
        assert_eq!(parse_isq_dtype("q4_k")?, GgmlDType::Q4K);
        assert_eq!(parse_isq_dtype("q4k")?, GgmlDType::Q4K);
        assert_eq!(parse_isq_dtype("q8")?, GgmlDType::Q8_0);
        assert_eq!(parse_isq_dtype("auto")?, GgmlDType::Q8_0);
        Ok(())
    }

    #[test]
    fn test_auto_isq_resolves_like_mistralrs_on_cpu() -> anyhow::Result<()> {
        let device = candle_core::Device::Cpu;
        assert_eq!(
            IsqBits::Four.resolve(&device),
            ResolvedIsq::Ggml(GgmlDType::Q4K)
        );
        assert_eq!(
            resolve_isq_dtype("auto6", &device)?,
            ResolvedIsq::Ggml(GgmlDType::Q6K)
        );
        assert_eq!(
            resolve_isq_dtype("8", &device)?,
            ResolvedIsq::Ggml(GgmlDType::Q8_0)
        );
        Ok(())
    }

    #[test]
    fn test_isq_linear_preserves_shape_and_dtype() -> anyhow::Result<()> {
        let device = candle_core::Device::Cpu;
        let weight =
            candle_core::Tensor::zeros((4usize, 32usize), candle_core::DType::F32, &device)?;
        let linear = candle_nn::Linear::new(weight, None);
        let linear = IsqLinear::new(linear, Some("q8_0"), &device)?;
        let x = candle_core::Tensor::ones((2usize, 32usize), candle_core::DType::F32, &device)?;
        let y = linear.forward(&x)?;
        assert_eq!(y.dims(), &[2, 4]);
        assert_eq!(y.dtype(), candle_core::DType::F32);
        Ok(())
    }

    #[test]
    fn test_linear_prefill_guard_restores_previous_state() {
        assert!(!linear_is_prefill());
        {
            let _guard = set_linear_is_prefill(true);
            assert!(linear_is_prefill());
            {
                let _nested = set_linear_is_prefill(false);
                assert!(!linear_is_prefill());
            }
            assert!(linear_is_prefill());
        }
        assert!(!linear_is_prefill());
    }

    #[test]
    fn test_isq_skip_module_keeps_dense_linear() -> anyhow::Result<()> {
        let device = candle_core::Device::Cpu;
        let weight =
            candle_core::Tensor::zeros((4usize, 32usize), candle_core::DType::F32, &device)?;
        let linear = candle_nn::Linear::new(weight, None);
        let linear = IsqLinear::new_with_module_path(
            linear,
            Some("q8_0;skip=layers.0.self_attn.q_proj"),
            &device,
            "thinker.model.layers.0.self_attn.q_proj",
        )?;
        assert!(matches!(linear, LinearX::Linear(_)));
        assert!(module_path_matches_not_convert(
            "thinker.lm_head",
            "lm_head"
        ));
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_q8_mmvq_fast_path_matches_fallback_on_cuda() -> anyhow::Result<()> {
        use candle_core::Module;

        let device = candle_core::Device::new_cuda(0)?;
        let weight = candle_core::Tensor::randn(0f32, 0.02f32, (64usize, 128usize), &device)?
            .to_dtype(candle_core::DType::BF16)?;
        let linear = candle_nn::Linear::new(weight, None);
        let linear = IsqLinear::new(linear, Some("q8_0"), &device)?;
        let LinearX::QLinear(qlinear) = linear else {
            anyhow::bail!("expected QLinear");
        };
        let x = candle_core::Tensor::randn(0f32, 0.02f32, (1usize, 1usize, 128usize), &device)?
            .to_dtype(candle_core::DType::BF16)?;

        let fast = qlinear.forward(&x)?.to_dtype(candle_core::DType::F32)?;
        let fallback = qlinear
            .matmul
            .as_ref()
            .forward(&x.to_dtype(candle_core::DType::F32)?)?
            .to_dtype(candle_core::DType::BF16)?
            .to_dtype(candle_core::DType::F32)?;
        let diff = (fast - fallback)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(
            diff < 0.05,
            "q8 mmvq fast path diverged from fallback: max abs diff {diff}"
        );
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_q8_mmvq_fused_silu_gate_up_matches_unfused_on_cuda() -> anyhow::Result<()> {
        let device = candle_core::Device::new_cuda(0)?;
        let gate_weight = candle_core::Tensor::randn(0f32, 0.02f32, (64usize, 128usize), &device)?
            .to_dtype(candle_core::DType::BF16)?;
        let up_weight = candle_core::Tensor::randn(0f32, 0.02f32, (64usize, 128usize), &device)?
            .to_dtype(candle_core::DType::BF16)?;
        let gate = IsqLinear::new(
            candle_nn::Linear::new(gate_weight, None),
            Some("q8_0"),
            &device,
        )?;
        let up = IsqLinear::new(
            candle_nn::Linear::new(up_weight, None),
            Some("q8_0"),
            &device,
        )?;
        let x = candle_core::Tensor::randn(0f32, 0.02f32, (1usize, 1usize, 128usize), &device)?
            .to_dtype(candle_core::DType::BF16)?;

        let fused = try_fused_q8_silu_gate_up(&gate, &up, &x)?
            .ok_or_else(|| anyhow::anyhow!("expected fused q8 gate/up path"))?
            .to_dtype(candle_core::DType::F32)?;
        let gate_y = gate.forward(&x)?;
        let up_y = up.forward(&x)?;
        let unfused = candle_nn::ops::silu(&gate_y)?
            .broadcast_mul(&up_y)?
            .to_dtype(candle_core::DType::F32)?;

        let diff = (fused - unfused)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(
            diff < 0.05,
            "q8 fused gate/up path diverged from unfused: max abs diff {diff}"
        );
        Ok(())
    }
}
