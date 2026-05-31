//! X-infer style LinearX/QLinear layers with in-situ quantization.

use std::cell::Cell;
use std::sync::Arc;
#[cfg(feature = "timing")]
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "metal")]
use candle_core::quantized::k_quants;
use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder};

#[cfg(feature = "timing")]
static ISQ_QUANTIZE_US: AtomicU64 = AtomicU64::new(0);

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
}

pub type IsqLinear = LinearX;

#[derive(Clone)]
pub struct QLinear {
    matmul: Arc<QMatMul>,
    bias: Option<Tensor>,
    output_dtype: DType,
    dtype: GgmlDType,
}

impl std::fmt::Debug for LinearX {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Linear(linear) => f.debug_tuple("Linear").field(linear).finish(),
            Self::QLinear(linear) => f.debug_tuple("QLinear").field(linear).finish(),
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

        let spec = parse_isq_spec(isq)?;
        if should_skip_module(module_path, &spec.modules_to_not_convert) {
            let linear = linear_to_device(linear, target_device)?;
            return Ok(Self::Linear(linear));
        }

        let requested = spec.dtype;
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
        }
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

impl candle_core::Module for QLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
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

pub fn parse_isq_dtype(value: &str) -> Result<GgmlDType> {
    let dtype = value
        .split([';', ':', ','])
        .next()
        .unwrap_or(value)
        .trim()
        .to_ascii_lowercase();
    match dtype.as_str() {
        "q40" | "q4_0" => Ok(GgmlDType::Q4_0),
        "q4" | "q41" | "q4_1" => Ok(GgmlDType::Q4_1),
        "q50" | "q5_0" => Ok(GgmlDType::Q5_0),
        "q5" | "q51" | "q5_1" => Ok(GgmlDType::Q5_1),
        "q8" | "q80" | "q8_0" => Ok(GgmlDType::Q8_0),
        "q2k" | "q2_k" => Ok(GgmlDType::Q2K),
        "q3k" | "q3_k" => Ok(GgmlDType::Q3K),
        "q4k" | "q4_k" => Ok(GgmlDType::Q4K),
        "q5k" | "q5_k" => Ok(GgmlDType::Q5K),
        "q6k" | "q6_k" => Ok(GgmlDType::Q6K),
        other => candle_core::bail!(
            "unsupported isq dtype {other:?}; use q4_0/q4_1/q5_0/q5_1/q8_0/q2_k/q3_k/q4_k/q5_k/q6_k"
        ),
    }
}

#[derive(Debug, Clone)]
struct IsqSpec {
    dtype: GgmlDType,
    modules_to_not_convert: Vec<String>,
}

fn parse_isq_spec(value: &str) -> Result<IsqSpec> {
    let dtype = parse_isq_dtype(value)?;
    let mut modules_to_not_convert = vec!["lm_head".to_string()];

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
        modules_to_not_convert,
    })
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
    use super::{
        IsqLinear, LinearX, linear_is_prefill, module_path_matches_not_convert, parse_isq_dtype,
        set_linear_is_prefill,
    };
    use candle_core::quantized::GgmlDType;

    #[test]
    fn test_parse_isq_dtype_aliases() -> anyhow::Result<()> {
        assert_eq!(parse_isq_dtype("q4_k")?, GgmlDType::Q4K);
        assert_eq!(parse_isq_dtype("q4k")?, GgmlDType::Q4K);
        assert_eq!(parse_isq_dtype("q8")?, GgmlDType::Q8_0);
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
}
