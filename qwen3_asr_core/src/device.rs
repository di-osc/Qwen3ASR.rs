use std::fmt;

use anyhow::{Result, bail};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DevicePreference {
    Auto,
    Cpu,
    Cuda,
    Metal,
}

impl DevicePreference {
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "cuda" => Ok(Self::Cuda),
            "metal" => Ok(Self::Metal),
            other => bail!("unknown device {other:?}; expected auto, cpu, cuda, or metal"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DTypePreference {
    Auto,
    F32,
    F16,
    BF16,
}

impl DTypePreference {
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "f32" => Ok(Self::F32),
            "f16" => Ok(Self::F16),
            "bf16" => Ok(Self::BF16),
            other => bail!("unknown dtype {other:?}; expected auto, f32, f16, or bf16"),
        }
    }

    pub fn to_candle(self) -> candle_core::DType {
        match self {
            Self::Auto => candle_core::DType::F32,
            Self::F32 => candle_core::DType::F32,
            Self::F16 => candle_core::DType::F16,
            Self::BF16 => candle_core::DType::BF16,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedDevice {
    Cpu,
    Cuda,
    Metal,
}

impl fmt::Display for ResolvedDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => f.write_str("cpu"),
            Self::Cuda => f.write_str("cuda"),
            Self::Metal => f.write_str("metal"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedOptions {
    pub device: ResolvedDevice,
    pub dtype: DTypePreference,
}

impl ResolvedOptions {
    pub fn to_candle_device(self) -> Result<candle_core::Device> {
        match self.device {
            ResolvedDevice::Cpu => Ok(candle_core::Device::Cpu),
            ResolvedDevice::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    candle_core::Device::new_cuda(0)
                        .map_err(|err| anyhow::anyhow!("failed to create CUDA device 0: {err}"))
                }
                #[cfg(not(feature = "cuda"))]
                {
                    bail!("cuda device requested but this package was built without CUDA support")
                }
            }
            ResolvedDevice::Metal => {
                #[cfg(feature = "metal")]
                {
                    candle_core::Device::new_metal(0)
                        .map_err(|err| anyhow::anyhow!("failed to create Metal device 0: {err}"))
                }
                #[cfg(not(feature = "metal"))]
                {
                    bail!("metal device requested but this package was built without Metal support")
                }
            }
        }
    }
}

pub fn resolve_options(device: &str, dtype: &str) -> Result<ResolvedOptions> {
    let device_pref = DevicePreference::parse(device)?;
    let dtype_pref = DTypePreference::parse(dtype)?;
    let resolved_device = resolve_device(device_pref)?;
    let resolved_dtype = match dtype_pref {
        DTypePreference::Auto => match resolved_device {
            ResolvedDevice::Cpu => DTypePreference::F32,
            ResolvedDevice::Cuda | ResolvedDevice::Metal => DTypePreference::F16,
        },
        explicit => explicit,
    };
    Ok(ResolvedOptions {
        device: resolved_device,
        dtype: resolved_dtype,
    })
}

fn resolve_device(pref: DevicePreference) -> Result<ResolvedDevice> {
    match pref {
        DevicePreference::Auto => Ok(auto_device()),
        DevicePreference::Cpu => Ok(ResolvedDevice::Cpu),
        DevicePreference::Cuda => {
            #[cfg(feature = "cuda")]
            {
                Ok(ResolvedDevice::Cuda)
            }
            #[cfg(not(feature = "cuda"))]
            {
                bail!("cuda device requested but this package was built without CUDA support")
            }
        }
        DevicePreference::Metal => {
            #[cfg(feature = "metal")]
            {
                Ok(ResolvedDevice::Metal)
            }
            #[cfg(not(feature = "metal"))]
            {
                bail!("metal device requested but this package was built without Metal support")
            }
        }
    }
}

fn auto_device() -> ResolvedDevice {
    #[cfg(feature = "cuda")]
    {
        return ResolvedDevice::Cuda;
    }
    #[cfg(all(not(feature = "cuda"), feature = "metal"))]
    {
        return ResolvedDevice::Metal;
    }
    #[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
    {
        ResolvedDevice::Cpu
    }
}

#[cfg(test)]
mod tests {
    use super::{DTypePreference, ResolvedDevice, resolve_options};

    #[test]
    fn parses_cpu_device_and_auto_dtype() -> anyhow::Result<()> {
        let resolved = resolve_options("cpu", "auto")?;
        assert_eq!(resolved.device, ResolvedDevice::Cpu);
        assert_eq!(resolved.dtype, DTypePreference::F32);
        Ok(())
    }

    #[test]
    fn rejects_unknown_device() {
        let err = resolve_options("tpu", "auto").unwrap_err().to_string();
        assert!(err.contains("unknown device"));
    }

    #[test]
    fn rejects_unknown_dtype() {
        let err = resolve_options("cpu", "int8").unwrap_err().to_string();
        assert!(err.contains("unknown dtype"));
    }

    #[test]
    fn parses_explicit_dtype_case_insensitively() -> anyhow::Result<()> {
        let resolved = resolve_options("cpu", "BF16")?;
        assert_eq!(resolved.dtype, DTypePreference::BF16);
        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_auto_dtype_uses_f16_compute() -> anyhow::Result<()> {
        let resolved = resolve_options("metal", "auto")?;
        assert_eq!(resolved.device, ResolvedDevice::Metal);
        assert_eq!(resolved.dtype, DTypePreference::F16);
        Ok(())
    }
}
