use anyhow::{Result, bail};
use candle_core::{DType, Device};

/// Resolve a device string to a candle Device.
pub fn resolve_device(value: &str) -> Result<Device> {
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

/// Auto-select the best available device.
pub fn auto_device() -> Result<Device> {
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

/// Auto-select dtype for the given device.
pub fn auto_dtype(device: &Device) -> Result<DType> {
    Ok(if device.is_cpu() {
        DType::F32
    } else {
        DType::BF16
    })
}

/// Human-readable label for a device.
pub fn device_label(device: &Device) -> &'static str {
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
