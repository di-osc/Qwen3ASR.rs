//! Resampling utilities.

use anyhow::{Result, bail};

#[cfg(feature = "audio-loading")]
use anyhow::Context;

#[cfg(feature = "audio-loading")]
pub fn resample_mono_f32(samples: &[f32], from_hz: u32, to_hz: u32) -> Result<Vec<f32>> {
    use rubato::audioadapter_buffers::direct::InterleavedSlice;
    use rubato::{
        Async, FixedAsync, Resampler, SincInterpolationParameters, SincInterpolationType,
        WindowFunction,
    };

    if from_hz == 0 || to_hz == 0 {
        bail!("invalid sample rates: from_hz={from_hz} to_hz={to_hz}");
    }
    if from_hz == to_hz {
        return Ok(samples.to_vec());
    }
    if samples.is_empty() {
        return Ok(vec![]);
    }

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let ratio = to_hz as f64 / from_hz as f64;
    let chunk_size = 1024.min(samples.len().max(1));
    let mut resampler =
        Async::<f32>::new_sinc(ratio, 2.0, &params, chunk_size, 1, FixedAsync::Input).map_err(
            |e| {
                anyhow::anyhow!(
                    "resampler creation failed for from_hz={from_hz} to_hz={to_hz}: {e}"
                )
            },
        )?;

    let input_frames = samples.len();
    let input = InterleavedSlice::new(samples, 1, input_frames)
        .map_err(|e| anyhow::anyhow!("failed to create resampler input buffer: {e}"))?;

    let output_len = resampler.process_all_needed_output_len(input_frames);
    let mut outdata = vec![0.0f32; output_len];
    let mut output = InterleavedSlice::new_mut(&mut outdata, 1, output_len)
        .map_err(|e| anyhow::anyhow!("failed to create resampler output buffer: {e}"))?;

    let (_nbr_in, nbr_out) = resampler
        .process_all_into_buffer(&input, &mut output, input_frames, None)
        .context("resampling failed")?;

    outdata.truncate(nbr_out);
    Ok(outdata)
}

#[cfg(not(feature = "audio-loading"))]
pub fn resample_mono_f32(_samples: &[f32], _from_hz: u32, _to_hz: u32) -> Result<Vec<f32>> {
    bail!("resample_mono_f32 requires the `audio-loading` feature")
}

#[cfg(all(test, feature = "audio-loading"))]
mod tests {
    use super::resample_mono_f32;

    #[test]
    fn test_resample_identity() -> anyhow::Result<()> {
        let x = vec![0.0f32, 0.5, -0.25, 1.0];
        let y = resample_mono_f32(&x, 16_000, 16_000)?;
        if y != x {
            anyhow::bail!("identity resample changed samples");
        }
        Ok(())
    }
}
