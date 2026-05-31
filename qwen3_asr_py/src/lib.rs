#![allow(clippy::useless_conversion)]

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use qwen3_asr_core::{Qwen3Asr, TranscribeOptions};

fn to_py_err(err: anyhow::Error) -> PyErr {
    PyRuntimeError::new_err(format!("{err:#}"))
}

#[pyclass(name = "TranscriptionResult")]
#[derive(Clone)]
struct PyTranscriptionResult {
    #[pyo3(get)]
    text: String,
    #[pyo3(get)]
    language: Option<String>,
    #[pyo3(get)]
    raw: String,
}

#[pymethods]
impl PyTranscriptionResult {
    fn __repr__(&self) -> String {
        format!(
            "TranscriptionResult(text={:?}, language={:?})",
            self.text, self.language
        )
    }
}

#[pyclass(name = "Qwen3ASR")]
struct PyQwen3Asr {
    inner: Qwen3Asr,
}

#[pymethods]
impl PyQwen3Asr {
    #[staticmethod]
    #[pyo3(signature = (model_id_or_path, device = "auto", dtype = "auto", use_flash_attn = false))]
    fn from_pretrained(
        model_id_or_path: &str,
        device: &str,
        dtype: &str,
        use_flash_attn: bool,
    ) -> PyResult<Self> {
        let inner = Qwen3Asr::from_pretrained(model_id_or_path, device, dtype, use_flash_attn)
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn model_id_or_path(&self) -> String {
        self.inner.model_id_or_path().to_string()
    }

    #[getter]
    fn device(&self) -> String {
        self.inner.device_label()
    }

    #[pyo3(signature = (audio, language = None, context = "", max_new_tokens = 256))]
    fn transcribe(
        &self,
        audio: &str,
        language: Option<String>,
        context: &str,
        max_new_tokens: usize,
    ) -> PyResult<PyTranscriptionResult> {
        let result = self
            .inner
            .transcribe_path(
                audio,
                TranscribeOptions {
                    language,
                    context: context.to_string(),
                    max_new_tokens,
                },
            )
            .map_err(to_py_err)?;
        Ok(PyTranscriptionResult {
            text: result.text,
            language: result.language,
            raw: result.raw,
        })
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyQwen3Asr>()?;
    m.add_class::<PyTranscriptionResult>()?;
    Ok(())
}
