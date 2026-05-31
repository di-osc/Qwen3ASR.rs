#![allow(clippy::useless_conversion)]

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use qwen3_asr_core::{Qwen3Asr, StreamOptions, TranscribeOptions};

fn to_py_err(err: anyhow::Error) -> PyErr {
    PyRuntimeError::new_err(format!("{err:#}"))
}

fn samples_to_vec(samples: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    if let Ok(samples) = samples.extract::<Vec<f32>>() {
        return Ok(samples);
    }

    samples.call_method0("tolist")?.extract::<Vec<f32>>()
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

#[pyclass(name = "Qwen3ASRStream")]
struct PyQwen3AsrStream {
    inner: qwen3_asr_core::Qwen3AsrStream,
}

#[pymethods]
impl PyQwen3Asr {
    #[staticmethod]
    #[pyo3(signature = (model_id_or_path, device = "auto", dtype = "auto", use_flash_attn = false, isq = None))]
    fn from_pretrained(
        model_id_or_path: &str,
        device: &str,
        dtype: &str,
        use_flash_attn: bool,
        isq: Option<&str>,
    ) -> PyResult<Self> {
        let inner = Qwen3Asr::from_pretrained(model_id_or_path, device, dtype, use_flash_attn, isq)
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

    #[getter]
    fn isq(&self) -> Option<String> {
        self.inner.isq().map(str::to_string)
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

    #[pyo3(signature = (
        language = None,
        context = "",
        chunk_size_sec = 2.0,
        unfixed_chunk_num = 2,
        unfixed_token_num = 5,
        max_new_tokens = 256,
        audio_window_sec = None,
        text_window_tokens = None
    ))]
    fn start_stream(
        &self,
        language: Option<String>,
        context: &str,
        chunk_size_sec: f32,
        unfixed_chunk_num: usize,
        unfixed_token_num: usize,
        max_new_tokens: usize,
        audio_window_sec: Option<f32>,
        text_window_tokens: Option<usize>,
    ) -> PyResult<PyQwen3AsrStream> {
        let inner = self
            .inner
            .start_stream(StreamOptions {
                language,
                context: context.to_string(),
                chunk_size_sec,
                unfixed_chunk_num,
                unfixed_token_num,
                max_new_tokens,
                audio_window_sec,
                text_window_tokens,
            })
            .map_err(to_py_err)?;
        Ok(PyQwen3AsrStream { inner })
    }
}

#[pymethods]
impl PyQwen3AsrStream {
    #[pyo3(signature = (samples, sample_rate = 16000))]
    fn push_audio_chunk(
        &mut self,
        samples: &Bound<'_, PyAny>,
        sample_rate: u32,
    ) -> PyResult<Option<PyTranscriptionResult>> {
        let samples = samples_to_vec(samples)?;
        let result = self
            .inner
            .push_audio_chunk(samples.as_slice(), sample_rate)
            .map_err(to_py_err)?;
        Ok(result.map(PyTranscriptionResult::from))
    }

    fn finish(&mut self) -> PyResult<PyTranscriptionResult> {
        let result = self.inner.finish().map_err(to_py_err)?;
        Ok(PyTranscriptionResult::from(result))
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyQwen3Asr>()?;
    m.add_class::<PyQwen3AsrStream>()?;
    m.add_class::<PyTranscriptionResult>()?;
    Ok(())
}

impl From<qwen3_asr_core::TranscriptionResult> for PyTranscriptionResult {
    fn from(result: qwen3_asr_core::TranscriptionResult) -> Self {
        Self {
            text: result.text,
            language: result.language,
            raw: result.raw,
        }
    }
}
