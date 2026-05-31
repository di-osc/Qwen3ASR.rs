# Qwen3-ASR Python Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first local-development Python package skeleton for Qwen3-ASR backed by Rust and Candle.

**Architecture:** Create a Rust workspace with a pure Rust core crate and a thin PyO3 extension crate. The core crate owns device/dtype parsing, transcription options, and a narrow model facade; the PyO3 crate exposes `qwen3_asr_rs.Qwen3ASR` and `TranscriptionResult` for Python. The first implementation intentionally returns a clear runtime error for real transcription until the Candle model port is added.

**Tech Stack:** Rust 2024, Candle 0.9, PyO3, maturin, pytest, Python 3.9+.

---

## File Structure

- `Cargo.toml`: root workspace and shared lints.
- `pyproject.toml`: maturin project metadata for the Python extension.
- `.gitignore`: Rust, Python, and maturin build outputs.
- `README.md`: local development commands and first API example.
- `qwen3_asr_core/Cargo.toml`: core library dependencies and Candle backend features.
- `qwen3_asr_core/src/lib.rs`: public exports from the core crate.
- `qwen3_asr_core/src/error.rs`: core `Result` alias.
- `qwen3_asr_core/src/device.rs`: parse and resolve `device` and `dtype` strings.
- `qwen3_asr_core/src/transcribe.rs`: transcription option/result structs and validation.
- `qwen3_asr_core/src/model.rs`: first `Qwen3Asr` facade with model metadata and a clear not-yet-implemented inference error.
- `qwen3_asr_py/Cargo.toml`: PyO3 extension crate.
- `qwen3_asr_py/src/lib.rs`: Python module, classes, and error conversion.
- `python/qwen3_asr_rs/__init__.py`: Python package shim and `__version__`.
- `tests/test_python_api.py`: Python smoke tests.

## Task 1: Workspace And Package Metadata

**Files:**
- Create: `Cargo.toml`
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `README.md`

- [ ] **Step 1: Create workspace metadata**

Write `Cargo.toml`:

```toml
[workspace]
members = ["qwen3_asr_core", "qwen3_asr_py"]
resolver = "3"

[workspace.package]
version = "0.1.0"
edition = "2024"
license = "MIT OR Apache-2.0"
repository = "https://github.com/wangmengdi/Qwen3ASR.rs"

[workspace.lints.rust]
unsafe_op_in_unsafe_fn = "deny"

[workspace.lints.clippy]
dbg_macro = "deny"
todo = "deny"
unwrap_used = "deny"
expect_used = "deny"
```

- [ ] **Step 2: Create maturin metadata**

Write `pyproject.toml`:

```toml
[build-system]
requires = ["maturin>=1.7,<2"]
build-backend = "maturin"

[project]
name = "qwen3-asr-rs"
version = "0.1.0"
description = "Python bindings for Qwen3-ASR inference using Rust and Candle"
readme = "README.md"
requires-python = ">=3.9"
license = { text = "MIT OR Apache-2.0" }
authors = [{ name = "Qwen3ASR.rs contributors" }]
classifiers = [
  "Development Status :: 3 - Alpha",
  "Intended Audience :: Developers",
  "Intended Audience :: Science/Research",
  "Programming Language :: Python :: 3",
  "Programming Language :: Rust",
  "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

[project.optional-dependencies]
dev = ["pytest>=8"]

[tool.maturin]
manifest-path = "qwen3_asr_py/Cargo.toml"
module-name = "qwen3_asr_rs._native"
python-source = "python"
features = ["pyo3/extension-module"]
```

- [ ] **Step 3: Create ignore rules**

Write `.gitignore`:

```gitignore
/target/
/.venv/
/.maturin/
/.pytest_cache/
/dist/
__pycache__/
*.py[cod]
*.so
*.dylib
*.dll
```

- [ ] **Step 4: Create README**

Write `README.md`:

```markdown
# Qwen3ASR.rs

Python bindings for Qwen3-ASR inference using Rust and Candle.

The first development milestone provides the Python package shape, device/dtype validation, and a stable API for the upcoming Candle inference port.

## Development

```bash
python -m venv .venv
. .venv/bin/activate
pip install maturin pytest
maturin develop
pytest
cargo test --workspace
```

## Python API

```python
from qwen3_asr_rs import Qwen3ASR

model = Qwen3ASR.from_pretrained("Qwen/Qwen3-ASR-0.6B", device="auto")
result = model.transcribe("audio.wav", language="Chinese")
print(result.text)
```

Real Candle inference is being ported from `lumosimmo/qwen3-asr-rs` in the next milestone.
```

- [ ] **Step 5: Verify metadata is present**

Run: `test -f Cargo.toml && test -f pyproject.toml && test -f README.md`

Expected: command exits with status 0.

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml pyproject.toml .gitignore README.md
git commit -m "chore: add rust python package metadata"
```

## Task 2: Core Device And Dtype Validation

**Files:**
- Create: `qwen3_asr_core/Cargo.toml`
- Create: `qwen3_asr_core/src/lib.rs`
- Create: `qwen3_asr_core/src/error.rs`
- Create: `qwen3_asr_core/src/device.rs`

- [ ] **Step 1: Write the failing device tests**

Create `qwen3_asr_core/src/device.rs` with tests first:

```rust
use std::fmt;

#[cfg(test)]
mod tests {
    use super::{DevicePreference, DTypePreference, ResolvedDevice, resolve_options};

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
}
```

- [ ] **Step 2: Add crate manifest**

Create `qwen3_asr_core/Cargo.toml`:

```toml
[package]
name = "qwen3_asr_core"
version.workspace = true
edition.workspace = true
license.workspace = true

[dependencies]
anyhow = "1"
candle-core = { version = "0.9" }
serde = { version = "1", features = ["derive"] }

[features]
default = []
cuda = ["candle-core/cuda"]
metal = ["candle-core/metal"]
```

- [ ] **Step 3: Add core exports**

Create `qwen3_asr_core/src/lib.rs`:

```rust
pub mod device;
pub mod error;

pub use device::{DTypePreference, DevicePreference, ResolvedDevice, ResolvedOptions};
```

- [ ] **Step 4: Add result alias**

Create `qwen3_asr_core/src/error.rs`:

```rust
pub type Result<T> = anyhow::Result<T>;
```

- [ ] **Step 5: Run test to verify it fails**

Run: `cargo test -p qwen3_asr_core device::tests -- --nocapture`

Expected: FAIL because the device types and `resolve_options` are not implemented.

- [ ] **Step 6: Implement device parsing**

Replace `qwen3_asr_core/src/device.rs` with:

```rust
use std::fmt;

use anyhow::{bail, Result};

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
}
```

- [ ] **Step 7: Run tests**

Run: `cargo test -p qwen3_asr_core device::tests -- --nocapture`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add qwen3_asr_core Cargo.lock
git commit -m "feat: add core device option validation"
```

## Task 3: Core Model Facade And Transcription Types

**Files:**
- Modify: `qwen3_asr_core/src/lib.rs`
- Create: `qwen3_asr_core/src/transcribe.rs`
- Create: `qwen3_asr_core/src/model.rs`

- [ ] **Step 1: Write transcription validation tests**

Create `qwen3_asr_core/src/transcribe.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::TranscribeOptions;

    #[test]
    fn default_max_new_tokens_is_256() {
        let opts = TranscribeOptions::default();
        assert_eq!(opts.max_new_tokens, 256);
    }

    #[test]
    fn rejects_zero_max_new_tokens() {
        let opts = TranscribeOptions {
            max_new_tokens: 0,
            ..TranscribeOptions::default()
        };
        let err = opts.validate().unwrap_err().to_string();
        assert!(err.contains("max_new_tokens must be greater than zero"));
    }
}
```

- [ ] **Step 2: Write model facade tests**

Create `qwen3_asr_core/src/model.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::Qwen3Asr;
    use crate::transcribe::TranscribeOptions;

    #[test]
    fn constructs_with_resolved_options() -> anyhow::Result<()> {
        let model = Qwen3Asr::from_pretrained("Qwen/Qwen3-ASR-0.6B", "cpu", "auto", false)?;
        assert_eq!(model.model_id_or_path(), "Qwen/Qwen3-ASR-0.6B");
        assert_eq!(model.device_label(), "cpu");
        Ok(())
    }

    #[test]
    fn transcription_reports_inference_not_wired_yet() -> anyhow::Result<()> {
        let model = Qwen3Asr::from_pretrained("Qwen/Qwen3-ASR-0.6B", "cpu", "auto", false)?;
        let err = model
            .transcribe_path("audio.wav", TranscribeOptions::default())
            .unwrap_err()
            .to_string();
        assert!(err.contains("Candle Qwen3-ASR inference is not wired yet"));
        Ok(())
    }
}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cargo test -p qwen3_asr_core -- --nocapture`

Expected: FAIL because the structs and methods are not implemented.

- [ ] **Step 4: Implement transcription types**

Replace `qwen3_asr_core/src/transcribe.rs` with:

```rust
use anyhow::{bail, Result};

#[derive(Debug, Clone)]
pub struct TranscribeOptions {
    pub language: Option<String>,
    pub context: String,
    pub max_new_tokens: usize,
}

impl Default for TranscribeOptions {
    fn default() -> Self {
        Self {
            language: None,
            context: String::new(),
            max_new_tokens: 256,
        }
    }
}

impl TranscribeOptions {
    pub fn validate(&self) -> Result<()> {
        if self.max_new_tokens == 0 {
            bail!("max_new_tokens must be greater than zero");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscriptionResult {
    pub text: String,
    pub language: Option<String>,
    pub raw: String,
}
```

- [ ] **Step 5: Implement model facade**

Replace `qwen3_asr_core/src/model.rs` with:

```rust
use anyhow::{bail, Result};

use crate::device::{ResolvedOptions, resolve_options};
use crate::transcribe::{TranscribeOptions, TranscriptionResult};

#[derive(Debug, Clone)]
pub struct Qwen3Asr {
    model_id_or_path: String,
    options: ResolvedOptions,
    use_flash_attn: bool,
}

impl Qwen3Asr {
    pub fn from_pretrained(
        model_id_or_path: &str,
        device: &str,
        dtype: &str,
        use_flash_attn: bool,
    ) -> Result<Self> {
        let trimmed = model_id_or_path.trim();
        if trimmed.is_empty() {
            bail!("model_id_or_path must not be empty");
        }
        let options = resolve_options(device, dtype)?;
        Ok(Self {
            model_id_or_path: trimmed.to_string(),
            options,
            use_flash_attn,
        })
    }

    pub fn model_id_or_path(&self) -> &str {
        &self.model_id_or_path
    }

    pub fn device_label(&self) -> String {
        self.options.device.to_string()
    }

    pub fn use_flash_attn(&self) -> bool {
        self.use_flash_attn
    }

    pub fn transcribe_path(
        &self,
        audio_path: &str,
        options: TranscribeOptions,
    ) -> Result<TranscriptionResult> {
        if audio_path.trim().is_empty() {
            bail!("audio path must not be empty");
        }
        options.validate()?;
        bail!(
            "Candle Qwen3-ASR inference is not wired yet; model={} device={}",
            self.model_id_or_path,
            self.options.device
        );
    }
}

#[cfg(test)]
mod tests {
    use super::Qwen3Asr;
    use crate::transcribe::TranscribeOptions;

    #[test]
    fn constructs_with_resolved_options() -> anyhow::Result<()> {
        let model = Qwen3Asr::from_pretrained("Qwen/Qwen3-ASR-0.6B", "cpu", "auto", false)?;
        assert_eq!(model.model_id_or_path(), "Qwen/Qwen3-ASR-0.6B");
        assert_eq!(model.device_label(), "cpu");
        Ok(())
    }

    #[test]
    fn transcription_reports_inference_not_wired_yet() -> anyhow::Result<()> {
        let model = Qwen3Asr::from_pretrained("Qwen/Qwen3-ASR-0.6B", "cpu", "auto", false)?;
        let err = model
            .transcribe_path("audio.wav", TranscribeOptions::default())
            .unwrap_err()
            .to_string();
        assert!(err.contains("Candle Qwen3-ASR inference is not wired yet"));
        Ok(())
    }
}
```

- [ ] **Step 6: Export new modules**

Replace `qwen3_asr_core/src/lib.rs` with:

```rust
pub mod device;
pub mod error;
pub mod model;
pub mod transcribe;

pub use device::{DTypePreference, DevicePreference, ResolvedDevice, ResolvedOptions};
pub use model::Qwen3Asr;
pub use transcribe::{TranscribeOptions, TranscriptionResult};
```

- [ ] **Step 7: Run core tests**

Run: `cargo test -p qwen3_asr_core -- --nocapture`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add qwen3_asr_core Cargo.lock
git commit -m "feat: add core qwen3 asr facade"
```

## Task 4: PyO3 Extension Module

**Files:**
- Create: `qwen3_asr_py/Cargo.toml`
- Create: `qwen3_asr_py/src/lib.rs`
- Create: `python/qwen3_asr_rs/__init__.py`

- [ ] **Step 1: Add PyO3 crate manifest**

Create `qwen3_asr_py/Cargo.toml`:

```toml
[package]
name = "qwen3_asr_py"
version.workspace = true
edition.workspace = true
license.workspace = true

[lib]
name = "qwen3_asr_rs"
crate-type = ["cdylib"]

[dependencies]
qwen3_asr_core = { path = "../qwen3_asr_core" }
pyo3 = { version = "0.22", features = ["abi3-py39"] }

[features]
default = []
cuda = ["qwen3_asr_core/cuda"]
metal = ["qwen3_asr_core/metal"]
```

- [ ] **Step 2: Add Python package shim**

Create `python/qwen3_asr_rs/__init__.py`:

```python
from ._native import Qwen3ASR, TranscriptionResult

__all__ = ["Qwen3ASR", "TranscriptionResult", "__version__"]
__version__ = "0.1.0"
```

- [ ] **Step 3: Write extension module**

Create `qwen3_asr_py/src/lib.rs`:

```rust
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use qwen3_asr_core::{Qwen3Asr, TranscribeOptions};

fn to_py_err(err: anyhow::Error) -> PyErr {
    PyRuntimeError::new_err(err.to_string())
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
            "TranscriptionResult(text={!r}, language={!r})",
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
        let inner =
            Qwen3Asr::from_pretrained(model_id_or_path, device, dtype, use_flash_attn)
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
```

- [ ] **Step 4: Run Rust tests**

Run: `cargo test --workspace -- --nocapture`

Expected: PASS.

- [ ] **Step 5: Build Python extension locally**

Run: `python -m pip install maturin pytest && maturin develop`

Expected: maturin installs `qwen3-asr-rs` in the active Python environment.

- [ ] **Step 6: Verify import manually**

Run: `python -c "import qwen3_asr_rs; print(qwen3_asr_rs.__version__)"`

Expected output contains `0.1.0`.

- [ ] **Step 7: Commit**

```bash
git add qwen3_asr_py python Cargo.lock
git commit -m "feat: expose initial python bindings"
```

## Task 5: Python Smoke Tests

**Files:**
- Create: `tests/test_python_api.py`

- [ ] **Step 1: Write Python API tests**

Create `tests/test_python_api.py`:

```python
import pytest

from qwen3_asr_rs import Qwen3ASR, __version__


def test_version_is_exposed():
    assert __version__ == "0.1.0"


def test_constructs_cpu_model():
    model = Qwen3ASR.from_pretrained("Qwen/Qwen3-ASR-0.6B", device="cpu")
    assert model.model_id_or_path == "Qwen/Qwen3-ASR-0.6B"
    assert model.device == "cpu"


def test_unknown_device_has_clear_error():
    with pytest.raises(RuntimeError, match="unknown device"):
        Qwen3ASR.from_pretrained("Qwen/Qwen3-ASR-0.6B", device="tpu")


def test_transcribe_reports_inference_not_wired_yet():
    model = Qwen3ASR.from_pretrained("Qwen/Qwen3-ASR-0.6B", device="cpu")
    with pytest.raises(RuntimeError, match="Candle Qwen3-ASR inference is not wired yet"):
        model.transcribe("audio.wav")
```

- [ ] **Step 2: Run pytest**

Run: `python -m pytest -q`

Expected: all tests pass after `maturin develop`.

- [ ] **Step 3: Run full verification**

Run: `cargo test --workspace && python -m pytest -q`

Expected: both commands pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_python_api.py
git commit -m "test: add python api smoke tests"
```

## Task 6: Final Cleanup

**Files:**
- Modify as needed based on formatter output.

- [ ] **Step 1: Format Rust**

Run: `cargo fmt --all -- --check`

Expected: PASS. If it fails, run `cargo fmt --all`, review the diff, and rerun `cargo fmt --all -- --check`.

- [ ] **Step 2: Run clippy**

Run: `cargo clippy --workspace --all-targets -- -D warnings`

Expected: PASS.

- [ ] **Step 3: Run final tests**

Run: `cargo test --workspace && python -m pytest -q`

Expected: PASS.

- [ ] **Step 4: Check git status**

Run: `git status --short`

Expected: no unstaged or uncommitted implementation changes remain.

## Self-Review

Spec coverage:

- Python package and maturin metadata are covered by Tasks 1 and 4.
- Core/PyO3 split is covered by Tasks 2, 3, and 4.
- Device and dtype validation are covered by Task 2.
- Python API shape is covered by Tasks 4 and 5.
- Fast offline tests are covered by Tasks 2, 3, and 5.
- Real Candle inference is intentionally deferred and represented by a clear runtime error in Task 3.

Placeholder scan:

- The plan does not use undefined "TBD" or "TODO" work.
- The only deferred inference behavior has an explicit error string and tests.

Type consistency:

- Rust core type is `Qwen3Asr`.
- Python class is `Qwen3ASR`.
- Python result type is `TranscriptionResult`.
- Module import path is `qwen3_asr_rs`.

