import pytest

from qwen3_asr_rs import Qwen3ASR, __version__


def test_version_is_exposed():
    assert __version__ == "0.1.0"


def test_missing_local_model_dir_fails_during_model_loading(tmp_path):
    missing = tmp_path / "missing-model"
    missing.mkdir()
    with pytest.raises(RuntimeError) as exc:
        Qwen3ASR.from_pretrained(str(missing), device="cpu")
    assert "Candle Qwen3-ASR inference is not wired yet" not in str(exc.value)


def test_unknown_device_has_clear_error():
    with pytest.raises(RuntimeError, match="unknown device"):
        Qwen3ASR.from_pretrained("Qwen/Qwen3-ASR-0.6B", device="tpu")


def test_empty_model_id_has_clear_error():
    with pytest.raises(RuntimeError, match="model_id_or_path must not be empty"):
        Qwen3ASR.from_pretrained("", device="cpu")
