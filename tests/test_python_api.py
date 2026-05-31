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
