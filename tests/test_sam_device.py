"""SAM device resolution (#567): ``auto`` prefers CUDA, then Apple MPS, then
CPU; an accelerator that cannot host or run the model falls back to CPU
explicitly (with a warning) instead of failing or, worse, silently forcing
CPU when a GPU exists."""
import logging
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scilink.skills._shared import particle_analyzer as pa  # noqa: E402


def _fake_backends(monkeypatch, *, cuda: bool, mps):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    if mps is None:
        monkeypatch.setattr(torch.backends, "mps", None, raising=False)
    else:
        monkeypatch.setattr(torch.backends, "mps",
                            SimpleNamespace(is_available=lambda: mps), raising=False)


@pytest.mark.parametrize("cuda,mps,expected", [
    (True, True, "cuda"),
    (False, True, "mps"),
    (False, False, "cpu"),
    (False, None, "cpu"),   # a torch build without the MPS backend at all
])
def test_auto_prefers_cuda_then_mps_then_cpu(monkeypatch, cuda, mps, expected):
    _fake_backends(monkeypatch, cuda=cuda, mps=mps)
    assert pa.ParticleAnalyzer._resolve_device("auto") == expected


def test_explicit_device_is_honored(monkeypatch):
    _fake_backends(monkeypatch, cuda=False, mps=True)
    assert pa.ParticleAnalyzer._resolve_device("cpu") == "cpu"
    assert pa.ParticleAnalyzer._resolve_device("cuda:1") == "cuda:1"


class _Model:
    def __init__(self, fail_on=()):
        self.fail_on = set(fail_on)
        self.device = None

    def to(self, device):
        if device in self.fail_on:
            raise RuntimeError(f"cannot place on {device}")
        self.device = device
        return self


def _analyzer(monkeypatch, model, device):
    """A ParticleAnalyzer with the model registry and checkpoint download
    stubbed out, on the given resolved device."""
    import sys
    registry = {"vit_h": lambda checkpoint: model}
    monkeypatch.setitem(sys.modules, "segment_anything",
                        SimpleNamespace(sam_model_registry=registry,
                                        SamAutomaticMaskGenerator=None))
    monkeypatch.setattr(pa.ParticleAnalyzer, "_ensure_checkpoint",
                        classmethod(lambda cls, p, m: "ckpt.pth"))
    return pa.ParticleAnalyzer(checkpoint_path="ckpt.pth", model_type="vit_h", device=device)


def test_placement_failure_falls_back_to_cpu_with_warning(monkeypatch, caplog):
    model = _Model(fail_on={"mps"})
    with caplog.at_level(logging.WARNING, logger=pa.logger.name):
        an = _analyzer(monkeypatch, model, "mps")
    assert an.device == "cpu" and model.device == "cpu"
    assert "falling back to CPU" in caplog.text


def test_placement_failure_on_cpu_is_not_swallowed(monkeypatch):
    with pytest.raises(RuntimeError):
        _analyzer(monkeypatch, _Model(fail_on={"cpu"}), "cpu")


def test_inference_failure_on_accelerator_retries_on_cpu(monkeypatch, caplog):
    """MPS gaps show up at inference, not placement: the first generate()
    raises, the retry runs on CPU, and the analyzer remembers the switch."""
    import sys
    model = _Model()
    calls = []

    class _Gen:
        def __init__(self, sam, **params):
            self.sam = sam

        def generate(self, img):
            calls.append(self.sam.device)
            if self.sam.device == "mps":
                raise NotImplementedError("aten::upsample_bicubic2d not implemented for MPS")
            return [{"area": 10}]

    an = _analyzer(monkeypatch, model, "mps")
    sys.modules["segment_anything"].SamAutomaticMaskGenerator = _Gen
    with caplog.at_level(logging.WARNING, logger=pa.logger.name):
        masks = an._run_sam(np.zeros((8, 8, 3), dtype=np.uint8), "default")
    assert masks == [{"area": 10}] and calls == ["mps", "cpu"]
    assert an.device == "cpu" and "retrying on CPU" in caplog.text
    # a later call goes straight to CPU
    calls.clear()
    an._run_sam(np.zeros((8, 8, 3), dtype=np.uint8), "default")
    assert calls == ["cpu"]


def test_mps_float32_coords_casts_the_generator_points():
    """MPS has no float64: the generator's point coordinates are cast to
    float32 on MPS (the live failure was 'Cannot convert a MPS Tensor to
    float64'), once, and not on other devices."""
    calls = []

    class _Transform:
        def apply_coords(self, coords, original_size):
            calls.append(original_size)
            return np.asarray(coords, dtype=np.float64) * 2

    gen = SimpleNamespace(predictor=SimpleNamespace(transform=_Transform()))
    pa.ParticleAnalyzer._mps_float32_coords(gen)
    out = gen.predictor.transform.apply_coords(np.array([[1.0, 2.0]]), (8, 8))
    assert out.dtype == np.float32 and out.tolist() == [[2.0, 4.0]] and calls == [(8, 8)]
    pa.ParticleAnalyzer._mps_float32_coords(gen)  # idempotent: not wrapped twice
    gen.predictor.transform.apply_coords(np.array([[1.0, 2.0]]), (8, 8))
    assert len(calls) == 2
    pa.ParticleAnalyzer._mps_float32_coords(SimpleNamespace())  # no predictor: no-op
