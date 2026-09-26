"""atomai's empty TensorDataset placeholder must survive current PyTorch."""
import pytest


def test_atomai_trainer_constructs_on_this_torch():
    torch = pytest.importorskip("torch")
    pytest.importorskip("atomai")
    from scilink.skills._shared._atomai_compat import ensure_atomai_compat
    ensure_atomai_compat()
    import torch.utils.data as tud
    empty = tud.TensorDataset()                       # the placeholder atomai makes
    assert len(empty) == 0
    a = torch.zeros(3, 2); b = torch.ones(3)
    ds = tud.TensorDataset(a, b)                      # the real thing is untouched
    assert len(ds) == 3 and ds[1][1].item() == 1.0
    from atomai.trainers.trainer import BaseTrainer  # constructs the placeholders
    BaseTrainer()
