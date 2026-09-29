"""atomai 0.8.1 on PyTorch >= 2.10.

atomai's trainers create an EMPTY ``torch.utils.data.TensorDataset()`` as a
placeholder before any data is set. PyTorch 2.10 rejects a TensorDataset with
no tensors ("Size mismatch between tensors"), so every atomai model load, and
with it the DCNN atom finder, fails at construction. Upstream atomai has not
changed since 2025-06. Until it does, the placeholder is allowed here: the
class is replaced by a subclass that accepts zero tensors and is otherwise
identical. Call :func:`ensure_atomai_compat` before importing atomai.
"""
from __future__ import annotations

_done = False


def ensure_atomai_compat() -> None:
    global _done
    if _done:
        return
    _done = True
    try:
        import torch.utils.data as tud
    except Exception:  # noqa: BLE001 - no torch, nothing to patch
        return
    original = tud.TensorDataset
    try:
        original()
        return                                   # this torch accepts the placeholder
    except (AssertionError, IndexError):
        pass

    class TensorDataset(original):               # type: ignore[misc,valid-type]
        """torch's TensorDataset that also accepts no tensors (atomai's placeholder)."""

        def __init__(self, *tensors):
            if not tensors:
                self.tensors = ()
                return
            super().__init__(*tensors)

        def __len__(self):
            return self.tensors[0].size(0) if self.tensors else 0

    TensorDataset.__name__ = original.__name__
    TensorDataset.__qualname__ = original.__qualname__
    TensorDataset.__module__ = original.__module__
    tud.TensorDataset = TensorDataset
