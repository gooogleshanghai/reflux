"""Decoder-layer resolution and capture/injection hooks."""
from __future__ import annotations
from typing import List, Optional

import torch


def decoder_layers(model) -> List[torch.nn.Module]:
    """Return the decoder-layer module list for common HF architectures
    (Gemma3 / Qwen3 / LLaMA-style)."""
    m = model
    for attr in ("model", "language_model", "transformer"):
        if hasattr(m, attr):
            m = getattr(m, attr)
        if hasattr(m, "layers"):
            return list(m.layers)
    raise ValueError("could not locate decoder layers")


class Capture:
    """Forward hook that records the residual stream after a given layer."""

    def __init__(self, module: torch.nn.Module):
        self.value: Optional[torch.Tensor] = None
        self.handle = module.register_forward_hook(self._hook, with_kwargs=True)

    def _hook(self, module, args, kwargs, output):
        out = output[0] if isinstance(output, tuple) else output
        self.value = out.detach()
        return output

    def remove(self):
        self.handle.remove()


class Inject:
    """Forward-pre hook that rewrites the residual stream entering a layer.

    ``fn(h) -> h`` is applied to the hidden states when ``active`` is set.
    """

    def __init__(self, module: torch.nn.Module, fn):
        self.fn = fn
        self.active = False
        self.handle = module.register_forward_pre_hook(self._hook, with_kwargs=True)

    def _hook(self, module, args, kwargs):
        if not self.active:
            return args, kwargs
        h = args[0] if args else kwargs.get("hidden_states")
        h = self.fn(h)
        if args:
            return (h,) + args[1:], kwargs
        kwargs = dict(kwargs)
        kwargs["hidden_states"] = h
        return args, kwargs

    def remove(self):
        self.handle.remove()


def shift_prev(z: torch.Tensor) -> torch.Tensor:
    """Shift payloads one position back: position t carries position t-1."""
    zero = torch.zeros_like(z[:, :1, :])
    return torch.cat([zero, z[:, :-1, :].clone()], dim=1)
