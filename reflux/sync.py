"""ReFlux-sync: same-token increment feedback (synchronous schedule).

Two batched traversals over the input:
  record : plain forward; capture z_s(t) and z_d(t) at every position;
           payloads P(t) = z_s(t) - z_d(t).
  inject : forward again; the write at position t applies P(t), so every
           token is interpreted with its own committed increment in place.

Compared to the streaming schedule this trades one extra traversal for
higher fidelity (~2x FLOPs) and has no 1x decoding path.
"""
from __future__ import annotations

import torch

from .hooks import Capture, Inject, decoder_layers
from .write import feedback_write


class ReFluxSync:
    def __init__(self, model, source: int, target: int,
                 alpha: float = 0.2, beta: float = 0.9,
                 preserve_norm: bool = True):
        assert source > target, "source depth must exceed target depth"
        self.model = model
        self.source = source
        self.target = target
        self.alpha = alpha
        self.beta = beta
        self.preserve_norm = preserve_norm
        self.payloads: torch.Tensor | None = None

        layers = decoder_layers(model)
        self._cap_s = Capture(layers[source])
        self._cap_d = Capture(layers[target])
        self._inj = Inject(layers[target], self._write)

    def _write(self, h: torch.Tensor) -> torch.Tensor:
        if self.payloads is None:
            return h
        p = self.payloads
        if p.shape[1] != h.shape[1]:
            p = p[:, -h.shape[1]:, :]
        return feedback_write(h, p, self.alpha, self.beta, self.preserve_norm)

    @torch.no_grad()
    def forward(self, ids: torch.Tensor):
        """Returns logits for the input ids under same-token increment feedback."""
        # record pass
        self._inj.active = False
        self._cap_s.value = None
        self._cap_d.value = None
        self.model(ids, use_cache=False)
        self.payloads = self._cap_s.value - self._cap_d.value
        # inject pass
        self._inj.active = True
        try:
            out = self.model(ids, use_cache=False)
        finally:
            self._inj.active = False
            self.payloads = None
        return out.logits

    def remove(self):
        self._cap_s.remove()
        self._cap_d.remove()
        self._inj.remove()
