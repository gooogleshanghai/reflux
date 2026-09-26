"""Recirculation (v1) reproduction: full-state deep-to-shallow feedback.

Faithful to the released v1 protocol of Mozer et al. (2026): the full deep
residual state z_s is written back to the shallower stream with the convex,
norm-rescaled mixture

    z_d <- beta * z_d + alpha * N_{z_d}(z_s),   beta = 1 - alpha.

Two batched traversals: a record pass captures z_s at every position; an
inject pass writes the same-position state into the target layer input.
"""
from __future__ import annotations

import torch

from .hooks import Capture, Inject, decoder_layers
from .write import feedback_write


class RecirculationV1:
    def __init__(self, model, source: int, target: int, alpha: float = 0.15):
        assert source > target
        self.model = model
        self.alpha = alpha
        self.payloads: torch.Tensor | None = None

        layers = decoder_layers(model)
        self._cap_s = Capture(layers[source])
        self._inj = Inject(layers[target], self._write)

    def _write(self, h: torch.Tensor) -> torch.Tensor:
        if self.payloads is None:
            return h
        p = self.payloads
        if p.shape[1] != h.shape[1]:
            p = p[:, -h.shape[1]:, :]
        return feedback_write(h, p, self.alpha,
                              beta=1.0 - self.alpha, preserve_norm=False)

    @torch.no_grad()
    def forward(self, ids: torch.Tensor):
        self._inj.active = False
        self._cap_s.value = None
        out = self.model(ids, use_cache=False)
        self.payloads = self._cap_s.value
        self._inj.active = True
        try:
            out = self.model(ids, use_cache=False)
        finally:
            self._inj.active = False
            self.payloads = None
        return out.logits

    def remove(self):
        self._cap_s.remove()
        self._inj.remove()
