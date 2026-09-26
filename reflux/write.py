"""The feedback write operator.

Effective update (ReFlux-streaming):

    h <- rescale( beta * h + alpha * N_h(p) ),   N_h(p) = p * (|h| / |p|)

where ``rescale`` restores the pre-write vector norm.  Recirculation-v1 uses
the same operator with the full source state as payload, the convex mixture
beta = 1 - alpha, and no rescaling.
"""
from __future__ import annotations

import torch


def feedback_write(h: torch.Tensor,
                   payload: torch.Tensor,
                   alpha: float,
                   beta: float,
                   preserve_norm: bool = False) -> torch.Tensor:
    if alpha <= 0:
        return h
    with torch.autocast("cuda", enabled=False):
        hf = h.float()
        pf = payload.float()
        matched = pf * (hf.norm(dim=-1, keepdim=True)
                        / (pf.norm(dim=-1, keepdim=True) + 1e-6))
        delta = alpha * matched + (beta - 1.0) * hf
        h0_norm = None
        if preserve_norm:
            h0_norm = hf.norm(dim=-1, keepdim=True)
    h = h + delta.to(h.dtype)
    if preserve_norm:
        with torch.autocast("cuda", enabled=False):
            hf = h.float()
            h = (hf * (h0_norm /
                       hf.norm(dim=-1, keepdim=True).clamp_min(1e-6))).to(h.dtype)
    return h
