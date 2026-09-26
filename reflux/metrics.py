"""Batched perplexity over fixed windows."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def window_ppl(model, ids_all: torch.Tensor, num_windows: int,
               window: int = 1024, forward_fn=None, batch_size: int = 2,
               device: str = "cuda:0") -> float:
    """Mean NLL over all (window-1) next-token predictions, in PPL.

    ``forward_fn(ids) -> logits`` overrides the plain model forward (used by
    the feedback methods); targets are the input ids shifted by one.
    """
    forward_fn = forward_fn or (lambda x: model(x, use_cache=False).logits)
    nll_sum, ntok = 0.0, 0
    for i in range(0, num_windows, batch_size):
        ids = ids_all[i * window:(i + batch_size) * window] \
            .reshape(batch_size, window).to(device)
        logits = forward_fn(ids)
        lg = logits[:, :-1].reshape(-1, logits.size(-1))
        tgt = ids[:, 1:].reshape(-1)
        for c in range(0, lg.shape[0], 2048):
            nll_sum += F.cross_entropy(
                lg[c:c + 2048].float(), tgt[c:c + 2048],
                reduction="sum").item()
        ntok += tgt.numel()
    return math.exp(nll_sum / ntok)
