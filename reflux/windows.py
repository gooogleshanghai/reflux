"""Window construction for language-modeling evaluation.

Default protocol (aligned with the v1 release of Recirculation): documents
are tokenized without special tokens and partitioned into fixed-length
windows; short corpora are cycled to fill the requested number of windows.
An option prefixes each window with the tokenizer's BOS token when one
exists (per-tokenizer convention).
"""
from __future__ import annotations

import math

import torch


def build_windows(tok, text: str, num_windows: int, window: int = 1024,
                  bos: str = "none") -> torch.Tensor:
    """Returns a flat 1-D tensor of ``num_windows * window`` token ids."""
    ids = tok(text, return_tensors="pt",
              add_special_tokens=False).input_ids[0]
    n_content = num_windows * (window - 1 if bos == "auto" else window)
    if len(ids) < n_content:
        reps = math.ceil(n_content / max(len(ids), 1))
        ids = ids.repeat(reps)[:n_content]
    else:
        ids = ids[:n_content]
    if bos == "auto" and tok.bos_token_id is not None:
        rows = ids.reshape(num_windows, window - 1)
        bos_col = torch.full((num_windows, 1), tok.bos_token_id,
                             dtype=rows.dtype)
        return torch.cat([bos_col, rows], dim=1).reshape(-1)
    return ids
