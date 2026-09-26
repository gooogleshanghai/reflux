"""Model loading helpers."""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(path: str, device: str = "cuda:0", dtype=torch.bfloat16):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=dtype, device_map=device).eval()
    return tok, model
