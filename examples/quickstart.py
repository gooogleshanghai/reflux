import torch

from reflux import (ReFluxStreaming, build_windows,  # noqa: E402
                        load_model, window_ppl)

MODEL = "google/gemma-3-1b-pt"
CORPUS = "data/corpus_c4.txt"  # any plain-text file

tok, model = load_model(MODEL)
ids = build_windows(tok, open(CORPUS).read(), num_windows=8, window=1024)

base = window_ppl(model, ids, num_windows=8)

method = ReFluxStreaming(model, source=11, target=4, alpha=0.2)
reflux = window_ppl(model, ids, num_windows=8, forward_fn=method.forward)
method.remove()

print(f"base PPL: {base:.3f}")
print(f"ReFlux-streaming PPL: {reflux:.3f} ({100 * (1 - reflux / base):+.2f}%)")
