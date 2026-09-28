#!/usr/bin/env python
"""Perplexity evaluation for base / ReFlux-streaming / ReFlux-sync / Recirculation-v1.

Example:
    python scripts/eval_ppl.py --model google/gemma-3-1b-pt \
        --method reflux-streaming --source 11 --target 4 --alpha 0.2 \
        --corpus data/corpus_c4.txt --windows 64 --output results/c4.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402

from reflux import (ReFluxStreaming, ReFluxSync, RecirculationV1,  # noqa: E402
                        build_windows, load_model, window_ppl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--method", default="base",
                    choices=["base", "reflux-streaming", "reflux-sync",
                             "recirculation-v1"])
    ap.add_argument("--source", type=int, default=None)
    ap.add_argument("--target", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=0.15)
    ap.add_argument("--beta", type=float, default=0.9,
                    help="target retention for reflux-streaming")
    ap.add_argument("--corpus", required=True, help="path to a plain-text file")
    ap.add_argument("--windows", type=int, default=64)
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--bos", default="none", choices=["none", "auto"],
                    help="'auto' prefixes each window with the tokenizer's "
                         "BOS when it defines one")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    tok, model = load_model(args.model, args.device)
    text = open(args.corpus).read()
    ids = build_windows(tok, text, args.windows, args.window, bos=args.bos)

    if args.method == "base":
        ppl = window_ppl(model, ids, args.windows, args.window,
                         batch_size=args.batch_size, device=args.device)
    else:
        assert args.source is not None and args.target is not None, \
            "--source/--target required for feedback methods"
        cls = (ReFluxStreaming if args.method == "reflux-streaming"
               else ReFluxSync if args.method == "reflux-sync"
               else RecirculationV1)
        method = cls(model, args.source, args.target, alpha=args.alpha,
                     beta=args.beta) \
            if args.method != "recirculation-v1" \
            else cls(model, args.source, args.target, alpha=args.alpha)
        ppl = window_ppl(model, ids, args.windows, args.window,
                         forward_fn=method.forward,
                         batch_size=args.batch_size, device=args.device)
        method.remove()

    result = {"model": args.model, "method": args.method,
              "corpus": args.corpus, "windows": args.windows,
              "window": args.window, "bos": args.bos,
              "source": args.source, "target": args.target,
              "alpha": args.alpha, "beta": args.beta, "ppl": round(ppl, 3)}
    print(json.dumps(result, indent=1))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
