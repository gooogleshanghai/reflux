#!/usr/bin/env python
"""Dump a plain-text corpus file for perplexity evaluation.

Example:
    python scripts/prepare_corpus.py --dataset allenai/c4 --subset en \
        --split validation --n_chars 3000000 --output data/corpus_c4.txt
"""
import argparse

from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--subset", default=None)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--text-column", default="text")
    ap.add_argument("--n_chars", type=int, default=3_000_000,
                    help="enough characters for ~64 windows of 1024 tokens")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    ds = load_dataset(args.dataset, args.subset, split=args.split,
                      streaming=True)
    buf = []
    total = 0
    for ex in ds:
        t = ex[args.text_column]
        buf.append(t)
        total += len(t)
        if total >= args.n_chars:
            break
    text = "\n".join(buf)
    with open(args.output, "w") as f:
        f.write(text)
    print(f"wrote {len(text)} chars to {args.output}")


if __name__ == "__main__":
    main()
