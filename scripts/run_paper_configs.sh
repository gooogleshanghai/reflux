#!/usr/bin/env bash
# Run all per-model configurations on one corpus (base + both methods).
# Usage: CORPUS=data/corpus_c4.txt WINDOWS=64 ./scripts/run_paper_configs.sh
set -e
CORPUS=${CORPUS:?set CORPUS=/path/to/corpus.txt}
WINDOWS=${WINDOWS:-64}
DEV=${DEVICE:-cuda:0}

for cfg in configs/*.json; do
  model=$(python -c "import json;print(json.load(open('$cfg'))['model'])")
  v1=$(python - "$cfg" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))["recirculation_v1"]
print(f"--source {c['source']} --target {c['target']} --alpha {c['alpha']}")
PY
)
  rx=$(python - "$cfg" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))["reflux_streaming"]
print(f"--source {c['source']} --target {c['target']} --alpha {c['alpha']} --beta {c['beta']}")
PY
)
  name=$(basename "$cfg" .json)
  echo "=== $name (base) ==="
  python scripts/eval_ppl.py --model "$model" --method base \
    --corpus "$CORPUS" --windows "$WINDOWS" --device "$DEV" \
    --output "results/sample/${name}_base.json"
  echo "=== $name (recirculation-v1) ==="
  python scripts/eval_ppl.py --model "$model" --method recirculation-v1 \
    $v1 --corpus "$CORPUS" --windows "$WINDOWS" --device "$DEV" \
    --output "results/sample/${name}_recirculation_v1.json"
  echo "=== $name (reflux-streaming) ==="
  python scripts/eval_ppl.py --model "$model" --method reflux-streaming \
    $rx --corpus "$CORPUS" --windows "$WINDOWS" --device "$DEV" \
    --output "results/sample/${name}_reflux_streaming.json"
done
