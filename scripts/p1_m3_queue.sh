#!/usr/bin/env bash
# P1-M3 serial run queue (S1 — one run at a time, R2 concurrency rejected).
# 9 runs: {t1a, t1b, t1c} x seeds {0, 1, 2} x 1e5 transitions each.
set -u
cd "$(dirname "$0")/.."

for task in a b c; do
  for seed in 0 1 2; do
    tag="t1${task}_s${seed}"
    if [ -f "outputs/metrics/p1_run_${tag}.json" ] && \
       python3 -c "import json,sys; d=json.load(open('outputs/metrics/p1_run_${tag}.json')); sys.exit(0 if d['transitions']>=d['total_budget'] and d['halt_reason'] is None else 1)" 2>/dev/null; then
      echo "[queue] ${tag} already complete — skip"
      continue
    fi
    echo "[queue] start ${tag} $(date -u +%FT%TZ)"
    uv run python scripts/p1_train.py \
      --config "configs/p1_t1_${task}.yaml" \
      --seed "${seed}" \
      --run-tag "${tag}"
    status=$?
    echo "[queue] done ${tag} exit=${status} $(date -u +%FT%TZ)"
    if [ "${status}" -eq 2 ]; then
      echo "[queue] TRAINING NaN HALT on ${tag} — stopping queue for factual report (rule 6)"
      exit 2
    fi
  done
done
echo "[queue] all 9 runs complete $(date -u +%FT%TZ)"
