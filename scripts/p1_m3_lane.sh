#!/usr/bin/env bash
# P1-M3 parallel lane runner (R2 re-decision 2026-07-04: human-approved
# supersession of the serial S1 constraint — see STEP_LOG and issue #12).
#
# One lane = a serial list of run tags. Multiple lanes run as separate
# processes in parallel. Completed runs (metrics JSON with transitions >=
# budget and no halt) are skipped, so restarting a lane never duplicates a
# finished run. A training-level NaN halt (exit=2) stops only this lane
# (global rule 6 — factual report before continuing).
#
# Usage: scripts/p1_m3_lane.sh <lane-name> <tag> [<tag> ...]
#   tag format: t1{a|b|c}_s{0|1|2}
set -u
cd "$(dirname "$0")/.."

lane="$1"
shift

# Cap CPU threads per lane to avoid oversubscription across parallel lanes.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

for tag in "$@"; do
  task="${tag:2:1}"   # t1a_s0 -> a
  seed="${tag: -1}"   # t1a_s0 -> 0
  if [ -f "outputs/metrics/p1_run_${tag}.json" ] && \
     python3 -c "import json,sys; d=json.load(open('outputs/metrics/p1_run_${tag}.json')); sys.exit(0 if d['transitions']>=d['total_budget'] and d['halt_reason'] is None else 1)" 2>/dev/null; then
    echo "[${lane}] ${tag} already complete — skip"
    continue
  fi
  echo "[${lane}] start ${tag} $(date -u +%FT%TZ)"
  uv run python scripts/p1_train.py \
    --config "configs/p1_t1_${task}.yaml" \
    --seed "${seed}" \
    --run-tag "${tag}"
  status=$?
  echo "[${lane}] done ${tag} exit=${status} $(date -u +%FT%TZ)"
  if [ "${status}" -eq 2 ]; then
    echo "[${lane}] TRAINING NaN HALT on ${tag} — stopping lane for factual report (rule 6)"
    exit 2
  fi
done
echo "[${lane}] lane complete $(date -u +%FT%TZ)"
