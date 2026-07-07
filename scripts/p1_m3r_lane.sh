#!/usr/bin/env bash
# P1-M3R parallel lane runner (M3R + 2-lane directive): each lane owns a
# serial list of m3r_ run tags while peer lanes run independently.
#
# Completed runs (metrics JSON with transitions >= budget and no halt) are
# skipped, so restarting a lane never duplicates a finished run. A
# training-level NaN halt (exit=2) stops only this lane (global rule 6 —
# factual report before continuing).
#
# Usage: scripts/p1_m3r_lane.sh <lane-name> <tag> [<tag> ...]
#   tag format: m3r_t1{a|b|c}_s{0|1|2}
set -u
cd "$(dirname "$0")/.."

if [ "$#" -lt 2 ]; then
  echo "usage: scripts/p1_m3r_lane.sh <lane-name> <m3r_t1{a|b|c}_s{0|1|2}> [...]" >&2
  exit 64
fi

lane="$1"
shift

# Cap CPU threads per lane to avoid oversubscription across parallel lanes.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

for tag in "$@"; do
  case "${tag}" in
    m3r_t1[abc]_s[012]) ;;
    *)
      echo "[${lane}] invalid tag ${tag}; expected m3r_t1{a|b|c}_s{0|1|2}" >&2
      exit 64
      ;;
  esac

  task="${tag:6:1}"  # m3r_t1a_s0 -> a
  seed="${tag: -1}"  # m3r_t1a_s0 -> 0
  metrics="outputs/metrics/p1_run_${tag}.json"

  if [ -f "${metrics}" ] && \
     python3 -c "import json,sys; d=json.load(open('${metrics}')); sys.exit(0 if d['transitions']>=d['total_budget'] and d['halt_reason'] is None else 1)" 2>/dev/null; then
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
  if [ "${status}" -ne 0 ]; then
    # Non-halt crash (e.g. rebuild-limit escalation): document and continue
    # with the lane's remaining runs (M3 lane semantics — only exit=2 stops a
    # lane). Crashed-run disposition is a gate matter; the leader archives the
    # incomplete artifacts so the skip-check cannot silently re-run the seed.
    echo "[${lane}] ${tag} crashed (exit=${status}) — documented; continuing lane"
  fi
done

echo "[${lane}] lane complete $(date -u +%FT%TZ)"
