#!/bin/bash
# G6b serial 17-run campaign: V1 {0,1,2,3,4,6,7} -> matched {0..4} -> random {0..4}.
# Crash-class auto-retry per adopted pocket-guard/crash-disposition rules:
#   TRAIN-stage failure -> archive crash artifacts (MANIFEST.sha256) -> same-seed relaunch, max 3 attempts total.
#   GATE/SELECT/HELDOUT-stage failure -> hard stop (not crash-class; needs disposition).
set -u
cd /home/simx2204/Workspaces/DGCC
RUNS="v1:0 v1:1 v1:2 v1:3 v1:4 v1:6 v1:7 matched:0 matched:1 matched:2 matched:3 matched:4 random:0 random:1 random:2 random:3 random:4"
for spec in $RUNS; do
  arm="${spec%%:*}"; seed="${spec##*:}"
  tag="sprint_t2_${arm}_s${seed}"
  if [ -f "outputs/metrics/p1_${arm}_sprint_heldout_${tag}.json" ]; then
    echo "SKIP_DONE ${tag}"; continue
  fi
  attempt=0
  while [ $attempt -lt 3 ]; do
    attempt=$((attempt+1))
    echo "RUN_ATTEMPT ${tag} attempt=${attempt} $(date -u +%FT%TZ)"
    df -BG --output=avail / | tail -1 | tr -d ' G' | awk '{print "DISK_AVAIL_G="$1; if ($1<20) print "DISK_SOFT_ALERT"; if ($1<5) print "DISK_HARD_ALERT"}'
    bash scripts/g6b_run_cycle.sh "$arm" "$seed"
    rc=$?
    if [ $rc -eq 0 ]; then break; fi
    if grep -q "CYCLE_FAIL stage=train" <<< "$(tail -5 /dev/null)"; then :; fi
    # Determine failed stage from last cycle output line semantics: rely on marker files.
    if [ ! -f "outputs/metrics/p1_run_${tag}.json" ] || ! uv run python -c "
import json,sys
r=json.load(open('outputs/metrics/p1_run_${tag}.json'))
sys.exit(0 if (r.get('transitions')==300032 and r.get('halt_reason') is None) else 1)
" 2>/dev/null; then
      # train-stage crash: archive and retry
      TS=$(date -u +%Y%m%dT%H%MZ); AR="outputs/archive/sprint_crash/${tag}-auto-${TS}"
      mkdir -p "$AR"
      mv "outputs/models/${tag}" "$AR/models" 2>/dev/null
      mv "outputs/metrics/p1_run_${tag}.json" "outputs/metrics/p1_diag_${tag}.json" "$AR/" 2>/dev/null
      cp "outputs/reports/p1_sprint_train_${tag}.log" "$AR/" 2>/dev/null
      ( cd "$AR" && find . -type f -exec sha256sum {} \; > MANIFEST.sha256 )
      echo "CRASH_ARCHIVED ${tag} attempt=${attempt} dir=${AR}"
      if [ $attempt -ge 3 ]; then echo "RETRY_LIMIT ${tag} — soft gate required"; exit 2; fi
    else
      echo "POST_TRAIN_FAIL ${tag} — hard stop (not crash-class)"; exit 3
    fi
  done
  echo "RUN_DONE ${tag} $(date -u +%FT%TZ)"
done
echo "CAMPAIGN_COMPLETE $(date -u +%FT%TZ)"
