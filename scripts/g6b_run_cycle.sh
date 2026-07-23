#!/bin/bash
# G6b single-run cycle: train -> guard-on val-50 reselect -> one-shot heldout.
# Usage: g6b_run_cycle.sh <arm> <seed>
# Canonical paths per G-EV convention; lock mandatory for non-BB heldout.
set -u
arm="$1"; seed="$2"
cd /home/simx2204/Workspaces/DGCC
tag="sprint_t2_${arm}_s${seed}"
cfg="configs/sprint_t2_${arm}.yaml"
log="outputs/reports/p1_sprint_train_${tag}.log"

echo "CYCLE_START arm=${arm} seed=${seed} $(date -u +%FT%TZ)"
# Resume guard: a completed, non-halted training run is never re-trained (protects 11h+ runs
# from post-train tooling failures; the per-tag gate below still re-verifies it).
if uv run python -c "
import json,sys
try: r=json.load(open('outputs/metrics/p1_run_${tag}.json'))
except Exception: sys.exit(1)
sys.exit(0 if (r.get('transitions')==300032 and r.get('halt_reason') is None) else 1)
" 2>/dev/null; then
  echo "TRAIN_SKIP_COMPLETE arm=${arm} seed=${seed}"
else
  uv run python scripts/p1_sprint_train.py --config "${cfg}" --arm "${arm}" --seed "${seed}" --run-tag "${tag}" > "${log}" 2>&1
  rc=$?
  echo "TRAIN_END arm=${arm} seed=${seed} rc=${rc} $(date -u +%FT%TZ)"
  if [ $rc -ne 0 ]; then echo "CYCLE_FAIL stage=train arm=${arm} seed=${seed}"; exit 1; fi
fi

# Per-tag gate (fail-closed): budget 300,032 · halt None · run-complete log line · init hash recorded.
uv run python - "$tag" <<'PYEOF'
import json, sys
from pathlib import Path
tag = sys.argv[1]
r = json.load(open(f"outputs/metrics/p1_run_{tag}.json"))
log = Path(f"outputs/reports/p1_sprint_train_{tag}.log").read_text(errors="replace")
problems = []
if r.get("transitions") != 300032: problems.append(f"budget {r.get('transitions')}")
if r.get("halt_reason") is not None: problems.append(f"halt {r.get('halt_reason')}")
if "run complete" not in log: problems.append("no run-complete line")
if not str(r.get("initial_weights_sha256", "")): problems.append("no F-a init hash")
if problems:
    print("GATE_FAIL", tag, ";".join(problems)); sys.exit(1)
print("GATE_PASS", tag, "init", r["initial_weights_sha256"][:8])
PYEOF
rc=$?
echo "GATE_END arm=${arm} seed=${seed} rc=${rc}"
if [ $rc -ne 0 ]; then echo "CYCLE_FAIL stage=gate arm=${arm} seed=${seed}"; exit 1; fi

sel="outputs/metrics/sprint_sel_t2_${arm}_s${seed}.json"
uv run python scripts/sprint_select_ckpt.py --run-tag "${tag}" --arm "${arm}" --seed "${seed}" --config "${cfg}" --selection-out "${sel}" >> "${log}" 2>&1
rc=$?
echo "SELECT_END arm=${arm} seed=${seed} rc=${rc} $(date -u +%FT%TZ)"
if [ $rc -ne 0 ]; then echo "CYCLE_FAIL stage=select arm=${arm} seed=${seed}"; exit 1; fi

uv run python scripts/sprint_heldout_eval.py \
  --run-tag "${tag}" --arm "${arm}" --seed "${seed}" --config "${cfg}" \
  --lock /home/simx2204/Workspaces/DGCC/outputs/metrics/sprint_metric.lock \
  --selection-manifest "/home/simx2204/Workspaces/DGCC/${sel}" \
  --claim "/home/simx2204/Workspaces/DGCC/outputs/metrics/p1_${arm}_sprint_heldout_${tag}_claim.json" \
  --out "/home/simx2204/Workspaces/DGCC/outputs/metrics/p1_${arm}_sprint_heldout_${tag}.json" >> "${log}" 2>&1
rc=$?
echo "HELDOUT_END arm=${arm} seed=${seed} rc=${rc} $(date -u +%FT%TZ)"
if [ $rc -ne 0 ]; then echo "CYCLE_FAIL stage=heldout arm=${arm} seed=${seed}"; exit 1; fi
echo "CYCLE_COMPLETE arm=${arm} seed=${seed}"
