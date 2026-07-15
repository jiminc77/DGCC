#!/usr/bin/env bash
# P1-M4 lane/supervisor (gate verdict gate-m3r-reconvene-2-20260713 choice B;
# consensus plan pending-approval sha256 78292468..., P4/P5 + stage-5 PGID
# amendment). ONE shared implementation for the smoke drill and the main
# lanes: launch -> readiness-verified PGID handoff (atomic durable
# publication) -> optional flush-confirmed halfway group-STOP with durable
# quiescent marker and no-advance hold.
#
# Modes:
#   drill/smoke:  p1_m4_lane.sh --drill --tag <tag> --config <yaml> --seed <n> \
#                     --total <n> --halfway <n>
#                 (exports P1_LOG_EVERY_ROUND=1; CONTs after the hold and
#                  waits for trainer completion)
#   main:         p1_m4_lane.sh <tag> --config <yaml> --seed <n> [--halfway <n>]
#                 (strips P1_LOG_EVERY_ROUND; with --halfway leaves the group
#                  STOPPED for the leader's HER decision and exits 0)
#
# Exit codes: trainer exit propagated (2 = TrainingNaNError, rule 6);
#   70 = pre-launch refusal (flag set on main / lock held);
#   71 = readiness failure (PID/PGID/SID/cmd identity not established);
#   72 = STOP verification failure (member not T / advance during hold).
set -u
cd "$(dirname "$0")/.."

OPS=/tmp/dgcc_ops
mkdir -p "$OPS" outputs/archive/m4ops

# ---------- argument parsing ------------------------------------------------
DRILL=0 TAG="" CONFIG="" SEED="" TOTAL="" HALFWAY=""
if [ "${1:-}" = "--drill" ]; then DRILL=1; shift; else TAG="${1:?tag required}"; shift; fi
while [ "$#" -gt 0 ]; do
  case "$1" in
    --tag)     TAG="$2"; shift 2 ;;
    --config)  CONFIG="$2"; shift 2 ;;
    --seed)    SEED="$2"; shift 2 ;;
    --total)   TOTAL="$2"; shift 2 ;;
    --halfway) HALFWAY="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 64 ;;
  esac
done
: "${TAG:?--tag required}" "${CONFIG:?--config required}" "${SEED:?--seed required}"
if [ "$DRILL" = 1 ]; then : "${TOTAL:?--total required in drill mode}" "${HALFWAY:?--halfway required in drill mode}"; fi

RUN_JSON="outputs/metrics/p1_run_${TAG}.json"
TRAIN_LOG="outputs/reports/p1_train_${TAG}.log"

# ---------- pre-launch assertions + shared GPU lock --------------------------
if [ "$DRILL" = 0 ] && [ -n "${P1_LOG_EVERY_ROUND:-}" ]; then
  echo 'FLAG SET — refuse main launch'; exit 70   # pre-launch assertion (plan P5)
fi
exec 9>"$OPS/m4_gpu.lock"
flock -s 9 || { echo "GPU lock unavailable"; exit 70; }

# ---------- (1) launch: $! is the setsid session leader ----------------------
launch_args=(uv run python scripts/p1_train.py --config "$CONFIG" --seed "$SEED" --run-tag "$TAG")
# Identical launch shape in both modes ($! IS the setsid session leader); the
# drill differs only in the env flag and --total-override. Evidence mirroring
# for the smoke lives at the tmux layer (pipe-pane), never inside this script.
if [ "$DRILL" = 1 ]; then
  launch_args+=(--total-override "$TOTAL")
  P1_LOG_EVERY_ROUND=1 setsid "${launch_args[@]}" &
else
  env -u P1_LOG_EVERY_ROUND setsid "${launch_args[@]}" &
fi
PID=$!
LANEPIPE=$PID

# ---------- (2) bounded readiness loop (<=10 s, 0.5 s interval) --------------
ready=""
for i in $(seq 1 20); do
  line=$(ps -o pgid=,sid=,cmd= -p "$PID" 2>/dev/null) || { sleep 0.5; continue; }
  pgid=$(echo "$line" | awk '{print $1}'); sid=$(echo "$line" | awk '{print $2}')
  case "$line" in *scripts/p1_train.py*"$TAG"*) cmd_ok=1 ;; *) cmd_ok=0 ;; esac
  if [ "$pgid" = "$PID" ] && [ "$sid" = "$PID" ] && [ "$cmd_ok" = 1 ]; then ready=1; break; fi
  sleep 0.5
done
if [ -z "$ready" ]; then
  echo "LAUNCH FAIL: PID/PGID/SID/cmd identity not established for ${TAG}"
  kill -TERM "$PID" 2>/dev/null
  exit 71
fi

# ---------- (3) atomic durable publication of BOTH handoff files -------------
publish() {  # publish <path> <value>
  tmp="$1.tmp.$$"; printf '%s\n' "$2" > "$tmp"
  python3 - "$tmp" "$1" <<'PY'
import os, sys
tmp, dst = sys.argv[1], sys.argv[2]
fd = os.open(tmp, os.O_RDONLY); os.fsync(fd); os.close(fd)
os.rename(tmp, dst)
dfd = os.open(os.path.dirname(dst) or '.', os.O_RDONLY); os.fsync(dfd); os.close(dfd)
PY
}
publish "$OPS/${TAG}.pid"  "$PID"
publish "$OPS/${TAG}.pgid" "$PID"   # leader: PID == PGID == SID (asserted above)
echo "[lane] ${TAG} launched pid/pgid=${PID} $(date -u +%FT%TZ)"

wait_trainer() {  # wait for the launched pipeline/leader; propagate exit code
  wait "$LANEPIPE"; rc=$?
  if [ "$rc" = 2 ]; then echo "[lane] ${TAG} TrainingNaNError halt (rule 6)"; fi
  echo "[lane] ${TAG} trainer exit=${rc} $(date -u +%FT%TZ)"
  return "$rc"
}

# ---------- no halfway: plain supervised run ---------------------------------
if [ -z "$HALFWAY" ]; then
  wait_trainer; exit $?
fi

# ---------- halfway supervisor -----------------------------------------------
PGID=$(cat "$OPS/${TAG}.pgid")

assert_group_identity() {  # full stage-5 re-assertion — run before ANY signal
  line=$(ps -o pgid=,sid=,cmd= -p "$PGID" 2>/dev/null) || return 1
  [ "$(echo "$line" | awk '{print $1}')" = "$PGID" ] || return 1
  [ "$(echo "$line" | awk '{print $2}')" = "$PGID" ] || return 1
  case "$line" in *scripts/p1_train.py*"--run-tag ${TAG}"*) : ;; *)
    # leader may be the uv wrapper whose argv still names the trainer+tag;
    # otherwise require a group member carrying it
    ps -o cmd= -g "$PGID" 2>/dev/null | grep -q "p1_train.py .*--run-tag ${TAG}" || return 1 ;;
  esac
  [ -n "$(ps -o pid= -g "$PGID" 2>/dev/null)" ] || return 1   # non-empty group enumeration
  return 0
}
assert_group_identity || { echo "identity mismatch pre-poll"; wait_trainer 2>/dev/null; exit 71; }

# (a) flush-confirmed STOP trigger: run-JSON poll ONLY (never the eval print).
echo "[lane] ${TAG} halfway supervisor armed at ${HALFWAY} (poll ${RUN_JSON})"
while :; do
  if ! kill -0 "$PGID" 2>/dev/null; then
    echo "[lane] ${TAG} trainer ended before halfway trigger"
    wait_trainer; exit $?
  fi
  trig=$(python3 - "$RUN_JSON" "$HALFWAY" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    evals = d.get("evals") or []
    print("FIRE" if evals and evals[-1]["transitions"] >= int(sys.argv[2]) else "WAIT")
except (FileNotFoundError, json.JSONDecodeError, KeyError, IndexError, TypeError):
    print("WAIT")   # explicitly continue polling (plan step 1)
PY
)
  [ "$trig" = "FIRE" ] && break
  sleep 5
done

# (b) group STOP + bounded all-member-T verification (single-PID STOP forbidden)
assert_group_identity || { echo "identity mismatch pre-STOP"; wait_trainer 2>/dev/null; exit 71; }
kill -STOP -- "-$PGID"
stopped=""
for i in $(seq 1 20); do
  stats=$(ps -o stat= -g "$PGID" 2>/dev/null | tr -d ' ')
  if [ -z "$stats" ]; then break; fi
  if ! echo "$stats" | grep -qv '^T'; then stopped=1; break; fi
  sleep 0.5
done
if [ -z "$stopped" ]; then
  echo "STOP VERIFY FAIL: group members not all T within 10s — CONT + fail closed"
  kill -CONT -- "-$PGID" 2>/dev/null
  exit 72
fi

# (c) durable quiescent marker (file fsync + dir fsync), mirrored to archive
MARKER="$OPS/m4_half_s${SEED}.marker"
python3 - "$MARKER" "$TAG" "$PGID" "$RUN_JSON" "$TRAIN_LOG" <<'PY'
import hashlib, json, os, sys
from datetime import datetime, timezone
marker, tag, pgid, run_json, train_log = sys.argv[1:6]
payload = {
    "tag": tag,
    "pgid": int(pgid),
    "stopped_at": datetime.now(timezone.utc).isoformat(),
    "evals_len": len(json.load(open(run_json)).get("evals") or []),
    "run_json_sha256": hashlib.sha256(open(run_json, "rb").read()).hexdigest(),
    "log_size": os.path.getsize(train_log),
}
def durable_write(path):
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1); fh.write("\n"); fh.flush(); os.fsync(fh.fileno())
    dfd = os.open(os.path.dirname(path) or ".", os.O_RDONLY); os.fsync(dfd); os.close(dfd)
durable_write(marker)
durable_write(os.path.join("outputs/archive/m4ops", os.path.basename(marker)))
print(f"HALF_STOPPED marker durable: {marker}")
PY

# (d) 60 s no-advance hold: all members T; log size AND run-JSON sha256
#     unchanged vs marker (plan step 4)
LOG_SIZE=$(python3 -c "import json;print(json.load(open('$MARKER'))['log_size'])")
JSON_SHA=$(python3 -c "import json;print(json.load(open('$MARKER'))['run_json_sha256'])")
sleep 60
stats=$(ps -o stat= -g "$PGID" 2>/dev/null | tr -d ' ')
now_size=$(stat -c %s "$TRAIN_LOG")
now_sha=$(sha256sum "$RUN_JSON" | awk '{print $1}')
if echo "$stats" | grep -qv '^T' || [ "$now_size" != "$LOG_SIZE" ] || [ "$now_sha" != "$JSON_SHA" ]; then
  echo "NO-ADVANCE FAIL: stats=[$stats] log ${LOG_SIZE}->${now_size} json-sha changed=$([ "$now_sha" != "$JSON_SHA" ] && echo yes || echo no) — CONT + fail closed"
  kill -CONT -- "-$PGID" 2>/dev/null
  exit 72
fi
echo "[lane] ${TAG} HALF_STOPPED verified (no-advance 60s) marker=${MARKER}"

if [ "$DRILL" = 1 ]; then
  # drill: resume and run to completion through the same code path
  kill -CONT -- "-$PGID"
  echo "[lane] ${TAG} drill CONT issued $(date -u +%FT%TZ)"
  wait_trainer; exit $?
fi
# main: leave the group STOPPED; the leader performs the HER decision and CONT.
echo "[lane] ${TAG} left STOPPED for HER decision: ${MARKER}"
exit 0
