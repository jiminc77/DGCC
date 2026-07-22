#!/usr/bin/env python3
"""Generate the G6a BB closeout from immutable metrics, logs, archives, and STEP_LOG."""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path
from statistics import mean

from sprint_stability_recount import recount_log

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def load(path: Path):
    return json.loads(path.read_text())


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def val(row: dict) -> dict:
    evals = row["evals"]
    final = evals[-1]
    return {
        "transitions": row["transitions"],
        "final_val_success": final.get("success_rate"), "final_val_return": final.get("mean_return"),
        "eval_walls_s": [e.get("wall_s") for e in evals],
    }


def main() -> None:
    current_logs = sorted((OUT / "reports").glob("p1_sprint_train_sprint_t2_bb_s[3467].log"))
    archive_logs = sorted((OUT / "archive" / "sprint_crash").glob("*/p1_sprint_train_sprint_t2_bb_s*.log.*"))
    logs = current_logs + archive_logs
    stability = []
    for path in logs:
        r = recount_log(path)
        run_path = next(iter(path.parent.glob("p1_run_*.json*")), None)
        if run_path is None:
            candidate = OUT / "metrics" / f"p1_run_{r['run_tag']}.json"
            run_path = candidate if candidate.is_file() else None
        run = load(run_path) if run_path else {}
        status = "complete" if path in current_logs else "archived"
        stability.append({
            "log": rel(path), "status": status, "run_tag": r["run_tag"],
            "reported": r["reported"], "recounted_lower_bound": r["recounted_lower_bound"],
            "rebuilds": r["rebuilds"], "reset_boundaries": r["reset_boundaries"],
            "initial_weights_sha256": run.get("initial_weights_sha256"),
            "transitions": run.get("transitions"), "halt_reason": run.get("halt_reason"),
        })
    stability.sort(key=lambda x: x["log"])
    (OUT / "metrics" / "sprint_g6a_stability.json").write_text(json.dumps({
        "schema_version": 1,
        "method": "scripts/sprint_stability_recount.py recount_log individually over current and archived immutable logs",
        "runs": stability,
    }, indent=2, sort_keys=True) + "\n")

    held = []
    for p in sorted((OUT / "metrics").glob("p1_t2_sprint_heldout_m4_t2_s[012].json")) + sorted((OUT / "metrics").glob("p1_bb_sprint_heldout_sprint_t2_bb_s[3467].json")):
        d = load(p); s = d["summary"]
        manifest = load(Path(d["selection_manifest"])) if d.get("selection_manifest") else None
        claim_path = (OUT / "metrics" / f"p1_sprint_heldout_claim_m4_t2_s{d['seed']}.json")
        claim_sha = d.get("claim_sha256") or __import__("hashlib").sha256(claim_path.read_bytes()).hexdigest()
        ckpt = manifest["selected_ckpt"] if manifest else d["ckpt"]
        held.append({"seed": d["seed"], "kind": "reuse" if d["seed"] < 3 else "new", "result": rel(p),
                     "ckpt": ckpt, "ckpt_sha256": d["ckpt_sha256"], "claim_sha256": claim_sha,
                     "success": s["success_rate"], "return": s["mean_return"]})
    runs = {d["seed"]: d for p in (OUT / "metrics").glob("p1_run_sprint_t2_bb_s[3467].json") for d in [load(p)]}
    wall_hours = {}
    for path in current_logs:
        match = re.search(r"run complete .*?wall_h=([0-9.]+)", path.read_text(errors="replace"))
        if match:
            wall_hours[int(re.search(r"_s(\d+)\.log$", path.name).group(1))] = float(match.group(1))
    runrows = [(seed, val(runs[seed])) for seed in sorted(runs)]
    init = {x["run_tag"]: x["initial_weights_sha256"] for x in stability if x["initial_weights_sha256"]}
    step = (ROOT / "STEP_LOG.md").read_text()
    # Source-anchored operational facts; regex makes absence a generation error.
    pockets = ["3.66h", "5.2h", "12h", "12.5h"]
    for token in pockets:
        if token not in step: raise RuntimeError(f"missing STEP_LOG pocket: {token}")
    attempts = ["s5×2", "s6 외부-kill", "attempt1 ENOSPC-kill"]
    for token in attempts:
        if token not in step: raise RuntimeError(f"missing STEP_LOG accounting: {token}")
    h = []
    h.append("# G6a BB 종결 리포트")
    h.append("")
    h.append("생성: `uv run python scripts/generate_sprint_g6a_report.py`. 수치는 생성 시 원천 JSON/로그에서 추출했다. BB 사실만 기록하며 unblinding 전 성능 비교·V1 추론은 하지 않는다.")
    h.append("\n## 1. Held-out (7 seed)")
    h.append("|seed|구분|선택 ckpt/sha256|claim sha256|success|mean return|원천|")
    h.append("|-:|---|---|---|--:|--:|---|")
    for x in held:
        h.append(f"|{x['seed']}|{x['kind']}|`{x['ckpt']}` / `{x['ckpt_sha256']}`|`{x['claim_sha256']}`|{x['success']:.3f}|{x['return']:.6f}|`{x['result']}`|")
    h.append("\n## 2. 학습 궤적 및 eval-wall 감시")
    h.append("|seed|transitions|final val success|final val return|wall h|eval-wall max s|<3600s 전건|")
    h.append("|-:|--:|--:|--:|--:|--:|---|")
    for seed, x in runrows:
        ws=[w for w in x['eval_walls_s'] if w is not None]
        h.append(f"|{seed}|{x['transitions']}|{x['final_val_success']:.3f}|{x['final_val_return']:.6f}|{wall_hours[seed]:.2f}|{max(ws):.3f}|{'yes' if max(ws)<3600 else 'no'}|")
    h.append("\n## 3. Stability (A-4 재집계)")
    h.append("`outputs/metrics/sprint_g6a_stability.json`은 현행 4건과 아카이브 crash/kill 전부를 `--log`와 동등한 개별 `recount_log` 호출로 재집계한다. reported는 로그 종결 행, recounted는 rebuild/reset 경계 합산 하한이다.")
    h.append("|log|상태|reported nan/mag|recount 하한 nan/mag|rebuilds|경계|")
    h.append("|---|---|---|---|--:|---|")
    for x in stability:
        r=x['reported'] or {}; q=x['recounted_lower_bound']
        h.append(f"|`{x['log']}`|{x['status']}|{r.get('nan','—')}/{r.get('mag','—')}|{q['nan']}/{q['mag']}|{x['rebuilds']}|{len(x['reset_boundaries'])}|")
    h.append("\n인프라/기술 사건 분류: ENOSPC=s7 attempt 1, reaper/job-cancel=s3 launch 및 s6 kill, rebuild-limit=s5×2 및 s7r. 이 분류는 STEP_LOG 사건 기록이며 학습 halt와 구분한다.")
    h.append("\n## 4. Settle-pocket 및 경합-상관 대조")
    h.append("|계열|포켓 지속|경합 기록|결과|")
    h.append("|---|---:|---|---|")
    h.append("|s0|3.66h|기준 전례|완주|")
    h.append("|s5|5.2h|무경합|rebuild-limit crash|")
    h.append("|s5r|12h|경합 추정|rebuild-limit crash|")
    h.append("|s7r|12.5h|경합 집중 창|rebuild-limit crash|")
    h.append("|s6/s6r|3-chain / 없음|경합 중 완주 사례|s6 kill 후 s6r 완주|")
    h.append("STEP_LOG 시각 대조에서는 포켓 지속과 경합 기록을 병기하며 인과 귀속은 하지 않는다 — 미통제 관찰이다. batch-effect는 사전등록 3-way 감도분석 대상이다.")
    h.append("\n## 5. AMD-3 및 F-a")
    h.append("AMD-3 (verdict comment 5029426419): seed 5 페어(BB+V1)를 기술 결함으로 제외하고 대체하지 않는다; BB 평균이 상향되는 방향이므로 V1−BB 델타에는 보수적 방향의 민감도 노트다.")
    h.append("|seed/attempt|initial_weights_sha256|byte-일치 증거|")
    h.append("|---|---|---|")
    for tag, sha in sorted(init.items()): h.append(f"|{tag}|`{sha}`|run JSON/아카이브 run JSON|")
    h.append("s5 original↔retry, s6 kill↔retry, s7 3 attempts의 F-a byte-일치는 STEP_LOG 사건 기록으로 교차 확인한다.")
    h.append("\n## 6. 재시도 회계")
    h.append("|seed|attempt 이력|종결|")
    h.append("|---|---|---|")
    h.append("|5|crash → retry crash|AMD-3 제외|")
    h.append("|6|job-cancel kill → fresh retry|완주|")
    h.append("|7|ENOSPC kill → rebuild-limit crash → fresh retry|완주|")
    h.append("\n## Limitations")
    h.append("포켓 진입은 확률적이며, 경합 환경은 통제 실험이 아니다. 재사용 3 seed에는 retro probe가 없어 mechanism 분석 표본에 포함하지 않는다. held-out는 위 one-shot 결과만 사용한다.")
    (OUT / "reports" / "sprint_g6a_report.md").write_text("\n".join(h)+"\n")

if __name__ == "__main__": main()
