#!/usr/bin/env python3
"""Generate the artifact-only G6b campaign watch report."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sprint_stability_recount import recount_log

ROOT = Path(__file__).resolve().parents[1]
ARMS = {"v1": (0, 1, 2, 3, 4, 6, 7), "matched": (0, 1, 2, 3, 4), "random": (0, 1, 2, 3, 4)}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def _sha(path: Path | None) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path and path.is_file() else None


def _wall_hours(log: Path | None) -> float | None:
    if not log:
        return None
    match = re.search(r"run complete .*?wall_h=([0-9.]+)", log.read_text(encoding="utf-8", errors="replace"))
    return float(match.group(1)) if match else None


def _run_row(root: Path, arm: str, seed: int) -> dict[str, Any]:
    metrics, reports = root / "outputs" / "metrics", root / "outputs" / "reports"
    tag = f"sprint_t2_{arm}_s{seed}"
    run_path = metrics / f"p1_run_{tag}.json"
    log_path = reports / f"p1_sprint_train_{tag}.log"
    selection_path = metrics / f"sprint_sel_t2_{arm}_s{seed}.json"
    heldout_path = metrics / f"p1_{arm}_sprint_heldout_{tag}.json"
    claim_path = metrics / f"p1_{arm}_sprint_heldout_{tag}_claim.json"
    run = _load(run_path) if run_path.is_file() else {}
    log = log_path if log_path.is_file() else None
    recount = recount_log(log) if log else None
    gate_complete = (run.get("transitions") == 300032 and run.get("halt_reason") is None
                     and log is not None and "run complete" in log.read_text(encoding="utf-8", errors="replace"))
    complete = gate_complete and selection_path.is_file() and heldout_path.is_file() and claim_path.is_file()
    present = any((run_path.is_file(), log is not None, selection_path.is_file(), heldout_path.is_file(), claim_path.is_file()))
    eval_walls = [item.get("wall_s") for item in run.get("evals", []) if item.get("wall_s") is not None]
    return {
        "arm": arm, "seed": seed, "run_tag": tag,
        "status": "complete" if complete else "observed" if present else "pending",
        "transitions": run.get("transitions"), "halt_reason": run.get("halt_reason"),
        "nan": (recount or {}).get("recounted_lower_bound", {}).get("nan", run.get("nan_incidents_env")),
        "mag": (recount or {}).get("recounted_lower_bound", {}).get("mag", run.get("magnitude_incidents_env")),
        "rebuilds": (recount or {}).get("rebuilds", run.get("full_scene_rebuilds")),
        "wall_h": _wall_hours(log), "initial_weights_sha256": run.get("initial_weights_sha256"),
        "eval_wall_max_s": max(eval_walls) if eval_walls else None,
        "run": _relative(run_path, root) if run_path.is_file() else None,
        "log": _relative(log_path, root) if log else None,
        "selection": _relative(selection_path, root) if selection_path.is_file() else None,
        "heldout": _relative(heldout_path, root) if heldout_path.is_file() else None,
        "claim": _relative(claim_path, root) if claim_path.is_file() else None,
    }


def _heldout_row(root: Path, row: dict[str, Any]) -> dict[str, Any] | None:
    if not row["heldout"]:
        return None
    result_path = root / row["heldout"]
    result = _load(result_path)
    summary = result.get("summary", {})
    selection = _load(root / row["selection"]) if row["selection"] else {}
    ckpt = selection.get("selected_ckpt", result.get("ckpt"))
    return {"arm": row["arm"], "seed": row["seed"], "status": row["status"],
            "success": summary.get("success_rate"), "return": summary.get("mean_return"),
            "ckpt": ckpt, "ckpt_sha256": result.get("ckpt_sha256"),
            "claim_sha256": result.get("claim_sha256") or _sha(root / row["claim"]) if row["claim"] else result.get("claim_sha256"),
            "result": row["heldout"]}


def _cell(value: Any, digits: int | None = None) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}" if digits is not None and isinstance(value, float) else str(value)


def generate(root: Path = ROOT) -> dict[str, Any]:
    root = Path(root)
    rows = [_run_row(root, arm, seed) for arm, seeds in ARMS.items() for seed in seeds]
    heldout = [item for row in rows if (item := _heldout_row(root, row)) is not None]
    aggregates = []
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        aggregates.append({"arm": arm, "runs": len(arm_rows),
                           "complete": sum(row["status"] == "complete" for row in arm_rows),
                           "observed": sum(row["status"] == "observed" for row in arm_rows),
                           "pending": sum(row["status"] == "pending" for row in arm_rows),
                           "heldout_results": sum(row["arm"] == arm for row in heldout)})
    payload = {"schema_version": 1, "grid": {arm: list(seeds) for arm, seeds in ARMS.items()},
               "runs": rows, "heldout": heldout, "arm_aggregates": aggregates}
    metrics = root / "outputs" / "metrics"
    reports = root / "outputs" / "reports"
    metrics.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    (metrics / "sprint_g6b_watch.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = ["# G6b 캠페인 감시 리포트", "", "생성: `uv run python scripts/generate_sprint_g6b_report.py`. 원천 아티팩트만 스캔한 증분 감시표이며, unblinding 전 arm 간 비교·해석은 포함하지 않는다.",
             "", "## Per-run 감시", "", "|arm|seed|상태|transitions|halt|nan/mag/rebuild|wall h|init hash|eval-wall max s|", "|-|-:|---|--:|---|--:|---|--:|"]
    for row in rows:
        counters = "/".join(_cell(row[key]) for key in ("nan", "mag", "rebuilds"))
        lines.append(f"|{row['arm']}|{row['seed']}|{row['status']}|{_cell(row['transitions'])}|{_cell(row['halt_reason'])}|{counters}|{_cell(row['wall_h'], 2)}|`{row['initial_weights_sha256'] or '—'}`|{_cell(row['eval_wall_max_s'], 3)}|")
    lines += ["", "## Held-out", "", "|arm|seed|상태|success|mean return|ckpt / sha256|claim sha256|원천|", "|---|-:|---|--:|--:|---|---|---|"]
    if heldout:
        for item in heldout:
            lines.append(f"|{item['arm']}|{item['seed']}|{item['status']}|{_cell(item['success'], 3)}|{_cell(item['return'], 6)}|`{_cell(item['ckpt'])}` / `{_cell(item['ckpt_sha256'])}`|`{_cell(item['claim_sha256'])}`|`{item['result']}`|")
    else:
        lines.append("|—|—|pending|—|—|—|—|—|")
    lines += ["", "## Arm별 사실 집계", "", "|arm|grid runs|complete|observed|pending|heldout results|", "|---|--:|--:|--:|--:|--:|"]
    for item in aggregates:
        lines.append(f"|{item['arm']}|{item['runs']}|{item['complete']}|{item['observed']}|{item['pending']}|{item['heldout_results']}|")
    lines += ["", "`pending`은 해당 그리드 태그의 run/selection/heldout/claim/log 아티팩트가 아직 없는 행이다. `observed`는 일부 아티팩트만 있어 완료 판정을 하지 않은 행이다."]
    (reports / "sprint_g6b_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    generate()
