"""Generate the sprint-dedicated T2 held-out split (paper-sprint prereg).

Authority: HUMAN instruction issue #13 comment 4985559491, directive 4;
pre-registration Decision B = research-dashboard#36 (sprint_spec.md v1 pinned
at 82230d8). Generator code is NOT modified — the same procedural generation
runs under a NEW master seed (recorded below); the new payload's own held-out
selection becomes the sprint split.

Usage contract (pre-registered): grid primary metrics and GNG-2 use THIS
split only, one final evaluation per run. The M4 held-out split remains
P1-judgment-exclusive — no sprint experiment may touch it.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import dgcc.tasks.t2 as t2

#: NEW sprint master seed (recorded; original T2_MASTER_SEED = 20260703).
SPRINT_MASTER_SEED = 20260716
SPRINT_SPLIT_FILENAME = "t2_sprint_heldout_v1.json"
SPRINT_VERSION = "t2-sprint-heldout-v1"


def main() -> int:
    assert SPRINT_MASTER_SEED != t2.T2_MASTER_SEED
    original_seed = t2.T2_MASTER_SEED
    # Seed swap only — generator code paths untouched.
    t2.T2_MASTER_SEED = SPRINT_MASTER_SEED
    try:
        payload = t2.generate_t2_payload()
    finally:
        t2.T2_MASTER_SEED = original_seed
    assert payload["master_seed"] == SPRINT_MASTER_SEED

    heldout_ids = set(payload["splits"]["heldout"])
    sprint_specs = [s for s in payload["specs"] if s["goal_id"] in heldout_ids]
    assert len(sprint_specs) == 100, len(sprint_specs)

    # Overlap check vs ALL 650 goals of the committed M4 payload (parameter level).
    m4 = t2.load_t2_payload()

    def key(spec):
        return (
            spec["family"],
            tuple(sorted((k, round(float(v), 12)) for k, v in spec["params"].items())),
            tuple(round(float(a), 12) for a in spec["anchor"]),
        )

    m4_keys = {key(s) for s in m4["specs"]}
    dupes = [s["goal_id"] for s in sprint_specs if key(s) in m4_keys]

    per_family = {}
    asym = 0
    for s in sprint_specs:
        per_family[s["family"]] = per_family.get(s["family"], 0) + 1
        asym += bool(s["asymmetric"])

    out_payload = {
        "version": SPRINT_VERSION,
        "master_seed": SPRINT_MASTER_SEED,
        "generator_version": payload["version"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "authority": {
            "instruction": "DGCC#13 comment 4985559491 directive 4",
            "decision": "research-dashboard#36",
            "sprint_spec_sha": "82230d8",
        },
        "usage": "sprint grid primary metrics + GNG-2 ONLY; one final eval per run; M4 heldout untouched",
        "families": payload["families"],
        "n_goals": len(sprint_specs),
        "overlap_with_t2_v1": len(dupes),
        "specs": sprint_specs,
        "goal_ids": sorted(s["goal_id"] for s in sprint_specs),
    }
    out = REPO / "src" / "dgcc" / "tasks" / "splits" / SPRINT_SPLIT_FILENAME
    out.write_text(json.dumps(out_payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")

    report = REPO / "outputs" / "reports" / "sprint_heldout_gen.md"
    report.write_text(f"""# Sprint held-out split generation report

- Authority: DGCC#13 comment 4985559491 directive 4 · Decision B research-dashboard#36 · sprint_spec 82230d8
- Generator: `dgcc.tasks.t2` UNMODIFIED (seed swap only) — procedural rules identical to t2-v1
- Sprint master seed: **{SPRINT_MASTER_SEED}** (original: {original_seed})
- Selection: the new payload's own held-out 100 (same permutation rule, seed-derived)
- Goals: {len(sprint_specs)} · per-family: {json.dumps(per_family, sort_keys=True)} · asymmetric: {asym}
- **Overlap with t2_v1 (all 650 goals, parameter-level key family+params+anchor): {len(dupes)} duplicates**{'' if not dupes else ' — ' + ', '.join(dupes)}
- File: `src/dgcc/tasks/splits/{SPRINT_SPLIT_FILENAME}`
- Usage (pre-registered): grid primary metrics + GNG-2 judgments only; one final eval per run; **M4 held-out untouched by any sprint experiment**
- Stability preflight (instruction 2 procedure) for this split: pending seed-gap GPU window
""", encoding="utf-8")
    print(f"sprint split: {out} ({len(sprint_specs)} goals, overlap={len(dupes)})")
    print(f"report: {report}")
    return 0 if not dupes else 1


if __name__ == "__main__":
    raise SystemExit(main())
