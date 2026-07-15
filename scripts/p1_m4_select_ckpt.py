"""P1-M4 val-checkpoint selection manifest (provenance for the held-out eval).

Pre-registered rule: highest val success_rate; ties -> higher mean_return;
remaining ties -> earliest transitions (most training left unspent).
The selected periodic checkpoint file is hashed so the held-out evaluator can
reject any post-selection substitution.

Usage:
    uv run python scripts/p1_m4_select_ckpt.py --run-tag m4_t2_s0
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SELECTION_RULE = "max val success_rate; tie -> max mean_return; tie -> min transitions"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_best_eval(evals: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the pre-registered selection rule over run-JSON eval rows."""

    if not evals:
        raise ValueError("run JSON contains no eval rows")
    return sorted(
        evals,
        key=lambda ev: (-ev["success_rate"], -ev["mean_return"], ev["transitions"]),
    )[0]


def build_manifest(run_tag: str, metrics_dir: Path = Path("outputs/metrics"),
                   models_dir: Path = Path("outputs/models")) -> dict[str, Any]:
    run_json = metrics_dir / f"p1_run_{run_tag}.json"
    run = json.loads(run_json.read_text(encoding="utf-8"))
    if run.get("halt_reason") is not None:
        raise ValueError(f"refusing selection for halted run: {run['halt_reason']}")
    best = select_best_eval(run["evals"])
    ckpt = models_dir / run_tag / f"ckpt_{best['transitions']:07d}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"selected checkpoint missing: {ckpt}")
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_tag": run_tag,
        "run_json": str(run_json),
        "selection_rule": SELECTION_RULE,
        "selected_transitions": int(best["transitions"]),
        "selected_ckpt": str(ckpt),
        "ckpt_sha256": sha256_file(ckpt),
        "val_success_rate": float(best["success_rate"]),
        "val_mean_return": float(best["mean_return"]),
        "eval_rows_considered": len(run["evals"]),
        "initial_weights_sha256": run.get("initial_weights_sha256"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", required=True)
    args = parser.parse_args()
    manifest = build_manifest(args.run_tag)
    out = Path("outputs/metrics") / f"p1_m4_ckpt_selection_{args.run_tag}.json"
    out.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(f"selection manifest: {out} -> {manifest['selected_ckpt']} "
          f"(val success {manifest['val_success_rate']:.3f} @ {manifest['selected_transitions']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
