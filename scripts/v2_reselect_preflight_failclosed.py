#!/usr/bin/env python
"""Fail-closed demonstration for the reselection preflight (task B / D-9).

CPU-only: it exercises `dgcc.envs.env_config.resolve_env_kwargs` against the
real `DLOLabEnv` signature and the real shipped config, so it proves the
guard without touching the GPU or the training loop.

Cases
-----
  1. canonical  configs/v2_t2.yaml resolves and yields the corrected env kwargs
  2. legacy     the pre-correction `sim` block is REFUSED
  3. missing    `move_v_max` absent (the old silent-legacy path) is REFUSED
  4. typo       `move_vmax` (the old silent-drop path) is REFUSED
  5. partial    `move_v_max` without `move_hold_max_steps` is REFUSED
  6. null       `move_v_max: null` is REFUSED
  7. bypass     a constructed adapter that is NOT quasi-static is REFUSED
  8. parity     all 15 published cell configs are byte-identical to configs/
  9. nodes      `n_segments` absent (Rev 6: the silent-domain-default path) is REFUSED
 10. mass       `rope_mass_total` absent (Rev 6) is REFUSED
 11. mismatch   `sim` and the rope domain disagreeing on n_segments is REFUSED

Exit code 0 iff every case behaves as specified.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dgcc.envs.env_config import (  # noqa: E402
    EnvConfigError,
    allowed_sim_keys,
    assert_corrected_env,
    resolve_env_kwargs,
)


def _real_env_cls() -> type:
    """Import the adapter lazily.

    `DLOLabEnv` only touches Genesis inside `reset()`, so importing the class
    (which is all the resolver needs -- it reads the constructor signature)
    keeps this battery CPU-only.
    """
    from dgcc.envs.dlolab import DLOLabEnv

    return DLOLabEnv




def _expect_refusal(name: str, config: dict[str, Any], env_cls: type, needle: str) -> bool:
    try:
        resolve_env_kwargs(config, 4096, env_cls=env_cls, require_corrected=True)
    except EnvConfigError as error:
        ok = needle in str(error)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: refused -> {error}")
        return ok
    print(f"[FAIL] {name}: RESOLVED, expected refusal")
    return False


def main() -> int:
    env_cls = _real_env_cls()
    config_path = ROOT / "configs" / "v2_t2.yaml"
    canonical = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    results: list[bool] = []

    print(f"adapter whitelist ({env_cls.__name__}): {sorted(allowed_sim_keys(env_cls))}")
    print(f"config: {config_path} sha256={hashlib.sha256(config_path.read_bytes()).hexdigest()}")
    print()

    # 1. canonical config resolves to the corrected env.
    kwargs = resolve_env_kwargs(
        canonical, int(canonical["run"]["n_envs"]), env_cls=env_cls, require_corrected=True
    )
    ok = (
        kwargs.get("move_v_max") == 0.15
        and kwargs.get("move_hold_max_steps") == 2000
        and kwargs.get("n_segments") == 64
        and kwargs.get("rope_mass_total") == 0.040
        and kwargs.get("at1h_counters") is True
        and kwargs.get("n_envs") == 4096
        and "move_step_size" not in kwargs
        and "move_hold_steps" not in kwargs
    )
    print(f"[{'PASS' if ok else 'FAIL'}] canonical: {json.dumps(kwargs, sort_keys=True)}")
    results.append(ok)

    # 2. legacy `sim` block (exactly what the 15 cell copies used to carry).
    legacy = copy.deepcopy(canonical)
    legacy["sim"].pop("move_v_max")
    legacy["sim"].pop("move_hold_max_steps")
    legacy["sim"]["move_step_size"] = 0.03
    legacy["sim"]["move_hold_steps"] = 0
    results.append(_expect_refusal("legacy", legacy, env_cls, "DEPRECATED pre-correction key"))

    # 3. corrected keys simply absent -> the old silent legacy fallback.
    missing = copy.deepcopy(canonical)
    missing["sim"].pop("move_v_max")
    missing["sim"].pop("move_hold_max_steps")
    results.append(_expect_refusal("missing", missing, env_cls, "missing"))

    # 4. typo -> the old silent-drop path.
    typo = copy.deepcopy(canonical)
    typo["sim"]["move_vmax"] = typo["sim"].pop("move_v_max")
    results.append(_expect_refusal("typo", typo, env_cls, "unknown `sim` key"))

    # 5. half the indivisible bundle.
    partial = copy.deepcopy(canonical)
    partial["sim"].pop("move_hold_max_steps")
    results.append(_expect_refusal("partial", partial, env_cls, "missing"))

    # 6. explicit null.
    nulled = copy.deepcopy(canonical)
    nulled["sim"]["move_v_max"] = None
    results.append(_expect_refusal("null", nulled, env_cls, "is null"))

    # 7. direct-construction bypass: an adapter that came out legacy anyway.
    class _LegacyAdapter:
        quasi_static = False
        move_v_max = None
        move_hold_max_steps = None

    try:
        assert_corrected_env(_LegacyAdapter(), kwargs)
        print("[FAIL] bypass: accepted a non-quasi-static adapter")
        results.append(False)
    except EnvConfigError as error:
        print(f"[PASS] bypass: refused -> {error}")
        results.append(True)

    # 8. single-config-SHA premise across the 15 published cells.
    cells = sorted((ROOT / "release/v2/preflight_15_not_admitted/cells").glob("*/v2_t2.yaml"))
    digests = {hashlib.sha256(p.read_bytes()).hexdigest() for p in cells}
    canonical_digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    ok = len(cells) == 15 and digests == {canonical_digest}
    print(
        f"[{'PASS' if ok else 'FAIL'}] cell-parity: {len(cells)} cells, "
        f"{len(digests)} distinct sha256, matches configs/ = {digests == {canonical_digest}}"
    )
    results.append(ok)

    # 9/10. Rev 6: the rope discretization and total mass are physics, so an
    # omitted declaration must refuse rather than inherit a code default.
    for key in ("n_segments", "rope_mass_total"):
        dropped = copy.deepcopy(canonical)
        dropped["sim"].pop(key)
        results.append(_expect_refusal(f"no-{key}", dropped, env_cls, "missing"))

    # 11. Rev 6: the `sim` declaration and the rope domain object must agree.
    # The resolver cannot see the domain object, so the adapter enforces it;
    # this exercises the adapter guard directly (still CPU-only -- the check
    # runs before Genesis is touched).
    from dgcc.tasks.domain import p1_rope_params

    mismatched = env_cls(
        n_envs=1,
        move_v_max=0.15,
        move_hold_max_steps=2000,
        n_segments=int(p1_rope_params().n_segments) + 1,
        rope_mass_total=float(p1_rope_params().rope_mass_total_kg),
    )
    try:
        mismatched._assert_domain_matches_config(p1_rope_params())
        print("[FAIL] domain-mismatch: accepted a config/domain disagreement")
        results.append(False)
    except ValueError as error:
        print(f"[PASS] domain-mismatch: refused -> {error}")
        results.append(True)

    print()
    print(f"fail-closed battery: {sum(results)}/{len(results)} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
