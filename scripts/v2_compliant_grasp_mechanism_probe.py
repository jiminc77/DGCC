#!/usr/bin/env python3
"""M7-first grasp-transition mechanism probe (RALPLAN 72819147 revision-3 §2).

Replays one exact battery primitive at ``n_envs=1`` and captures solver-truth
state around the grasp transition.  The probe inserts **no** physics step: it
only reads state at the taps the adapter already exposes (``_set_gripper_positions``
and ``_step_scene``) plus one instance-level wrapper around the rod entity's
per-env attach call.

Taps (plan §2.2)::

    prev   previous primitive close (post-escalation constraint truth)
    T0a    before the new grasp command   <- M7 first partition, synchronized
    T0b    after set_pos / before step
    T1     after the free pre-attach step
    T2     after attach / before the constrained step
    T3     after the first hard-constrained step
    ...    every later natural step through settle

Arms:

    production  unmodified ``runner.step`` (baseline + M7 partition)
    replica     probe-driven reimplementation of the production ordering
                (harness-fidelity control: must match ``production``)
    nocommand   no gripper command, no attach, two natural steps  (M6)
    noop        ``set_pos(current gripper pose)``, no attach       (M6/M4)
    remote      equal-magnitude command to the farthest-from-rope
                candidate, no attach                                (M4/M1)
    C           exact placement -> attach -> constrained step       (candidate)
    cprime      exact placement -> attach -> first capped waypoint  (probe only)

Nothing here writes product state; the script is measurement-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v2_env_correction_acceptance import (  # noqa: E402
    battery_episode_plan,
    family_goals,
    init_genesis,
    sha256_file,
)

# ---------------------------------------------------------------------------
# Exact 91,000 action stream (byte-identical to the confirmatory precheck).
# ---------------------------------------------------------------------------

BATTERY_PRIMITIVES = 10
ACTION_STREAM_TAG = 91_000
GRASP_RNG_TAG = 88_000

CEILING = {"v": 2.0, "strain": 0.02, "ke_over_pe": 1.0}
ABSOLUTE_CAP = {"v": 10.0, "strain": 0.06, "ke_over_pe": 3.0}

# (episode, primitive) of the two AT-1H absolute-cap violators.
TARGETS = {
    "ep10p9": {"episode": 10, "primitive": 9},
    "ep25p8": {"episode": 25, "primitive": 8},
}


def stratified_actions(episode: int, seed: int) -> list[dict[str, Any]]:
    """Deterministic per-episode action schedule: lift balanced 5/5."""
    from dgcc.envs.dlolab import MAX_DELTA_NORM

    rng = np.random.default_rng([ACTION_STREAM_TAG, episode, seed])
    lifts = np.array(["low"] * 5 + ["high"] * 5, dtype=object)
    rng.shuffle(lifts)
    actions = []
    for _k in range(BATTERY_PRIMITIVES):
        radius = MAX_DELTA_NORM * float(np.sqrt(rng.uniform()))
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        actions.append(
            {
                "p": int(rng.integers(0, 32)),
                "delta": [radius * np.cos(angle), radius * np.sin(angle), 0.0],
                "lift": str(lifts[_k]),
            }
        )
    return actions


# ---------------------------------------------------------------------------
# Solver-truth readers
# ---------------------------------------------------------------------------


def _sync_backend() -> None:
    """Explicit device synchronization before a constraint snapshot (§2.2 T0a)."""
    import quadrants as qd

    qd.sync()


def constraint_truth(env) -> dict[str, Any]:
    """Full ``vertex_constraints`` snapshot for env 0, read after an explicit sync.

    Returns constrained mask, ``link_idx``, ``local_pos`` and the solver's
    currently stored ``target_pos`` for every vertex.  This is device truth,
    never a Python-side record (D1/D2 rationale).
    """
    solver = env.rod_entity._solver
    vc = solver.vertex_constraints
    _sync_backend()
    constrained = np.asarray(vc.constrained.to_numpy())
    link_idx = np.asarray(vc.link_idx.to_numpy())
    local_pos = np.asarray(vc.local_pos.to_numpy())
    target_pos = np.asarray(vc.target_pos.to_numpy())
    # Upstream layout is [n_vertices, B, ...]; keep env 0.
    constrained = constrained[:, 0].astype(bool)
    link_idx = link_idx[:, 0].astype(int)
    local_pos = local_pos[:, 0, :].astype(float)
    target_pos = target_pos[:, 0, :].astype(float)
    idx = np.flatnonzero(constrained)
    return {
        "n_constrained": int(constrained.sum()),
        "constrained_vertices": [int(v) for v in idx],
        "link_idx": {int(v): int(link_idx[v]) for v in idx},
        "local_pos": {int(v): local_pos[v].tolist() for v in idx},
        "target_pos": {int(v): target_pos[v].tolist() for v in idx},
        "_mask": constrained,
    }


def synchronized_constraint_partition(env) -> dict[str, Any]:
    """M7 first partition: two consistent synchronized snapshots (§2.5).

    Ordinary host-mask emptiness alone never rejects M7.  Two snapshots are
    taken with an explicit device synchronization in front of each; if they
    disagree, the observation is ``inconsistent`` and the caller must stop
    ``unidentified`` rather than infer emptiness.
    """
    snap_a = constraint_truth(env)
    snap_b = constraint_truth(env)
    consistent = bool(np.array_equal(snap_a["_mask"], snap_b["_mask"]))
    if consistent and snap_a["n_constrained"]:
        for key in ("link_idx", "local_pos", "target_pos"):
            if json.dumps(snap_a[key], sort_keys=True) != json.dumps(
                snap_b[key], sort_keys=True
            ):
                consistent = False
                break
    if not consistent:
        status = "inconsistent"
    elif snap_a["n_constrained"] == 0:
        status = "synchronized_empty"
    else:
        status = "synchronized_nonempty"
    out = {
        "status": status,
        "snapshots_consistent": consistent,
        "sync_call": "quadrants.sync()",
        "snapshot_a": {k: v for k, v in snap_a.items() if not k.startswith("_")},
        "snapshot_b": {k: v for k, v in snap_b.items() if not k.startswith("_")},
    }
    return out


def self_contact_truth(env) -> dict[str, Any]:
    """Rod-rod (non-adjacent self-contact) pairs with positive penetration."""
    solver = env.rod_entity._solver
    try:
        _sync_backend()
        info = np.asarray(solver.rr_constraint_info.valid_pair.to_numpy())
        pen = np.asarray(solver.rr_constraints.penetration.to_numpy())
    except Exception as exc:  # pragma: no cover - field availability is evidence
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    # penetration layout is [f, n_pairs, B]; take the current substep frame 0.
    pen = np.asarray(pen)
    if pen.ndim == 3:
        pen_now = pen[0, :, 0]
    elif pen.ndim == 2:
        pen_now = pen[:, 0]
    else:
        pen_now = pen.reshape(-1)
    active = np.flatnonzero(pen_now > 0.0)
    pairs = []
    for i_p in active[:64]:
        pair = np.asarray(info[int(i_p)]).reshape(-1)
        pairs.append(
            {
                "pair_index": int(i_p),
                "verts": [int(x) for x in pair.tolist()],
                "penetration": float(pen_now[int(i_p)]),
            }
        )
    return {
        "available": True,
        "n_active_pairs": int(active.size),
        "max_penetration": float(pen_now.max()) if pen_now.size else 0.0,
        "pairs": pairs,
    }


# ---------------------------------------------------------------------------
# Instrumented environment
# ---------------------------------------------------------------------------


def build_mechanism_env(n_envs: int = 1):
    """Same constructor as ``build_probe_env`` plus full per-step state capture."""
    from dgcc.envs.dlolab import DLOLabEnv, SEGMENT_MASS_BASE

    V_MAX = 0.05
    HOLD_MAX_STEPS = 2000

    class MechanismEnv(DLOLabEnv):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.trace_active = False
            self.trace: list[dict[str, Any]] = []
            self.step_ordinal = 0
            self.cmd_ordinal = 0
            self._pending_cmd: str | None = None
            self._last_cmd: np.ndarray | None = None
            self._causal_phase = "prefix"
            self._deep_steps: set[int] = set()
            self.t2_records: list[dict[str, Any]] = []
            self._attach_wrapped = False

        # -- taps -------------------------------------------------------
        def _set_gripper_positions(self, positions: np.ndarray) -> None:
            pos = np.asarray(positions, dtype=float)
            if self.trace_active:
                if self._last_cmd is not None and np.array_equal(pos, self._last_cmd):
                    self._pending_cmd = "hold"
                else:
                    self._pending_cmd = "move"
                self._last_cmd = pos.copy()
                self.cmd_ordinal += 1
                self._cmd_pre_gripper = self._gripper_positions()[0].copy()
                self._cmd_requested = pos[0].copy()
            super()._set_gripper_positions(positions)
            if self.trace_active:
                self._cmd_post_gripper = self._gripper_positions()[0].copy()

        def _step_scene(self) -> None:
            super()._step_scene()
            if self.trace_active:
                self.step_ordinal += 1
                phase = self._pending_cmd or "settle"
                self._pending_cmd = None
                self.trace.append(self.capture(phase))

        def _verified_detach_batch(self):
            if self.trace_active:
                self._causal_phase = "detach"
            residual, escalations = super()._verified_detach_batch()
            if self.trace_active:
                self.post_detach_partition = synchronized_constraint_partition(self)
                self.post_detach_counters = {
                    "detach_residuals": int(residual),
                    "detach_escalations": int(escalations),
                }
            return residual, escalations

        # -- state capture ----------------------------------------------
        def capture(self, phase: str, *, deep: bool | None = None) -> dict[str, Any]:
            raw = np.asarray(self._raw_batch(), dtype=float)[0]
            vels = np.asarray(self.rod_entity.get_all_vels(), dtype=float)
            if vels.ndim == 2:
                vels = vels[None, ...]
            vel = vels[0]
            speed = np.linalg.norm(vel, axis=-1)
            edges = np.linalg.norm(raw[1:] - raw[:-1], axis=-1)
            rest = float(self.params.length_m) / (raw.shape[0] - 1)
            signed = edges / rest - 1.0
            strain = np.abs(signed)
            ke = float(0.5 * SEGMENT_MASS_BASE * np.sum(vel**2))
            gripper = self._gripper_positions()[0]
            node_z = raw[:, 2]
            row: dict[str, Any] = {
                "step": int(self.step_ordinal),
                "compat_phase": phase,
                "causal_phase": self._causal_phase,
                "v_max": float(speed.max()),
                "v_argmax_node": int(speed.argmax()),
                "strain_max": float(strain.max()),
                "strain_argmax_edge": int(strain.argmax()),
                "strain_signed_at_argmax": float(signed[int(strain.argmax())]),
                "ke": ke,
                "arclen": float(edges.sum()),
                "min_node_z": float(node_z.min()),
                "min_node_z_index": int(node_z.argmin()),
                "n_nodes_below_zero": int((node_z < 0.0).sum()),
                "gripper": gripper.tolist(),
            }
            deep = self.step_ordinal in self._deep_steps if deep is None else deep
            if deep:
                row["positions"] = raw.tolist()
                row["velocities"] = vel.tolist()
                row["node_speeds"] = speed.tolist()
                row["edge_strain_signed"] = signed.tolist()
                row["constraints"] = {
                    k: v
                    for k, v in constraint_truth(self).items()
                    if not k.startswith("_")
                }
                row["self_contact"] = self_contact_truth(self)
            return row

        # -- bookkeeping -------------------------------------------------
        def begin_trace(self, deep_steps: set[int] | None = None) -> None:
            self.trace = []
            self.step_ordinal = 0
            self.cmd_ordinal = 0
            self._pending_cmd = None
            self._last_cmd = None
            self._causal_phase = "grasp_free"
            self._deep_steps = deep_steps or set()
            self.t2_records = []
            self.post_detach_partition = None
            self.post_detach_counters = None
            self.trace_active = True

        def end_trace(self) -> list[dict[str, Any]]:
            self.trace_active = False
            return self.trace

    return MechanismEnv(
        n_envs=n_envs,
        dt=1.0e-3,
        substeps=5,
        rod_damping=10.0,
        rod_angular_damping=5.0,
        initial_settle_steps=0,
        reset_settle_max_steps=10_000,
        move_v_max=V_MAX,
        move_hold_max_steps=HOLD_MAX_STEPS,
        grasp_realism=True,
    )


# ---------------------------------------------------------------------------
# Graph-distance evidence
# ---------------------------------------------------------------------------


def graph_distances(peak_node: int, peak_edge: int, anchors: dict[str, int | None]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, node in anchors.items():
        if node is None or node < 0:
            out[f"d_peaknode_to_{name}"] = None
            out[f"d_peakedge_to_{name}"] = None
            continue
        out[f"d_peaknode_to_{name}"] = int(abs(int(peak_node) - int(node)))
        out[f"d_peakedge_to_{name}"] = int(
            min(abs(int(peak_edge) - int(node)), abs(int(peak_edge) + 1 - int(node)))
        )
    return out


# ---------------------------------------------------------------------------
# Replay driver
# ---------------------------------------------------------------------------


def resolve_target(name: str) -> dict[str, Any]:
    spec = TARGETS[name]
    plan = {entry["episode"]: entry for entry in battery_episode_plan()}
    entry = plan[spec["episode"]]
    goals = family_goals()
    _family, _goal_id, goal = goals[entry["family_index"]]
    return {
        "name": name,
        "episode": int(spec["episode"]),
        "primitive": int(spec["primitive"]),
        "entry": entry,
        "goal": goal,
    }


def run_arm(arm: str, target_name: str, backend_info: dict[str, Any]) -> dict[str, Any]:
    from dgcc.envs.dlolab import LIFT_HEIGHTS, MAX_DELTA_NORM, sample_grasp
    from dgcc.tasks.domain import p1_rope_params
    from dgcc.tasks.episode import BatchedEpisodeRunner, EpisodeConfig

    tgt = resolve_target(target_name)
    entry = tgt["entry"]
    episode = tgt["episode"]
    k_target = tgt["primitive"]
    params = p1_rope_params()

    env = build_mechanism_env(1)
    env.reset(params, init_shape=entry["init_shape"], seed=1_000 + episode)
    runner = BatchedEpisodeRunner(env, params, EpisodeConfig())
    runner.begin_episodes(
        seed=entry["seed"],
        episode_index=episode,
        init_shapes=[entry["init_shape"]],
        goals=[tgt["goal"]],
    )

    actions = stratified_actions(episode, entry["seed"])

    # -- exact prefix 0..k-1 through the untouched production path --------
    # The last prefix primitive is traced so the T-5..T-1 natural horizon and
    # the previous post-escalation constraint truth come from real steps.
    # The probe inserts no physics step of its own.
    prefix_grasps: list[dict[str, Any]] = []
    prev_trace: list[dict[str, Any]] = []
    for k in range(k_target):
        traced = k == k_target - 1
        if traced:
            env.begin_trace(deep_steps=set())
            env._causal_phase = "prev_primitive"
        runner.step(
            np.asarray([actions[k]["p"]], dtype=int),
            np.asarray([actions[k]["delta"]], dtype=float),
            [actions[k]["lift"]],
            rng=np.random.default_rng([GRASP_RNG_TAG, episode, k]),
        )
        if traced:
            prev_trace = env.end_trace()
        prefix_grasps.append(
            {
                "primitive": k,
                "p_requested": int(actions[k]["p"]),
                "p_actual": int(env.last_grasp_actual_nodes[0]),
                "grasp_success": bool(env.last_grasp_successes[0]),
                "detach_residuals": int(env.last_detach_residuals),
            }
        )
    prefix_state = np.asarray(env._raw_batch(), dtype=float)
    prefix_hash = hashlib.sha256(prefix_state.tobytes()).hexdigest()

    # -- previous primitive close ----------------------------------------
    previous_close = {
        "previous_primitive": k_target - 1 if k_target else None,
        "previous_p_requested": prefix_grasps[-1]["p_requested"] if prefix_grasps else None,
        "previous_p_actual": prefix_grasps[-1]["p_actual"] if prefix_grasps else None,
        "previous_grasp_success": prefix_grasps[-1]["grasp_success"] if prefix_grasps else None,
        "previous_detach_residuals": prefix_grasps[-1]["detach_residuals"] if prefix_grasps else None,
        "previous_post_detach_partition": getattr(env, "post_detach_partition", None),
        "previous_post_detach_counters": getattr(env, "post_detach_counters", None),
        "detach_escalation_total": int(env.detach_escalation_total),
        "post_escalation_partition": synchronized_constraint_partition(env),
    }

    # -- T-5..T-1: the last five REAL steps of the previous primitive ------
    # (no step is injected by the probe)
    pre_horizon = prev_trace[-5:]

    # -- T0a: M7 FIRST PARTITION ------------------------------------------
    t0a = synchronized_constraint_partition(env)
    t0a_self_contact = self_contact_truth(env)
    gripper_before = env._gripper_positions()[0].copy()

    action = actions[k_target]
    delta = np.asarray(action["delta"], dtype=float).copy()
    delta[2] = 0.0
    norm = float(np.linalg.norm(delta))
    if norm > MAX_DELTA_NORM:
        delta = delta * (MAX_DELTA_NORM / norm)

    grasp_rng = np.random.default_rng([GRASP_RNG_TAG, episode, k_target])
    n_vertices = env._n_vertices()
    p_requested = int(action["p"])
    raw_before = env.get_centerline_raw_batch()

    # M7 predicted response: where would each pre-existing constraint target
    # move once the gripper link is commanded to the new node position?
    node_target_pos = raw_before[0, 0, :]  # placeholder, replaced below
    p_actual_preview, success_preview = sample_grasp(
        p_requested, n_vertices, np.random.default_rng([GRASP_RNG_TAG, episode, k_target]), env.grasp_realism
    )
    node_target_pos = raw_before[0, int(p_actual_preview), :]
    gripper_displacement = float(np.linalg.norm(node_target_pos - gripper_before))
    m7_prediction = {
        "p_requested": p_requested,
        "p_actual_preview": int(p_actual_preview),
        "grasp_success_preview": bool(success_preview),
        "gripper_before": gripper_before.tolist(),
        "gripper_commanded_to": node_target_pos.tolist(),
        "gripper_displacement_m": gripper_displacement,
        "predicted_old_target_shift_m": gripper_displacement if t0a["status"] == "synchronized_nonempty" else 0.0,
    }

    # ------------------------------------------------------------------
    # Arm execution
    # ------------------------------------------------------------------
    deep = {1, 2, 3, 4}
    result: dict[str, Any] = {}

    if arm == "production":
        env.begin_trace(deep_steps=deep)
        env.probe_hook_t2 = None
        out = _run_production_primitive(env, runner, action, episode, k_target)
        trace = env.end_trace()
        result["primitive_out"] = out
    else:
        trace, result = _run_arm_segment(
            arm,
            env,
            runner,
            action,
            episode,
            k_target,
            delta,
            raw_before,
            deep,
        )

    # ------------------------------------------------------------------
    # Aggregates / crossings
    # ------------------------------------------------------------------
    summary = _summarize(trace, env, action)
    p_actual = int(env.last_grasp_actual_nodes[0]) if env.last_grasp_actual_nodes is not None else None
    anchors = {
        "prev_p_actual": previous_close["previous_p_actual"],
        "cur_p_requested": p_requested,
        "cur_p_actual": p_actual,
    }
    if summary["first_crossing"] is not None:
        row = trace[summary["first_crossing"]["trace_index"]]
        summary["first_crossing"]["graph_distances"] = graph_distances(
            row["v_argmax_node"], row["strain_argmax_edge"], anchors
        )

    return {
        "arm": arm,
        "target": target_name,
        "episode": episode,
        "primitive": k_target,
        "init_shape": entry["init_shape"],
        "family": entry["family"],
        "goal_id": entry["goal_id"],
        "seed": int(entry["seed"]),
        "backend": backend_info,
        "action": action,
        "delta_clamped": delta.tolist(),
        "prefix_state_sha256": prefix_hash,
        "prefix_grasps": prefix_grasps,
        "previous_close": previous_close,
        "pre_horizon": pre_horizon,
        "T0a": t0a,
        "T0a_self_contact": t0a_self_contact,
        "m7_prediction": m7_prediction,
        "grasp": {
            "p_requested": p_requested,
            "p_actual": p_actual,
            "noise_offset": (p_actual - p_requested) if p_actual is not None else None,
            "grasp_success": bool(env.last_grasp_successes[0])
            if env.last_grasp_successes is not None
            else None,
        },
        "t2_records": env.t2_records,
        "post_detach_partition": getattr(env, "post_detach_partition", None),
        "post_detach_counters": getattr(env, "post_detach_counters", None),
        "summary": summary,
        "trace": trace,
        **result,
    }


def _run_production_primitive(env, runner, action, episode, k_target) -> dict[str, Any]:
    """Unmodified production path with a T2 wrapper on the entity attach call."""
    entity = env.rod_entity
    original = entity.attach_to_rigid_link_with_envs_idx

    def wrapped(link, verts_ids, envs_idx, *args, **kwargs):
        before = constraint_truth(env)
        out = original(link, verts_ids, envs_idx, *args, **kwargs)
        after = constraint_truth(env)
        node = int(verts_ids[0])
        raw = np.asarray(env._raw_batch(), dtype=float)[0]
        link_pos = env._gripper_positions()[0]
        local = np.asarray(after["local_pos"].get(node, [np.nan] * 3), dtype=float)
        env.t2_records.append(
            {
                "tap": "T2",
                "step_ordinal": int(env.step_ordinal),
                "attached_vertex": node,
                "env_idx": int(envs_idx),
                "link_idx_stored": after["link_idx"].get(node),
                "mask_before": before["constrained_vertices"],
                "mask_after": after["constrained_vertices"],
                "local_pos_stored": local.tolist(),
                "link_pos_world": link_pos.tolist(),
                "vertex_pos_world": raw[node].tolist(),
                "node_link_world_offset_m": float(
                    np.linalg.norm(raw[node] - link_pos)
                ),
                "local_pos_norm_m": float(np.linalg.norm(local)),
                # Upstream stores link-local coordinates; a nonzero world
                # offset is preserved, not snapped (plan §2.2).
                "offset_preserved_residual_m": float(
                    abs(float(np.linalg.norm(raw[node] - link_pos)) - float(np.linalg.norm(local)))
                ),
            }
        )
        return out

    entity.attach_to_rigid_link_with_envs_idx = wrapped
    try:
        env._causal_phase = "grasp_free"
        out = runner.step(
            np.asarray([action["p"]], dtype=int),
            np.asarray([action["delta"]], dtype=float),
            [action["lift"]],
            rng=np.random.default_rng([GRASP_RNG_TAG, episode, k_target]),
        )
    finally:
        entity.attach_to_rigid_link_with_envs_idx = original
    return {
        "settle_steps": int(out["settle_steps"][0]),
        "grasp_success": bool(out["grasp_success"][0]),
    }


def _run_arm_segment(
    arm, env, runner, action, episode, k_target, delta, raw_before, deep
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reimplemented grasp-transition segment used by the causal control arms.

    Mirrors ``DLOLabEnv.step_primitive_batch`` lines 973-1006 exactly for the
    ``replica`` arm; each control arm changes exactly one factor.
    """
    from dgcc.envs.dlolab import sample_grasp

    n_vertices = env._n_vertices()
    grasp_rng = np.random.default_rng([GRASP_RNG_TAG, episode, k_target])
    p_requested = int(action["p"])
    p_actual, success = sample_grasp(p_requested, n_vertices, grasp_rng, env.grasp_realism)
    env.last_grasp_actual_nodes = np.asarray([p_actual], dtype=int)
    env.last_grasp_successes = np.asarray([success], dtype=bool)

    node_pos = raw_before[0, int(p_actual), :]
    gripper_before = env._gripper_positions()[0].copy()

    env.begin_trace(deep_steps=deep)
    env._at1h_begin()
    info: dict[str, Any] = {"arm": arm}

    def do_attach():
        env._batched_active_nodes = np.full(1, -1, dtype=int)
        if success:
            before = constraint_truth(env)
            env.rod_entity.attach_to_rigid_link_with_envs_idx(
                env.gripper_link, [int(p_actual)], 0
            )
            after = constraint_truth(env)
            raw = np.asarray(env._raw_batch(), dtype=float)[0]
            link_pos = env._gripper_positions()[0]
            local = np.asarray(after["local_pos"].get(int(p_actual), [np.nan] * 3), dtype=float)
            env.t2_records.append(
                {
                    "tap": "T2",
                    "step_ordinal": int(env.step_ordinal),
                    "attached_vertex": int(p_actual),
                    "mask_before": before["constrained_vertices"],
                    "mask_after": after["constrained_vertices"],
                    "local_pos_stored": local.tolist(),
                    "link_pos_world": link_pos.tolist(),
                    "vertex_pos_world": raw[int(p_actual)].tolist(),
                    "node_link_world_offset_m": float(np.linalg.norm(raw[int(p_actual)] - link_pos)),
                    "local_pos_norm_m": float(np.linalg.norm(local)),
                }
            )
            env._batched_active_nodes[0] = int(p_actual)

    if arm == "replica":
        env._set_gripper_positions(node_pos.reshape(1, 3))
        env._causal_phase = "grasp_free"
        env._step_scene()
        do_attach()
        env._causal_phase = "grasp_constrained"
        env._step_scene()
        _finish_primitive(env, delta, action)
    elif arm == "C":
        env._set_gripper_positions(node_pos.reshape(1, 3))
        do_attach()
        env._causal_phase = "grasp_constrained"
        env._step_scene()
        _finish_primitive(env, delta, action)
    elif arm == "cprime":
        env._set_gripper_positions(node_pos.reshape(1, 3))
        do_attach()
        env._causal_phase = "grasp_constrained"
        _finish_primitive(env, delta, action)
    elif arm == "nocommand":
        # M6: exact prefix, no command, no attach; two natural steps.
        env._causal_phase = "grasp_free"
        env._step_scene()
        env._causal_phase = "grasp_constrained"
        env._step_scene()
        info["note"] = "no set_pos, no attach"
    elif arm == "noop":
        env._set_gripper_positions(gripper_before.reshape(1, 3))
        env._causal_phase = "grasp_free"
        env._step_scene()
        env._causal_phase = "grasp_constrained"
        env._step_scene()
        info["note"] = "set_pos(current gripper pose), no attach"
    elif arm == "remote":
        remote_target, valid, min_d = _remote_target(raw_before[0], gripper_before, node_pos)
        info["remote_target"] = remote_target.tolist()
        info["remote_min_node_distance_m"] = min_d
        info["remote_control_valid"] = valid
        env._set_gripper_positions(remote_target.reshape(1, 3))
        env._causal_phase = "grasp_free"
        env._step_scene()
        env._causal_phase = "grasp_constrained"
        env._step_scene()

    elif arm == "plane_lift":
        # M5 preregistered plane discriminator (plan §2.3): uniformly translate
        # the whole rope state and the gripper by dz so every 5 mm-radius node
        # centre sits 1 mm above the plane, restore identical node velocities
        # WITHOUT a scene step, then take the same two natural steps as the
        # `nocommand` arm.  Classification only; never enters production.
        raw = np.asarray(env._raw_batch(), dtype=float)
        vel = np.asarray(env.rod_entity.get_all_vels(), dtype=float)
        if vel.ndim == 2:
            vel = vel[None, ...]
        radius = float(env.params.radius)
        clearance = radius + 0.001
        dz = float(max(0.0, clearance - raw[0, :, 2].min()))
        edges_before = np.linalg.norm(raw[0, 1:] - raw[0, :-1], axis=-1)
        lifted = raw.copy()
        lifted[:, :, 2] += dz
        env.rod_entity.set_position(lifted)
        env.rod_entity.set_velocity(vel)
        env._set_gripper_positions((gripper_before + np.array([0.0, 0.0, dz])).reshape(1, 3))
        raw_after = np.asarray(env._raw_batch(), dtype=float)
        vel_after = np.asarray(env.rod_entity.get_all_vels(), dtype=float)
        if vel_after.ndim == 2:
            vel_after = vel_after[None, ...]
        edges_after = np.linalg.norm(raw_after[0, 1:] - raw_after[0, :-1], axis=-1)
        info["plane_lift"] = {
            "dz_m": dz,
            "node_radius_m": radius,
            "required_clearance_m": clearance,
            "min_node_z_before": float(raw[0, :, 2].min()),
            "min_node_z_after": float(raw_after[0, :, 2].min()),
            "max_edge_length_change_m": float(np.abs(edges_after - edges_before).max()),
            "max_velocity_change_mps": float(np.abs(vel_after[0] - vel[0]).max()),
            "invariants_hold": bool(
                float(np.abs(edges_after - edges_before).max()) < 1e-9
                and float(np.abs(vel_after[0] - vel[0]).max()) < 1e-9
                and float(raw_after[0, :, 2].min()) >= clearance - 1e-9
            ),
        }
        env._causal_phase = "grasp_free"
        env._step_scene()
        env._causal_phase = "grasp_constrained"
        env._step_scene()
        info["note"] = "uniform +dz rope+gripper translation, no command, no attach"
    else:
        raise ValueError(f"unknown arm {arm!r}")

    trace = env.end_trace()
    try:
        env._at1h_end([action["lift"]])
    except Exception:
        pass
    return trace, info


def _finish_primitive(env, delta, action) -> None:
    env._causal_phase = "walk"
    env._move_prepared_batch(delta.reshape(1, 3), [action["lift"]], vel_threshold=1e-3)
    env._verified_detach_batch()
    env._batched_active_nodes = None
    env._step_scene()
    env._causal_phase = "settle"
    env.settle_batch(vel_threshold=1e-3, max_steps=5000)


def _remote_target(raw, gripper_before, node_pos):
    """Equal-magnitude command to the signed-axis candidate farthest from the rope."""
    disp = node_pos - gripper_before
    norm = float(np.linalg.norm(disp))
    best = None
    best_d = -1.0
    for axis in range(3):
        for sign in (-1.0, 1.0):
            cand = gripper_before.copy()
            cand[axis] += sign * norm
            d = float(np.linalg.norm(raw - cand, axis=-1).min())
            if d > best_d:
                best_d = d
                best = cand
    # 0.0075 sphere radius + 0.005 rod radius = 0.0125 m geometric contact bound.
    return best, bool(best_d > 0.0125), best_d


def _summarize(trace, env, action) -> dict[str, Any]:
    from dgcc.envs.dlolab import LIFT_HEIGHTS, SEGMENT_MASS_BASE

    if not trace:
        return {"n_steps": 0, "first_crossing": None}
    rope_mass = 32 * 1.0e-3
    grav_pe = rope_mass * 9.81 * float(LIFT_HEIGHTS[action["lift"]])
    v = np.asarray([r["v_max"] for r in trace])
    s = np.asarray([r["strain_max"] for r in trace])
    ke = np.asarray([r["ke"] for r in trace])
    kepe = ke / grav_pe
    minz = np.asarray([r["min_node_z"] for r in trace])

    first_crossing = None
    for i, row in enumerate(trace):
        if (
            row["v_max"] > CEILING["v"]
            or row["strain_max"] > CEILING["strain"]
            or (row["ke"] / grav_pe) > CEILING["ke_over_pe"]
        ):
            first_crossing = {
                "trace_index": i,
                "step": row["step"],
                "compat_phase": row["compat_phase"],
                "causal_phase": row["causal_phase"],
                "v_max": row["v_max"],
                "strain_max": row["strain_max"],
                "ke_over_pe": row["ke"] / grav_pe,
                "v_argmax_node": row["v_argmax_node"],
                "strain_argmax_edge": row["strain_argmax_edge"],
            }
            break

    first_penetration = None
    pen_idx = np.flatnonzero(minz < 0.0)
    if pen_idx.size:
        i = int(pen_idx[0])
        first_penetration = {
            "trace_index": i,
            "step": trace[i]["step"],
            "causal_phase": trace[i]["causal_phase"],
            "min_node_z": trace[i]["min_node_z"],
        }

    order = None
    if first_crossing and first_penetration:
        if first_penetration["trace_index"] < first_crossing["trace_index"]:
            order = "penetration_before_crossing"
        elif first_penetration["trace_index"] > first_crossing["trace_index"]:
            order = "crossing_before_penetration"
        else:
            order = "same_step"
    elif first_crossing:
        order = "crossing_only"
    elif first_penetration:
        order = "penetration_only"

    by_phase: dict[str, Any] = {}
    for row in trace:
        ph = row["causal_phase"]
        rec = by_phase.setdefault(ph, {"steps": 0, "v_max": 0.0, "strain_max": 0.0, "ke_over_pe": 0.0})
        rec["steps"] += 1
        rec["v_max"] = max(rec["v_max"], row["v_max"])
        rec["strain_max"] = max(rec["strain_max"], row["strain_max"])
        rec["ke_over_pe"] = max(rec["ke_over_pe"], row["ke"] / grav_pe)

    return {
        "n_steps": len(trace),
        "v_peak_total": float(v.max()),
        "strain_peak_total": float(s.max()),
        "ke_over_pe_peak": float(kepe.max()),
        "min_node_z": float(minz.min()),
        "ground_penetration_steps": int((minz < 0.0).sum()),
        "grav_pe": grav_pe,
        "ceiling_crossed": bool(
            v.max() > CEILING["v"] or s.max() > CEILING["strain"] or kepe.max() > CEILING["ke_over_pe"]
        ),
        "absolute_cap_violated": bool(
            v.max() > ABSOLUTE_CAP["v"]
            or s.max() > ABSOLUTE_CAP["strain"]
            or kepe.max() > ABSOLUTE_CAP["ke_over_pe"]
        ),
        "first_crossing": first_crossing,
        "first_penetration": first_penetration,
        "temporal_ground_order": order,
        "by_causal_phase": by_phase,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--target", required=True, choices=sorted(TARGETS))
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--backend", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    backend_info = init_genesis(args.backend)
    result = run_arm(args.arm, args.target, backend_info)
    result["repeat"] = int(args.repeat)
    result["probe_sha256"] = sha256_file(Path(__file__).resolve())
    result["adapter_sha256"] = sha256_file(
        Path(__file__).resolve().parents[1] / "src" / "dgcc" / "envs" / "dlolab.py"
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, sort_keys=False))
    s = result["summary"]
    print(
        json.dumps(
            {
                "arm": args.arm,
                "target": args.target,
                "repeat": args.repeat,
                "T0a": result["T0a"]["status"],
                "p_actual": result["grasp"]["p_actual"],
                "v_peak": s.get("v_peak_total"),
                "strain_peak": s.get("strain_peak_total"),
                "ke_over_pe": s.get("ke_over_pe_peak"),
                "first_crossing": s.get("first_crossing"),
                "ground_order": s.get("temporal_ground_order"),
                "out": str(out),
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
