"""Process-isolated CPU Q1-versus-Qmin one-step counterfactual diagnostic."""
from __future__ import annotations

import copy
import hashlib
from io import BytesIO
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import random
import sys
import tempfile
import stat
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dgcc.envs.dlolab import DLOLabEnv
from dgcc.goals.dual_goal import goal_curve
from dgcc.rl.panel_artifacts import load_panel_bytes
from dgcc.rl.td3 import TD3Agent, TD3Config
from dgcc.tasks.domain import RewardConstants, SETTLE_MAX_STEPS, p1_rope_params
from dgcc.tasks.episode import BatchedEpisodeRunner, EpisodeConfig
from dgcc.tasks.t2 import load_t2_payload_bytes, load_t2_split_payload


def read_regular_bytes(path: Path) -> bytes:
    """Read one no-follow regular-file descriptor into an immutable snapshot."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise RuntimeError(f"unsafe diagnostic input {path}: {exc}") from exc
    with os.fdopen(fd, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise RuntimeError(f"diagnostic input is not a regular file: {path}")
        return handle.read()


def digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def digest_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def model_digest(agent: TD3Agent) -> str:
    digest = hashlib.sha256()
    for name in ("encoder", "critic", "actor", "encoder_target", "critic_target", "actor_target"):
        for key, tensor in sorted(getattr(agent, name).state_dict().items()):
            digest.update(f"{name}.{key}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
            digest.update(tensor.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def atomic_npz(path: Path, **payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        os.unlink(temporary)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def env_kwargs(config: dict[str, Any], n_envs: int) -> dict[str, Any]:
    sim = config.get("sim", {})
    return {
        "n_envs": n_envs, "dt": float(sim.get("dt", 1.0e-3)),
        "substeps": int(sim.get("substeps", 5)), "rod_damping": float(sim.get("rod_damping", 10.0)),
        "rod_angular_damping": float(sim.get("rod_angular_damping", 5.0)),
        "initial_settle_steps": int(sim.get("initial_settle_steps", 0)),
        "reset_settle_max_steps": int(sim.get("reset_settle_max_steps", SETTLE_MAX_STEPS)),
        "move_step_size": float(sim.get("move_step_size", 0.03)),
        "move_hold_steps": int(sim.get("move_hold_steps", 0)),
        "grasp_realism": bool(sim.get("grasp_realism", True)),
    }


def validation_batches(
    config: dict[str, Any], n_envs: int, val_pairs: list[tuple[str, Any]]
) -> list[list[tuple[str, Any]]]:
    per_goal = int(config.get("eval", {}).get("t2_episodes_per_goal", 2))
    goals = [(goal_id, goal) for goal_id, goal in val_pairs for _ in range(per_goal)]
    if not goals:
        raise RuntimeError("T2 validation split is empty")
    return [goals[i:i + n_envs] for i in range(0, len(goals), n_envs)]


def row_hashes(X: np.ndarray, G: np.ndarray, goal_ids: tuple[str, ...], start: int) -> np.ndarray:
    return np.asarray([
        hashlib.sha256(
            np.ascontiguousarray(x).tobytes() + np.ascontiguousarray(g).tobytes()
            + goal_id.encode() + str(start).encode()
        ).hexdigest()
        for x, g, goal_id in zip(X, G, goal_ids)
    ], dtype=str)


def execute_branch(
    selector: str, agent: TD3Agent, config: dict[str, Any], request: dict[str, Any],
    val_pairs: list[tuple[str, Any]], env_factory: Callable[..., Any] = DLOLabEnv,
    runner_factory: Callable[..., Any] = BatchedEpisodeRunner,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_envs = min(int(config.get("run", {}).get("n_envs", 256)), len(val_pairs))
    if n_envs < 1:
        raise RuntimeError("T2 validation split is empty")
    selected, progress, starts, goal_ids, initial_hashes = [], [], [], [], []
    for batch_number, batch in enumerate(validation_batches(config, n_envs, val_pairs)):
        ids, goals = zip(*batch)
        env = env_factory(**env_kwargs(config, len(goals)))
        runner = runner_factory(env, p1_rope_params(), EpisodeConfig(
            reward=RewardConstants(**config.get("reward", {}))))
        start = int(request["development_episode_index_start"]) + batch_number
        runner.begin_episodes(seed=int(request["seed"]) + 500, episode_index=start, goals=list(goals))
        X = np.asarray(env.get_centerline_batch(), dtype=float)
        G = np.asarray([goal_curve(goal, p1_rope_params().length_m) for goal in goals], dtype=float)
        initial_hashes.append(row_hashes(X, G, ids, start))
        p, delta, lift = agent.select_actions(X, G, step=int(request["transition"]),
            total_budget=int(request["total_budget"]), rng=np.random.default_rng(int(request["seed"]) + batch_number),
            deterministic=True, selector_operator=selector)
        outcome = runner.step(p, delta, lift, rng=np.random.default_rng(int(request["seed"]) + 7000 + batch_number))
        selected.append(np.asarray(p, dtype=np.int64))
        progress.append(np.asarray(outcome["d_before"], dtype=float) - np.asarray(outcome["d_after"], dtype=float))
        starts.append(np.full(len(goals), start, dtype=np.int64))
        goal_ids.extend(ids)
    return (np.concatenate(selected), np.concatenate(progress), np.concatenate(starts),
            np.asarray(goal_ids, dtype=str), np.concatenate(initial_hashes))


def select_panel(
    agent: TD3Agent, panel_arrays: tuple[np.ndarray, np.ndarray, np.ndarray], request: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run selectors over the authenticated frozen panel without an environment rollout."""
    X, G, order = panel_arrays
    if not np.array_equal(np.sort(order), np.arange(len(order), dtype=order.dtype)):
        raise RuntimeError("panel order is not a permutation of panel states")
    X, G = X[order], G[order]
    selections = []
    for selector in ("q1", "qmin"):
        p, _delta, _lift = agent.select_actions(
            X, G, step=int(request["transition"]), total_budget=int(request["total_budget"]),
            rng=np.random.default_rng(int(request["seed"]) + 9001), deterministic=True,
            selector_operator=selector,
        )
        selections.append(np.asarray(p, dtype=np.int64))
    return selections[0], selections[1], np.asarray(order, dtype=np.int64)
def run_counterfactual(
    request: dict[str, Any], *, agent_factory: Callable[..., TD3Agent] = TD3Agent,
    env_factory: Callable[..., Any] = DLOLabEnv, runner_factory: Callable[..., Any] = BatchedEpisodeRunner,
) -> dict[str, object]:
    required = {"config_snapshot", "config_sha256", "checkpoint", "checkpoint_sha256", "panel",
                "panel_sha256", "panel_artifact_sha256", "panel_metadata",
                "panel_metadata_sha256", "development_split_path",
                "development_split_sha256", "development_split_role", "seed",
                "transition", "total_budget", "eval_ordinal",
                "development_episode_index_start", "validation_goal_ids", "output"}
    missing = required.difference(request)
    if missing:
        raise RuntimeError(f"counterfactual request missing fields: {sorted(missing)}")
    if request["development_split_role"] != "development_t2_split":
        raise RuntimeError("counterfactual request has unauthorized development split role")
    config = request["config_snapshot"]
    if digest_json(config) != request["config_sha256"]:
        raise RuntimeError("counterfactual config hash mismatch")
    checkpoint = Path(request["checkpoint"])
    split_path = Path(request["development_split_path"])
    panel_path = Path(request["panel"])
    checkpoint_bytes = read_regular_bytes(checkpoint)
    split_bytes = read_regular_bytes(split_path)
    panel_bytes = read_regular_bytes(panel_path)
    metadata_bytes = read_regular_bytes(Path(request["panel_metadata"]))
    if digest_bytes(checkpoint_bytes) != request["checkpoint_sha256"]:
        raise RuntimeError("counterfactual checkpoint hash mismatch")
    if digest_bytes(split_bytes) != request["development_split_sha256"]:
        raise RuntimeError("counterfactual development split hash mismatch")
    if digest_bytes(metadata_bytes) != request["panel_metadata_sha256"]:
        raise RuntimeError("counterfactual panel metadata hash mismatch")
    panel, panel_arrays = load_panel_bytes(
        panel_bytes, metadata_bytes, path=panel_path,
        expected_canonical_sha256=request["panel_sha256"],
        expected_artifact_sha256=request["panel_artifact_sha256"],
    )
    split_payload = load_t2_payload_bytes(split_bytes)
    val_pairs = load_t2_split_payload("val", split_payload)
    expected_goal_ids = [str(goal_id) for goal_id, _ in val_pairs for _ in range(
        int(config.get("eval", {}).get("t2_episodes_per_goal", 2)))]
    if request["validation_goal_ids"] != expected_goal_ids:
        raise RuntimeError("counterfactual request validation goal sequence mismatch")
    if str(config.get("task")) != "t2":
        raise RuntimeError("counterfactual diagnostic requires T2 validation goals")

    py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    generator = np.random.default_rng(int(request["seed"]) + 9501)
    generator_state = copy.deepcopy(generator.bit_generator.state)
    rng_sha256_before = digest_json({
        "python": repr(py_state), "numpy": repr(np_state),
        "generator": generator_state, "torch_cpu": torch_state.tolist(),
    })
    try:
        torch.manual_seed(int(request["seed"]))
        agent = agent_factory(TD3Config(**config.get("td3", {})), device="cpu",
            reward_constants=RewardConstants(**config.get("reward", {})))
        agent._load_evaluation_checkpoint_payload(
            torch.load(BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
        )
        for module_name in ("encoder", "critic", "actor", "encoder_target", "critic_target", "actor_target"):
            getattr(agent, module_name).eval()
        model_before = model_digest(agent)
        panel_q1, panel_qmin, panel_order = select_panel(agent, panel_arrays, request)
        q1_p, q1_progress, q1_starts, q1_goal_ids, q1_hashes = execute_branch(
            "q1", agent, config, request, val_pairs, env_factory, runner_factory)
        qmin_p, qmin_progress, qmin_starts, qmin_goal_ids, qmin_hashes = execute_branch(
            "qmin", agent, config, request, val_pairs, env_factory, runner_factory)
        model_after = model_digest(agent)
        if model_before != model_after:
            raise RuntimeError("counterfactual worker mutated online model state")
        if not (np.array_equal(q1_starts, qmin_starts) and np.array_equal(q1_goal_ids, qmin_goal_ids)
                and np.array_equal(q1_hashes, qmin_hashes)):
            raise RuntimeError("selector branches did not use identical initial-state provenance")
        if not (np.isfinite(q1_progress).all() and np.isfinite(qmin_progress).all()):
            raise RuntimeError("counterfactual branch produced non-finite progress")
        return {
            "panel_sha256": panel.canonical_sha256, "checkpoint_sha256": request["checkpoint_sha256"],
            "model_sha256_before": model_before, "model_sha256_after": model_after,
            "panel_selector_agreement": float(np.mean(panel_q1 == panel_qmin)),
            "panel_q1_selected": panel_q1, "panel_qmin_selected": panel_qmin,
            "panel_order": panel_order,
            "rollout_realized_progress_difference_mean": float(np.mean(q1_progress - qmin_progress)),
            "rollout_q1_realized_progress_mean": float(np.mean(q1_progress)),
            "rollout_qmin_realized_progress_mean": float(np.mean(qmin_progress)),
            "rollout_q1_selected": q1_p, "rollout_qmin_selected": qmin_p,
            "rollout_validation_goal_ids": q1_goal_ids,
            "rollout_episode_starts": q1_starts,
            "rollout_seed": np.int64(request["seed"]),
            "rollout_config_sha256": request["config_sha256"],
            "rollout_development_episode_index_start": np.int64(
                request["development_episode_index_start"]),
            "rollout_initial_state_sha256": q1_hashes,
            "rollout_provenance_sha256": np.asarray([
                hashlib.sha256((goal_id + ":" + str(start)).encode()).hexdigest()
                for goal_id, start in zip(q1_goal_ids, q1_starts)
            ], dtype=str),
            "development_split_path": str(split_path.resolve()),
            "development_split_sha256": request["development_split_sha256"],
            "development_split_role": request["development_split_role"],
            "rollout_checkpoint_sha256": request["checkpoint_sha256"],
            "rollout_panel_sha256": request["panel_sha256"],
            "rollout_transition": np.int64(request["transition"]),
            "rollout_eval_ordinal": np.int64(request["eval_ordinal"]),
            "rollout_total_budget": np.int64(request["total_budget"]),
            "panel_artifact_sha256": request["panel_artifact_sha256"],
            "panel_metadata_sha256": request["panel_metadata_sha256"],
        }
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        generator.bit_generator.state = generator_state
        rng_sha256_after = digest_json({
            "python": repr(random.getstate()), "numpy": repr(np.random.get_state()),
            "generator": generator.bit_generator.state, "torch_cpu": torch.get_rng_state().tolist(),
        })
        if rng_sha256_after != rng_sha256_before:
            raise RuntimeError("counterfactual worker failed to restore RNG state")


def main(request_path: str, expected_request_sha256: str) -> int:
    request_bytes = read_regular_bytes(Path(request_path))
    if digest_bytes(request_bytes) != expected_request_sha256:
        raise RuntimeError("counterfactual request hash mismatch")
    request = json.loads(request_bytes)
    payload = run_counterfactual(request)
    payload.update(
        transition=np.int64(request["transition"]),
        eval_ordinal=np.int64(request.get("eval_ordinal", -1)),
        request_sha256=expected_request_sha256,
    )
    atomic_npz(Path(request["output"]), **payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))