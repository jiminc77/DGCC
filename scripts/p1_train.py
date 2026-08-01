"""P1 training driver (M2 smoke, M3 T1 runs, M4 T2 runs).

Implements the §7 protocol on DLOLabEnv: batched collection through the
episode layer (immutable settle budget), UTD=1 updates, deterministic eval
every 25k transitions (T1: 100 episodes; T2: 50 validation goals x 2),
checkpoints every 25k plus best-on-eval, §8 diagnostics with 25k auto-plots,
env-level NaN covenant with P0-pattern full scene rebuild escalation, and the
training-level NaN halt (preserve last checkpoint + factual report).

Usage:
    uv run python scripts/p1_train.py --config configs/p1_t1_a.yaml --seed 0 \
        --run-tag t1a_smoke_s0 --total-override 50000
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dgcc.envs.dlolab import DLOLabEnv
from dgcc.goals.dual_goal import goal_curve
from dgcc.models.networks import goal_residual_flips
from dgcc.rl.diagnostics import DiagnosticsLogger
from dgcc.rl.evaluation import evaluate_episodes
from dgcc.rl.replay import ReplayBuffer
from dgcc.rl.panel_artifacts import PanelArtifact, load_panel, persist_panel
from dgcc.rl.selection import SelectionSnapshot, compare_selection_snapshots
from dgcc.rl.td3 import TD3Agent, TD3Config, TrainingNaNError
from dgcc.tasks.domain import (
    P1_LENGTH_M,
    P1_N_SEGMENTS,
    RewardConstants,
    SETTLE_MAX_STEPS,
    p1_rope_params,
)
from dgcc.tasks.episode import BatchedEpisodeRunner, EpisodeConfig, is_nonfinite_error
from dgcc.tasks.t1 import sample_t1_goal
from dgcc.tasks.t2 import (
    default_split_path,
    load_t2_payload_bytes,
    load_t2_split_payload,
)
from dgcc.utils.meta import get_git_commit_hash
from dgcc.logging.attempt_registry import AttemptRegistry, sha256_file
from dgcc.logging.asset_firewall import (
    AssetAccessError,
    AssetFirewall,
    load_launch_asset_manifest,
    persist_launch_receipts,
    read_launch_asset_snapshot,
)

T1_TASKS = ("t1a_straighten", "t1b_single_bend", "t1c_endpoint_reposition")


def expand_t2_validation_pairs(
    val_pairs: list[tuple[str, Any]], episodes_per_goal: Any
) -> tuple[list[str], list[Any]]:
    if type(episodes_per_goal) is not int or episodes_per_goal < 1:
        raise ValueError("eval.t2_episodes_per_goal must be a positive integer")
    labels = [
        label for label, _ in val_pairs for _ in range(episodes_per_goal)
    ]
    goals = [
        goal for _, goal in val_pairs for _ in range(episodes_per_goal)
    ]
    return labels, goals

# Env-stability operational limits (M3R gate verdict gate-m3r-reconvene-20260710,
# choice D follow-ups — env/driver layer only; training code, hyperparameters,
# reward constants and covenant thresholds unchanged):
#   (a) a discard storm exceeding DISCARD_STORM_REBUILD_AFTER *consecutive*
#       discarded rounds escalates to a forced full scene rebuild (livelock
#       exit — m3r_t1a_s2 stalled 10.5 h at 272 discards with rebuild=0),
#   (b) the full-rebuild limit rises 5 -> 8 and the freshest agent state is
#       checkpointed before a rebuild-limit crash (m3r_t1a_s1 lost ~12k
#       transitions of progress past its last periodic checkpoint).
MAX_FULL_REBUILDS = 8
DISCARD_STORM_REBUILD_AFTER = 10

# M4-prep reproducibility fixes (gate verdict gate-m3r-reconvene-2-20260713,
# choice B follow-up 3 — batch boundary, forensics outputs/reports/
# p1_m3r_t1a_s2_forensics.md):
#   F-a  All RNGs are seeded BEFORE TD3Agent construction so same-seed
#        processes start from identical weights; the initial-weights hash is
#        captured immediately after construction and persisted (never
#        recomputed). This formally breaks M3R<->M4 same-seed comparability
#        (M3R runs seeded Torch after construction). No bitwise-CUDA-
#        determinism claim is added (extra determinism flags NOT approved).
#   F-b  Deterministic-eval episode indexing uses a one-based successful-eval
#        ordinal instead of the rebuild-coupled self.episode_index, so eval
#        curve seeds are identical across attempts regardless of rebuild
#        history. First eval -> 90_001.
#   P1b  Smoke-only dense round logging behind P1_LOG_EVERY_ROUND=1 (approved
#        at intent reconciliation 2026-07-15); main lanes launch with
#        `env -u P1_LOG_EVERY_ROUND` and assert zero roundlog lines post-run.


def initial_weights_sha256(agent: TD3Agent) -> str:
    """F-a: deterministic digest of all module state dicts (incl. targets).

    Sorted keys; each tensor contributes name, dtype, shape and its
    contiguous CPU bytes.
    """
    digest = hashlib.sha256()
    for name, module in (
        ("encoder", agent.encoder),
        ("critic", agent.critic),
        ("actor", agent.actor),
        ("encoder_target", agent.encoder_target),
        ("critic_target", agent.critic_target),
        ("actor_target", agent.actor_target),
    ):
        state = module.state_dict()
        for key in sorted(state):
            tensor = state[key]
            digest.update(f"{name}.{key}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
            digest.update(tensor.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def eval_episode_index_start(completed_evals: int) -> int:
    """F-b: rebuild-independent eval episode index (one-based ordinal).

    ``completed_evals`` counts previously *successful* evals; the first eval
    therefore starts at 90_001 for every attempt of a given seed.
    """
    return 90_000 + completed_evals + 1


def roundlog_line(transitions: int, count: int, collect_s: float, update_s: float) -> str | None:
    """P1b: dense per-round emission, active only when P1_LOG_EVERY_ROUND == "1"."""
    if os.environ.get("P1_LOG_EVERY_ROUND") != "1":
        return None
    return (
        f"roundlog transitions={transitions} count={count} "
        f"collect_s={collect_s:.1f} update_s={update_s:.1f}"
    )


class Tee:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def env_kwargs(config: dict[str, Any], n_envs: int) -> dict[str, Any]:
    sim = config.get("sim", {})
    kwargs = {
        "n_envs": int(n_envs),
        "dt": float(sim.get("dt", 1.0e-3)),
        "substeps": int(sim.get("substeps", 5)),
        "rod_damping": float(sim.get("rod_damping", 10.0)),
        "rod_angular_damping": float(sim.get("rod_angular_damping", 5.0)),
        "initial_settle_steps": int(sim.get("initial_settle_steps", 0)),
        "reset_settle_max_steps": int(sim.get("reset_settle_max_steps", SETTLE_MAX_STEPS)),
        "grasp_realism": bool(sim.get("grasp_realism", True)),
    }
    if "move_v_max" in sim:
        # R8 (env-correction Rev 3): quasi-static primitive configuration.
        kwargs["move_v_max"] = float(sim["move_v_max"])
        kwargs["move_hold_max_steps"] = int(sim.get("move_hold_max_steps", 2000))
    else:
        # Deprecated legacy keys (one release; pre-correction semantics).
        kwargs["move_step_size"] = float(sim.get("move_step_size", 0.03))
        kwargs["move_hold_steps"] = int(sim.get("move_hold_steps", 0))
    return kwargs


class TrainingRun:
    def __init__(
        self,
        args: argparse.Namespace,
        registry: AttemptRegistry | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.registry = registry
        self.asset_firewall: AssetFirewall | None = None
        self.asset_manifest: dict[str, Any] | None = None
        manifest_path = getattr(args, "asset_manifest", None)
        expected_manifest_sha256 = getattr(args, "expected_asset_manifest_sha256", None)
        if manifest_path is not None or expected_manifest_sha256 is not None:
            if manifest_path is None or expected_manifest_sha256 is None:
                raise AssetAccessError("asset manifest and independent manifest SHA-256 are both required")
            audit_path = (
                registry.attempt_path / "reports" / "protected_asset_audit.jsonl"
                if registry is not None else Path("outputs") / "protected_asset_audit.jsonl"
            )
            self.asset_firewall, self.asset_manifest = load_launch_asset_manifest(
                manifest_path, expected_manifest_sha256, audit_path
            )
            if config is None:
                _, config_bytes = self.asset_firewall.read_bytes(
                    args.config,
                    operation="config-load",
                    required_role="config",
                )
                config = yaml.safe_load(config_bytes)
        elif config is None:
            config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        self.config = config
        self.seed = int(args.seed)
        self.task = str(self.config["task"])
        self.run_tag = args.run_tag or f"{self.task}_s{self.seed}"
        run_cfg = self.config.get("run", {})
        self.total = int(args.total_override or run_cfg.get("total_transitions", 100_000))
        self.n_envs = int(run_cfg.get("n_envs", 256))
        self.eval_every = int(run_cfg.get("eval_every_transitions", 25_000))
        self.max_full_rebuilds = int(run_cfg.get("max_full_rebuilds", MAX_FULL_REBUILDS))
        self.discard_storm_rebuild_after = int(
            run_cfg.get("discard_storm_rebuild_after", DISCARD_STORM_REBUILD_AFTER)
        )
        self.device = args.device
        self.params = p1_rope_params()

        reward_cfg = self.config.get("reward", {})
        self.episode_config = EpisodeConfig(
            reward=RewardConstants(
                alpha=float(reward_cfg.get("alpha", 10.0)),
                c_step=float(reward_cfg.get("c_step", 0.1)),
                r_succ=float(reward_cfg.get("r_succ", 5.0)),
            )
        )
        td3_cfg = dict(self.config.get("td3", {}))
        self.agent_config = TD3Config(**{k: v for k, v in td3_cfg.items()})
        self.registry = registry
        # F-a (gate-m3r-reconvene-2-20260713 follow-up 3): seed ALL RNGs
        # strictly BEFORE agent construction so same-seed processes start
        # from identical weights.
        self.rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        self.agent = self.create_agent()
        # F-a: hash captured immediately after construction, before any
        # training; persisted via save_run_summary, never recomputed.
        self.initial_weights_sha256 = initial_weights_sha256(self.agent)
        if self.registry is not None:
            self.registry.initialized(self.initial_weights_sha256)
        self.buffer = ReplayBuffer(self.agent_config.replay_capacity)
        self.diag = DiagnosticsLogger(self.registry.attempt_id if self.registry is not None else self.run_tag)
        # F-b: one-based successful-eval ordinal (rebuild-independent).
        self._eval_ordinal = 0

        self.output_dir = self.registry.attempt_path if self.registry is not None else Path("outputs")
        self.models_dir = self.output_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.transitions = 0
        self.episode_index = 0
        self.full_rebuilds = 0
        self._consecutive_discards = 0
        self.last_checkpoint: Path | None = None
        self.best_success = -1.0
        self.eval_history: list[dict[str, Any]] = []
        self.halt_reason: str | None = None
        self._selection_panel_limit = int(
            self.config.get("eval", {}).get("selection_panel_states", 300)
        )
        if self._selection_panel_limit < 1:
            raise ValueError("eval.selection_panel_states must be positive")
        self._selection_panel_X: list[np.ndarray] = []
        self._selection_panel_G: list[np.ndarray] = []
        self._selection_panel_frozen_at: int | None = None
        self._previous_selection_snapshot: SelectionSnapshot | None = None
        self.panel_path = self.output_dir / "artifacts" / "v2_development_panel.npz"
        self.panel_artifact: PanelArtifact | None = None
        self._selection_snapshot_path = self.output_dir / "artifacts" / "v2_previous_selection_snapshot.npz"
        self._diagnostic_failure_path = self.output_dir / "metrics" / "counterfactual_failures.jsonl"
        self.counterfactual_diagnostic: dict[str, object] = {
            "status": "failed" if self.task == "t2" else "not_applicable",
        }
        expected_panel_sha = self.config.get("eval", {}).get("expected_panel_sha256")
        if self.panel_path.exists():
            self.panel_artifact = load_panel(
                self.panel_path, expected_canonical_sha256=expected_panel_sha
            )
            with np.load(self.panel_path, allow_pickle=False) as panel:
                self._selection_panel_X = [row.copy() for row in panel["X"]]
                self._selection_panel_G = [row.copy() for row in panel["G"]]
                self._selection_panel_frozen_at = int(panel["transition"])
        self._load_previous_selection_snapshot()
        self._prev_goal_flip = np.full(self.n_envs, -1, dtype=np.int8)
        self._episode_flip_transitions = np.zeros(self.n_envs, dtype=int)
        self._episode_flip_observations = np.zeros(self.n_envs, dtype=int)

        if self.task == "t2":
            split_path = default_split_path().resolve()
            if self.asset_firewall is not None:
                split_path, split_bytes = self.asset_firewall.read_bytes(
                    split_path,
                    operation="t2-split-load",
                    required_role="t2_split",
                )
            else:
                split_bytes = split_path.read_bytes()
            self.development_split_path = split_path
            self.development_split_sha256 = hashlib.sha256(
                split_bytes
            ).hexdigest()
            self.development_split_role = "development_t2_split"
            self._development_split_payload = load_t2_payload_bytes(split_bytes)
            self.train_goals = [
                goal
                for _, goal in load_t2_split_payload(
                    "train", self._development_split_payload
                )
            ]
            val_pairs = load_t2_split_payload(
                "val", self._development_split_payload
            )
            episodes_per_goal = self.config.get("eval", {}).get(
                "t2_episodes_per_goal", 2
            )
            self.val_labels, self.val_goals = expand_t2_validation_pairs(
                val_pairs, episodes_per_goal
            )
        elif self.task in T1_TASKS:
            self.train_goals = None
            self.val_goals = None
            self.val_labels = None
            self.development_split_path = None
            self.development_split_sha256 = None
            self.development_split_role = None
            self._development_split_payload = None
        else:
            raise ValueError(f"unknown task {self.task!r}")

        self.env: DLOLabEnv | None = None
        self.runner: BatchedEpisodeRunner | None = None
        self.goal_curves: np.ndarray | None = None

    def create_agent(self) -> TD3Agent:
        return TD3Agent(
            self.agent_config,
            device=self.device,
            reward_constants=self.episode_config.reward,
        )
    def _load_previous_selection_snapshot(self) -> None:
        if not self._selection_snapshot_path.exists():
            return
        try:
            with np.load(self._selection_snapshot_path, allow_pickle=False) as data:
                self._previous_selection_snapshot = SelectionSnapshot(
                    **{key: torch.from_numpy(data[key]) for key in (
                        "q1_selected", "qmin_selected", "weights", "top8",
                        "contact_histogram", "contact_histogram_counts",
                    )}
                )
        except (OSError, KeyError, ValueError) as exc:
            raise RuntimeError(f"invalid previous selection snapshot: {exc}") from exc

    def _persist_previous_selection_snapshot(self, snapshot: SelectionSnapshot) -> None:
        snapshot = snapshot.cpu()
        self._selection_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._selection_snapshot_path.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            np.savez(handle, **{key: getattr(snapshot, key).numpy() for key in (
                "q1_selected", "qmin_selected", "weights", "top8",
                "contact_histogram", "contact_histogram_counts",
            )})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self._selection_snapshot_path)
        directory = os.open(self._selection_snapshot_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def _persist_panel(self) -> None:
        self.panel_artifact = persist_panel(
            self.panel_path, X=np.asarray(self._selection_panel_X),
            G=np.asarray(self._selection_panel_G),
            order=np.arange(len(self._selection_panel_X), dtype=np.int64),
            seed=self.seed, transition=self.transitions, eval_ordinal=self._eval_ordinal,
        )
        expected = self.config.get("eval", {}).get("expected_panel_sha256")
        if expected is not None and expected != self.panel_artifact.canonical_sha256:
            raise RuntimeError("paired arm panel SHA mismatch")

    def _record_counterfactual_failure(self, error: object) -> None:
        self._diagnostic_failure_path.parent.mkdir(parents=True, exist_ok=True)
        with self._diagnostic_failure_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"generated_at": utc_now(), "transitions": self.transitions,
                "error": f"{type(error).__name__}: {error}"}) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        directory = os.open(self._diagnostic_failure_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def _validate_counterfactual_output(self, output: Path, request: dict[str, object]) -> None:
        # Charter amendment 3: the per-validation 9.4 requirement is the
        # rollout-free selector comparison over the frozen panel. The
        # realized-progress rollout is a post-winner, development-only
        # experiment, so its fields are validated only when present.
        required = {
            "panel_selector_agreement", "panel_disagreement_fraction",
            "panel_q1_selected", "panel_qmin_selected", "panel_order",
            "development_split_path", "development_split_sha256",
            "development_split_role", "panel_sha256", "panel_artifact_sha256",
            "panel_metadata_sha256", "request_sha256", "checkpoint_sha256",
            "model_sha256_before", "model_sha256_after", "transition", "eval_ordinal",
        }
        rollout_required = {
            "rollout_realized_progress_difference_mean", "rollout_q1_realized_progress_mean",
            "rollout_qmin_realized_progress_mean", "rollout_q1_selected", "rollout_qmin_selected",
            "rollout_validation_goal_ids", "rollout_episode_starts", "rollout_initial_state_sha256",
            "rollout_provenance_sha256", "rollout_seed", "rollout_config_sha256",
            "rollout_development_episode_index_start", "rollout_checkpoint_sha256",
            "rollout_panel_sha256", "rollout_transition", "rollout_eval_ordinal",
            "rollout_total_budget",
        }
        try:
            with np.load(output, allow_pickle=False) as data:
                present = set(data.files)
                has_rollout = bool(present & rollout_required)
                closed = required | rollout_required if has_rollout else required
                if present != closed:
                    raise RuntimeError("counterfactual output fields are not closed")
                for name in data.files:
                    value = data[name]
                    if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
                        raise RuntimeError(f"counterfactual output has non-finite {name}")
                if has_rollout:
                    row_names = ("rollout_q1_selected", "rollout_qmin_selected",
                        "rollout_validation_goal_ids", "rollout_episode_starts",
                        "rollout_initial_state_sha256", "rollout_provenance_sha256")
                    cardinality = len(data["rollout_validation_goal_ids"])
                    if cardinality < 1 or any(len(data[name]) != cardinality for name in row_names):
                        raise RuntimeError("counterfactual output row cardinalities differ")
                    if not np.array_equal(data["rollout_validation_goal_ids"],
                                          np.asarray(request["validation_goal_ids"], dtype=str)):
                        raise RuntimeError("counterfactual output goal sequence mismatch")
                    if not np.array_equal(data["rollout_episode_starts"], np.sort(data["rollout_episode_starts"])):
                        raise RuntimeError("counterfactual output episode starts are not ordered")
                if data["panel_q1_selected"].shape != data["panel_qmin_selected"].shape:
                    raise RuntimeError("counterfactual panel selector cardinalities differ")
                if not np.array_equal(np.sort(data["panel_order"]), np.arange(len(data["panel_order"]))):
                    raise RuntimeError("counterfactual output panel order is not a permutation")
                expected = {
                    "development_split_path": str(Path(request["development_split_path"]).resolve()),
                    "development_split_sha256": request["development_split_sha256"],
                    "development_split_role": request["development_split_role"],
                    "checkpoint_sha256": request["checkpoint_sha256"],
                    "panel_sha256": request["panel_sha256"],
                    "panel_artifact_sha256": request["panel_artifact_sha256"],
                    "panel_metadata_sha256": request["panel_metadata_sha256"],
                    "request_sha256": request["request_sha256"],
                    "transition": int(request["transition"]),
                    "eval_ordinal": int(request["eval_ordinal"]),
                }
                if has_rollout:
                    expected.update({
                        "rollout_seed": int(request["seed"]),
                        "rollout_config_sha256": request["config_sha256"],
                        "rollout_development_episode_index_start": int(
                            request["development_episode_index_start"]),
                        "rollout_checkpoint_sha256": request["checkpoint_sha256"],
                        "rollout_panel_sha256": request["panel_sha256"],
                        "rollout_transition": int(request["transition"]),
                        "rollout_eval_ordinal": int(request["eval_ordinal"]),
                        "rollout_total_budget": int(request["total_budget"]),
                    })
                for name, expected_value in expected.items():
                    value = data[name]
                    actual = value.item() if value.ndim == 0 else str(value)
                    if actual != expected_value:
                        raise RuntimeError(f"counterfactual output identity mismatch: {name}")
                if str(data["model_sha256_before"].item()) != str(data["model_sha256_after"].item()):
                    raise RuntimeError("counterfactual worker model immutability check failed")
        except (OSError, ValueError, KeyError) as error:
            raise RuntimeError(f"invalid counterfactual output: {error}") from error

    def _run_counterfactual_worker_unsafe(self, checkpoint: Path, development_episode_index_start: int) -> None:
        if self.task != "t2":
            self.counterfactual_diagnostic = {"status": "not_applicable"}
            return
        if self.panel_artifact is None:
            raise RuntimeError("V2 counterfactual diagnostic requires a frozen panel")
        request = self.output_dir / "diagnostics" / f"counterfactual_{self.transitions:07d}.json"
        request.parent.mkdir(parents=True, exist_ok=True)
        config_snapshot = self.config
        if self.development_split_path is None or self.development_split_sha256 is None:
            raise RuntimeError("counterfactual diagnostic lacks authorized development split identity")
        panel_metadata_path = self.panel_path.with_suffix(
            self.panel_path.suffix + ".json"
        )
        request_payload = {
            "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256_file(checkpoint),
            "panel": str(self.panel_path.resolve()), "panel_sha256": self.panel_artifact.canonical_sha256,
            "panel_artifact_sha256": self.panel_artifact.artifact_sha256,
            "panel_metadata": str(panel_metadata_path.resolve()),
            "panel_metadata_sha256": sha256_file(panel_metadata_path),
            "development_split_path": str(self.development_split_path),
            "development_split_sha256": self.development_split_sha256,
            "development_split_role": self.development_split_role,
            "output": str(request.with_suffix(".npz").resolve()), "seed": self.seed,
            "transition": self.transitions, "total_budget": self.total, "eval_ordinal": self._eval_ordinal,
            "development_episode_index_start": development_episode_index_start,
            "config_snapshot": config_snapshot,
            "config_sha256": hashlib.sha256(json.dumps(
                config_snapshot, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "validation_goal_ids": list(self.val_labels),
        }
        request_bytes = (
            json.dumps(request_payload, sort_keys=True, separators=(",", ":")).encode()
            + b"\n"
        )
        request_sha256 = hashlib.sha256(request_bytes).hexdigest()
        fd, temporary_name = tempfile.mkstemp(prefix=f".{request.name}.", suffix=".tmp", dir=request.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(request_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary, request)
            temporary.unlink()
            directory = os.open(request.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        output = request.with_suffix(".npz")
        try:
            subprocess.run([
                sys.executable,
                str(Path(__file__).with_name("v2_counterfactual_worker.py")),
                str(request),
                request_sha256,
            ], check=True,
                timeout=float(self.config.get("eval", {}).get("counterfactual_timeout_s", 120)),
                capture_output=True, text=True)
            self._validate_counterfactual_output(
                output, {**request_payload, "request_sha256": request_sha256}
            )
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            raise RuntimeError(f"counterfactual worker failed: {exc}") from exc
        self.counterfactual_diagnostic = {
            "status": "completed", "output": str(output),
            "panel_sha256": self.panel_artifact.canonical_sha256,
        }

    def _record_counterfactual_failure_safely(self, error: Exception) -> None:
        self.counterfactual_diagnostic = {
            "status": "failed", "downstream_ready": False,
            "error": f"{type(error).__name__}: {error}",
        }
        try:
            self._record_counterfactual_failure(error)
        except Exception:
            pass
        try:
            self.save_run_summary()
        except Exception:
            pass

    def _run_counterfactual_worker(self, checkpoint: Path, development_episode_index_start: int) -> None:
        """Keep all ordinary diagnostic failures out of training control flow."""
        if self.task != "t2":
            self.counterfactual_diagnostic = {"status": "not_applicable"}
            return
        try:
            self._run_counterfactual_worker_unsafe(checkpoint, development_episode_index_start)
        except Exception as exc:
            self._record_counterfactual_failure_safely(exc)
            return

    # ------------------------------------------------------------------

    def build_scene(self) -> None:
        if self.env is not None:
            self.runner = None
            self.env = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.env = DLOLabEnv(**env_kwargs(self.config, self.n_envs))
        self.env.reset(
            self.params,
            init_shape="straight",
            seed=self.seed + 10_000 * (self.full_rebuilds + 1),
        )
        if not self.env.supports_per_env_grasp():
            raise RuntimeError("per-env grasp hooks unavailable")
        self.runner = BatchedEpisodeRunner(self.env, self.params, self.episode_config)
        self.begin_training_episodes()

    def _goal_fn(self, env_idx: int, x: np.ndarray, goal_rng: np.random.Generator):
        return sample_t1_goal(self.task, x, goal_rng)

    def begin_training_episodes(self) -> None:
        assert self.runner is not None
        self.episode_index += 1
        if self.task == "t2":
            self.runner.begin_episodes(
                seed=self.seed,
                episode_index=self.episode_index,
                goal_pool=self.train_goals,
                auto_reset=True,
            )
        else:
            self.runner.begin_episodes(
                seed=self.seed,
                episode_index=self.episode_index,
                goal_fn=self._goal_fn,
                auto_reset=True,
            )
        self.refresh_goal_curves()
        self._reset_flip_tracking()

    def refresh_goal_curves(self) -> None:
        self.goal_curves = np.stack(
            [goal_curve(g, P1_LENGTH_M) for g in self.runner.goals]
        )

    def _reset_flip_tracking(
        self,
        env_indices: np.ndarray | list[int] | None = None,
        *,
        prev_flip: np.ndarray | None = None,
        flip_transitions: np.ndarray | None = None,
        flip_observations: np.ndarray | None = None,
    ) -> None:
        prev = self._prev_goal_flip if prev_flip is None else prev_flip
        transitions = (
            self._episode_flip_transitions if flip_transitions is None else flip_transitions
        )
        observations = (
            self._episode_flip_observations if flip_observations is None else flip_observations
        )
        if env_indices is None:
            prev.fill(-1)
            transitions.fill(0)
            observations.fill(0)
            return
        indices = np.asarray(env_indices, dtype=int).reshape(-1)
        if indices.size == 0:
            return
        prev[indices] = -1
        transitions[indices] = 0
        observations[indices] = 0

    def _log_lift_flip_diagnostics(
        self,
        transitions: int,
        *,
        X_before: np.ndarray,
        goal_curves: np.ndarray,
        lift: list[str],
        active: np.ndarray,
        templates: list[str],
        phase: str,
        done: np.ndarray | None = None,
        prev_flip: np.ndarray | None = None,
        flip_transitions: np.ndarray | None = None,
        flip_observations: np.ndarray | None = None,
    ) -> np.ndarray:
        active_arr = np.asarray(active, dtype=bool)
        templates_arr = np.asarray(templates, dtype=object)
        self.diag.log_lift_dist(
            transitions,
            templates=templates_arr,
            lift=lift,
            active=active_arr,
            phase=phase,
        )

        flips = goal_residual_flips(X_before, goal_curves, P1_LENGTH_M).astype(np.int8)
        prev = self._prev_goal_flip if prev_flip is None else prev_flip
        episode_flips = (
            self._episode_flip_transitions if flip_transitions is None else flip_transitions
        )
        observations = (
            self._episode_flip_observations if flip_observations is None else flip_observations
        )

        tracked = active_arr & (prev >= 0)
        changed = tracked & (prev != flips)
        observations[active_arr] += 1
        episode_flips[changed] += 1
        prev[active_arr] = flips[active_arr]

        done_arr = np.zeros_like(active_arr, dtype=bool) if done is None else np.asarray(done, dtype=bool)
        rows: list[dict[str, float | int | str]] = []
        for template in sorted({str(value) for value in templates_arr[active_arr]}):
            template_mask = active_arr & (templates_arr == template)
            n_active = int(template_mask.sum())
            n_tracked = int((tracked & template_mask).sum())
            flip_count = int((changed & template_mask).sum())
            completed = template_mask & done_arr
            rates: list[float] = []
            for env_idx in np.flatnonzero(completed):
                denominator = max(1, int(observations[int(env_idx)]) - 1)
                rates.append(float(episode_flips[int(env_idx)] / denominator))
            rows.append(
                {
                    "template": template,
                    "flip_transitions": flip_count,
                    "n_active": n_active,
                    "n_tracked": n_tracked,
                    "active_transition_rate": float(flip_count / n_tracked)
                    if n_tracked
                    else float("nan"),
                    "completed_episodes": int(completed.sum()),
                    "episode_flicker_rate_mean": float(np.mean(rates))
                    if rates
                    else float("nan"),
                }
            )
        if rows:
            self.diag.log_flip_flicker(transitions, rows, phase=phase)
        if done is not None:
            self._reset_flip_tracking(
                np.flatnonzero(active_arr & done_arr),
                prev_flip=prev,
                flip_transitions=episode_flips,
                flip_observations=observations,
            )
        return flips
    # ------------------------------------------------------------------

    def _register_rebuild(self, *, context: str, error: object) -> bool:
        """Count a full-scene rebuild escalation.

        Returns True when the rebuild limit (verdict (b): 8) is exceeded; in
        that case the freshest agent state has already been checkpointed and
        the caller must raise (lane contract: non-halt crash, exit=1).
        """

        self.full_rebuilds += 1
        print(
            f"{context} rebuild={self.full_rebuilds} error={error} "
            f"action=full_scene_rebuild transitions={self.transitions}"
        )
        if self.full_rebuilds > self.max_full_rebuilds:
            self._preserve_crash_checkpoint()
            return True
        return False

    def _preserve_crash_checkpoint(self) -> None:
        """Preserve the latest agent state before a rebuild-limit crash (verdict (b))."""

        try:
            ckpt = self.agent.save_checkpoint(
                self.models_dir / f"ckpt_crash_{self.transitions:07d}.pt"
            )
            self.last_checkpoint = ckpt
            print(f"crash checkpoint preserved: {ckpt}")
        except Exception as exc:  # keep the original crash path alive
            print(f"crash checkpoint preservation failed: {exc}")
        self.diag.save_history()
        self.save_run_summary()
    # ------------------------------------------------------------------

    def collect_round(self) -> int:
        """One batched primitive + buffer insertion. Returns active count."""

        assert self.runner is not None and self.env is not None and self.goal_curves is not None

        X = self.env.get_centerline_batch()
        goal_curves_before = self.goal_curves.copy()
        templates_before = list(self.runner.init_shapes)
        p, delta, lift, info = self.agent.select_actions(
            X,
            goal_curves_before,
            step=self.transitions,
            total_budget=self.total,
            rng=self.rng,
            return_info=True,
        )
        self.diag.log_action_info(self.transitions, info["q1_candidates"])

        try:
            record = self.runner.step(p, delta, lift, rng=self.rng)
        except (FloatingPointError, ValueError, RuntimeError) as exc:
            if not is_nonfinite_error(exc):
                raise
            if self._register_rebuild(context="round_recovery", error=exc):
                raise
            self._consecutive_discards = 0
            self._reset_flip_tracking()
            self.build_scene()
            return 0
        if record.get("discarded"):
            bad_envs = record.get("bad_envs", np.flatnonzero(record["active"]))
            self._reset_flip_tracking(np.asarray(bad_envs, dtype=int))
            self.diag.log_nan_incidents(
                self.transitions,
                self.runner.nan_incidents,
                self.runner.magnitude_incidents,
            )
            self._consecutive_discards += 1
            print(
                f"transition batch discarded (NaN covenant): {record['reason']} "
                f"consecutive={self._consecutive_discards}"
            )
            if self._consecutive_discards > self.discard_storm_rebuild_after:
                storm = self._consecutive_discards
                self._consecutive_discards = 0
                if self._register_rebuild(
                    context="discard_storm_escalation",
                    error=f"consecutive_discarded_rounds={storm}",
                ):
                    raise FloatingPointError(
                        f"rebuild limit ({self.max_full_rebuilds}) exceeded during "
                        f"discard-storm escalation (consecutive discarded rounds={storm})"
                    )
                self._reset_flip_tracking()
                self.build_scene()
                return 0
            self.refresh_goal_curves()
            return 0

        active = record["active"]
        self._consecutive_discards = 0
        count = int(active.sum())
        next_transitions = self.transitions + count
        if count:
            self._log_lift_flip_diagnostics(
                next_transitions,
                X_before=X,
                goal_curves=goal_curves_before,
                lift=lift,
                active=active,
                templates=templates_before,
                phase="collect",
                done=record["done"],
            )
            refresh_reset = np.flatnonzero(
                active & ~record["done"] & (self.runner.t < record["t"])
            )
            if refresh_reset.size:
                self._reset_flip_tracking(refresh_reset)
            self.diag.log_step_d(next_transitions, record["d_after"][active], phase="collect")
            self.buffer.add_batch(
                X_before=record["X_before"][active],
                X_after=record["X_after"][active],
                goal_curve=goal_curves_before[active],
                p=np.asarray(p)[active],
                delta=np.asarray(delta)[active],
                lift=np.asarray([1 if v == "high" else 0 for v in lift])[active],
                reward=record["reward"][active],
                done=record["done"][active],
                truncated=record["truncated"][active],
            )
            self.transitions = next_transitions
        self.diag.log_nan_incidents(
            self.transitions,
            self.runner.nan_incidents,
            self.runner.magnitude_incidents,
        )
        # Auto-reset may have refreshed goals for finished envs.
        self.refresh_goal_curves()
        return count

    def train_updates(self, n_updates: int) -> None:
        if self.buffer.size < self.agent_config.warmup_transitions:
            return
        for i in range(n_updates * self.agent_config.utd):
            stats = self.agent.update(self.buffer.sample(self.agent_config.batch_size, self.rng))
            if i % 32 == 0:
                self.diag.log_update(self.transitions, stats)
        self.diag.log_replay(
            self.transitions,
            size=self.buffer.size,
            reward_mean=float(self.buffer.reward[: self.buffer.size].mean()),
            done_frac=float(self.buffer.done[: self.buffer.size].mean()),
        )
        self.diag.log_nan_incidents(
            self.transitions,
            self.runner.nan_incidents,
            self.runner.magnitude_incidents,
        )

    # ------------------------------------------------------------------


    def deterministic_eval(
        self, *, episode_index_start: int, record_raw: bool = False, record_probe: bool = False
    ) -> dict[str, Any]:
        assert self.runner is not None

        eval_prev_flip = np.full(self.n_envs, -1, dtype=np.int8)
        eval_flip_transitions = np.zeros(self.n_envs, dtype=int)
        eval_flip_observations = np.zeros(self.n_envs, dtype=int)
        magnitude_before = self.runner.magnitude_incidents
        eval_incidents_seen = {
            "nan": self.runner.nan_incidents,
            "magnitude": self.runner.magnitude_incidents,
        }

        def eval_action_fn(X: np.ndarray, G: np.ndarray, _rng: np.random.Generator):
            if len(self._selection_panel_X) < self._selection_panel_limit:
                remaining = self._selection_panel_limit - len(self._selection_panel_X)
                take = min(remaining, len(X))
                self._selection_panel_X.extend(np.asarray(X[:take]).copy())
                self._selection_panel_G.extend(np.asarray(G[:take]).copy())
                if len(self._selection_panel_X) == self._selection_panel_limit:
                    self._selection_panel_frozen_at = self.transitions
                    self._persist_panel()
            p, delta, lift = self.agent.select_actions(
                X,
                G,
                step=self.transitions,
                total_budget=self.total,
                rng=_rng,
                deterministic=True,
            )
            assert self.runner is not None
            if (
                self.runner.nan_incidents != eval_incidents_seen["nan"]
                or self.runner.magnitude_incidents != eval_incidents_seen["magnitude"]
            ):
                eval_incidents_seen["nan"] = self.runner.nan_incidents
                eval_incidents_seen["magnitude"] = self.runner.magnitude_incidents
                self._reset_flip_tracking(
                    prev_flip=eval_prev_flip,
                    flip_transitions=eval_flip_transitions,
                    flip_observations=eval_flip_observations,
                )
            if np.all(self.runner.t == 0) and not np.any(self.runner.done):
                self._reset_flip_tracking(
                    prev_flip=eval_prev_flip,
                    flip_transitions=eval_flip_transitions,
                    flip_observations=eval_flip_observations,
                )
            self._log_lift_flip_diagnostics(
                self.transitions,
                X_before=X,
                goal_curves=G,
                lift=lift,
                active=~self.runner.done,
                templates=list(self.runner.init_shapes),
                phase="eval",
                prev_flip=eval_prev_flip,
                flip_transitions=eval_flip_transitions,
                flip_observations=eval_flip_observations,
            )
            return p, delta, lift

        eval_cfg = self.config.get("eval", {})
        # sprint_spec §5: per-episode discarded-retry cap (config flag,
        # default off = historical unlimited retries; sprint configs only).
        wall_guard_k = eval_cfg.get("wall_guard_k")
        wall_guard_k = int(wall_guard_k) if wall_guard_k is not None else None
        if self.task == "t2":
            result = evaluate_episodes(
                self.runner,
                n_episodes=len(self.val_goals),
                seed=self.seed + 500,
                episode_index_start=episode_index_start,
                action_fn=eval_action_fn,
                rng=np.random.default_rng(self.seed + 501),
                gamma=self.agent_config.gamma,
                goals=self.val_goals,
                goal_labels=self.val_labels,
                q_min_fn=self.agent.q_min_executed,
                wall_guard_k=wall_guard_k,
                record_raw=record_raw,
                record_probe=record_probe,
            )
        else:
            result = evaluate_episodes(
                self.runner,
                n_episodes=int(eval_cfg.get("t1_episodes_per_task", 100)),
                seed=self.seed + 500,
                episode_index_start=episode_index_start,
                action_fn=eval_action_fn,
                rng=np.random.default_rng(self.seed + 501),
                gamma=self.agent_config.gamma,
                goal_fn=self._goal_fn,
                q_min_fn=self.agent.q_min_executed,
                wall_guard_k=wall_guard_k,
                record_raw=record_raw,
                record_probe=record_probe,
            )
        panel_stats, panel_snapshot = self.agent.selection_panel(
            np.asarray(self._selection_panel_X),
            np.asarray(self._selection_panel_G),
        )
        if self._previous_selection_snapshot is None:
            panel_stats.update(
                {
                    "soft_weight_js_to_previous_checkpoint": None,
                    "soft_weight_cosine_to_previous_checkpoint": None,
                    "top8_contact_overlap": None,
                    "hard_q1_churn": None,
                    "hard_qmin_churn": None,
                }
            )
        else:
            panel_stats.update(
                compare_selection_snapshots(panel_snapshot, self._previous_selection_snapshot)
            )
        panel_stats["panel_states"] = len(self._selection_panel_X)
        panel_stats["panel_frozen_at_transition"] = self._selection_panel_frozen_at
        self._previous_selection_snapshot = panel_snapshot
        self._persist_previous_selection_snapshot(panel_snapshot)
        result["selection_diagnostics"] = panel_stats
        self.diag.log_selection_panel(self.transitions, panel_stats)
        result["magnitude_incidents_during_eval"] = (
            self.runner.magnitude_incidents - magnitude_before
        )
        return result

    def eval_and_checkpoint(self, *, final: bool = False) -> None:
        start = time.perf_counter()
        # F-b: capture the rebuild-independent eval index ONCE, outside the
        # recovery-retry loop, so retries reuse the same episode set.
        eval_index_start = eval_episode_index_start(self._eval_ordinal)
        # sprint_spec §3: raw trajectories on the FINAL eval only (periodic
        # evals excluded), behind a config flag (sprint configs only).
        record_raw = bool(final and self.config.get("eval", {}).get("record_raw_final", False))
        while True:
            try:
                result = self.deterministic_eval(
                    episode_index_start=eval_index_start, record_raw=record_raw
                )
                break
            except (FloatingPointError, ValueError, RuntimeError) as exc:
                if not is_nonfinite_error(exc):
                    raise
                if self._register_rebuild(context="eval_recovery", error=exc):
                    raise
                self.build_scene()
        # F-b: increment only on successful eval completion; record the
        # index actually used so run JSONs are auditable per attempt.
        self._eval_ordinal += 1
        result["eval_episode_index_start"] = eval_index_start
        result["wall_s"] = time.perf_counter() - start
        result["transitions"] = self.transitions
        if record_raw:
            # Preserve raw trajectories in a separate gz artifact; strip the
            # bulky fields from the in-memory history so run JSONs stay lean.
            import gzip

            raw_path = self.output_dir / "metrics" / f"p1_raw_final_eval_{self.transitions:07d}.json.gz"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_payload = {
                "run_tag": self.run_tag,
                "transitions": self.transitions,
                "eval_episode_index_start": eval_index_start,
                "wall_guard_k": result.get("wall_guard_k"),
                "record_raw": True,
                "episodes": result["episodes"],
            }
            with gzip.open(raw_path, "wt", encoding="utf-8") as handle:
                json.dump(raw_payload, handle)
            print(f"raw final-eval trajectories: {raw_path}")
            for ep in result["episodes"]:
                for key in ("x_initial", "x_steps", "x_terminal"):
                    ep.pop(key, None)
        self.eval_history.append(result)
        summary = {k: v for k, v in result.items() if k != "episodes"}
        self.diag.log_eval(self.transitions, summary)
        print(
            f"eval transitions={self.transitions} success={result['success_rate']:.3f} "
            f"return={result['mean_return']:.3f} final_d={result['mean_final_d']:.4f} "
            f"overest_gap={result['overestimation_gap_mean']} wall_s={result['wall_s']:.0f}"
        )

        ckpt = self.agent.save_checkpoint(self.models_dir / f"ckpt_{self.transitions:07d}.pt")
        self.last_checkpoint = ckpt
        if self.panel_artifact is not None:
            ckpt.with_suffix(ckpt.suffix + ".json").write_text(json.dumps({
                "panel_canonical_sha256": self.panel_artifact.canonical_sha256,
                "panel_artifact_sha256": self.panel_artifact.artifact_sha256,
            }) + "\n", encoding="utf-8")
        self._run_counterfactual_worker(ckpt, eval_index_start)
        if result["success_rate"] > self.best_success:
            self.best_success = result["success_rate"]
            self.agent.save_checkpoint(self.models_dir / "best.pt")
            print(f"new best checkpoint at {self.transitions} (success={self.best_success:.3f})")
        self.diag.maybe_plot(self.transitions, force=final)
        self.diag.save_history()
        self.save_run_summary()
        # Eval consumed the episode batch; restart training episodes.
        self.begin_training_episodes()

    def save_run_summary(self) -> None:
        payload = {
            "generated_at": utc_now(),
            "git_commit": get_git_commit_hash(),
            "run_tag": self.run_tag,
            "task": self.task,
            "seed": self.seed,
            "config": self.config,
            "td3_config": self.agent_config.to_dict(),
            "reward_constants": vars(self.episode_config.reward),
            "td_target_bound": dict(self.agent.td_target_bound),
            "initial_weights_sha256": self.initial_weights_sha256,
            "total_budget": self.total,
            "transitions": self.transitions,
            "updates": self.agent.update_count,
            "actor_updates": self.agent.actor_update_count,
            "target_updates": self.agent.target_update_count,
            "nan_incidents_env": self.runner.nan_incidents if self.runner else None,
            "magnitude_incidents_env": self.runner.magnitude_incidents if self.runner else None,
            "full_scene_rebuilds": self.full_rebuilds,
            "max_full_rebuilds": self.max_full_rebuilds,
            "discard_storm_rebuild_after": self.discard_storm_rebuild_after,
            "halt_reason": self.halt_reason,
            "best_success": self.best_success,
            "last_checkpoint": str(self.last_checkpoint) if self.last_checkpoint else None,
            "panel_canonical_sha256": self.panel_artifact.canonical_sha256 if self.panel_artifact else None,
            "panel_artifact_sha256": self.panel_artifact.artifact_sha256 if self.panel_artifact else None,
            "diagnostic_failure_ledger": str(self._diagnostic_failure_path),
            "counterfactual_diagnostic": self.counterfactual_diagnostic,
            "v2_downstream_ready": (
                self.task != "t2"
                or self.counterfactual_diagnostic["status"] == "completed"
            ),
            "exact_training_resume_supported": False,
            "evals": [
                {k: v for k, v in ev.items() if k != "episodes"} for ev in self.eval_history
            ],
            "eval_episodes": [
                {"transitions": ev["transitions"], "episodes": ev["episodes"]}
                for ev in self.eval_history
            ],
        }
        if self.asset_firewall is not None and self.asset_manifest is not None:
            receipts = persist_launch_receipts(
                self.output_dir, self.asset_manifest, self.asset_firewall
            )
            if (sha256_file(Path(receipts["r1"])) != receipts["r1_sha256"]
                    or sha256_file(Path(receipts["r2"])) != receipts["r2_sha256"]):
                raise RuntimeError("asset firewall receipt verification failed")
            payload["asset_firewall"] = receipts
        path = self.output_dir / "metrics" / "run_summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------

    def run(self) -> int:
        print(
            f"p1_train start {utc_now()} run_tag={self.run_tag} task={self.task} "
            f"seed={self.seed} total={self.total} n_envs={self.n_envs} device={self.device}"
        )
        print(
            f"td3={self.agent_config.to_dict()} reward={vars(self.episode_config.reward)} "
            f"td_target_bound={self.agent.td_target_bound}"
        )
        print(f"initial_weights_sha256={self.initial_weights_sha256}")
        start_wall = time.perf_counter()
        self.build_scene()
        next_eval = self.eval_every

        try:
            while self.transitions < self.total:
                round_start = time.perf_counter()
                count = self.collect_round()
                collect_s = time.perf_counter() - round_start
                update_start = time.perf_counter()
                self.train_updates(count)
                update_s = time.perf_counter() - update_start
                dense_line = roundlog_line(self.transitions, count, collect_s, update_s)
                if dense_line:
                    print(dense_line)
                if count and (self.transitions // count) % 20 == 0:
                    elapsed = time.perf_counter() - start_wall
                    rate = self.transitions / elapsed if elapsed > 0 else 0.0
                    eta_h = (self.total - self.transitions) / rate / 3600 if rate > 0 else 0.0
                    print(
                        f"round transitions={self.transitions}/{self.total} "
                        f"collect_s={collect_s:.1f} update_s={update_s:.1f} "
                        f"rate={rate:.2f}tr/s eta_h={eta_h:.2f} "
                        f"nan_env={self.runner.nan_incidents} "
                        f"mag={self.runner.magnitude_incidents} rebuilds={self.full_rebuilds}"
                    )
                if self.transitions >= next_eval:
                    self.eval_and_checkpoint(final=self.transitions >= self.total)
                    next_eval += self.eval_every
        except TrainingNaNError as exc:
            # Global rule 6, training level: halt + preserve last checkpoint +
            # factual report. No silent continuation.
            self.halt_reason = f"TrainingNaNError: {exc}"
            print(f"TRAINING HALT (rule 6): {self.halt_reason}")
            print(f"last checkpoint preserved: {self.last_checkpoint}")
            self.diag.save_history()
            self.save_run_summary()
            return 2

        if not self.eval_history or self.eval_history[-1]["transitions"] < self.transitions:
            self.eval_and_checkpoint(final=True)
        else:
            self.diag.maybe_plot(self.transitions, force=True)
            self.diag.save_history()
            self.save_run_summary()
        wall_h = (time.perf_counter() - start_wall) / 3600
        print(
            f"run complete transitions={self.transitions} updates={self.agent.update_count} "
            f"wall_h={wall_h:.2f} nan_env={self.runner.nan_incidents} "
            f"mag={self.runner.magnitude_incidents} rebuilds={self.full_rebuilds}"
        )
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="P1 training driver")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-tag", type=str, default=None)
    parser.add_argument("--total-override", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--asset-manifest", type=Path, default=None)
    parser.add_argument("--expected-asset-manifest-sha256", type=str, default=None)
    args = parser.parse_args()

    registry_root = Path("outputs/attempts")
    if args.asset_manifest is None and args.expected_asset_manifest_sha256 is None:
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    elif args.asset_manifest is not None and args.expected_asset_manifest_sha256 is not None:
        _, config_bytes = read_launch_asset_snapshot(
            args.asset_manifest,
            args.expected_asset_manifest_sha256,
            args.config,
            "config",
        )
        config = yaml.safe_load(config_bytes)
    else:
        raise AssetAccessError(
            "asset manifest and independent manifest SHA-256 are both required"
        )
    AttemptRegistry.recover(registry_root)
    run_tag = args.run_tag or f"{config['task']}_s{args.seed}"
    registry = AttemptRegistry(
        registry_root,
        run_tag=run_tag,
        config=config,
        code_sha256=sha256_file(Path(__file__).resolve()),
        seed=args.seed,
    )
    log_path = registry.attempt_path / "reports" / "p1_train.log"
    original_stdout = sys.stdout
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            sys.stdout = Tee(original_stdout, log_file)  # type: ignore[assignment]
            try:
                run = TrainingRun(args, registry, config=config)
                exit_code = run.run()
            finally:
                sys.stdout = original_stdout
                log_file.flush()
                os.fsync(log_file.fileno())
    except KeyboardInterrupt as error:
        try:
            registry.finalize_once("ABORTED", detail=str(error))
        except BaseException as finalization_error:
            print(
                f"attempt finalization failed after KeyboardInterrupt: {finalization_error}",
                file=sys.stderr,
            )
        raise
    except BaseException as error:
        try:
            registry.finalize_once(
                "TECHNICAL_FAILURE", detail=f"{type(error).__name__}: {error}"
            )
        except BaseException as finalization_error:
            print(
                f"attempt finalization failed after {type(error).__name__}: "
                f"{finalization_error}",
                file=sys.stderr,
            )
        raise
    registry.finalize_once(
        "SUCCEEDED" if exit_code == 0 else "TECHNICAL_FAILURE",
        exit_code=exit_code,
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
