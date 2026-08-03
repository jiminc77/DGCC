from __future__ import annotations

import importlib.util
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from dgcc.rl.panel_artifacts import load_panel, load_panel_bytes, persist_panel
from dgcc.rl.selection import selection_statistics


def test_panel_canonical_hash_is_stable_and_artifact_hash_detects_tampering(tmp_path: Path) -> None:
    X = np.arange(3 * 32 * 3, dtype=np.float32).reshape(3, 32, 3)
    G = X + 1
    first = persist_panel(tmp_path / "panel.npz", X=X, G=G, order=np.arange(3), seed=4, transition=9, eval_ordinal=1)
    loaded = load_panel(first.path, expected_canonical_sha256=first.canonical_sha256)
    assert loaded.canonical_sha256 == first.canonical_sha256
    assert loaded.artifact_sha256 == first.artifact_sha256
    (first.path.with_suffix(first.path.suffix + ".json")).write_text("{}")
    with pytest.raises(
        RuntimeError,
        match="hash verification failed|metadata verification failed|manifest keys",
    ):
        load_panel(first.path)
    byte_artifact = persist_panel(
        tmp_path / "panel_bytes.npz", X=X, G=G, order=np.arange(3),
        seed=4, transition=9, eval_ordinal=1,
    )
    panel_bytes = byte_artifact.path.read_bytes()
    metadata_bytes = byte_artifact.path.with_suffix(byte_artifact.path.suffix + ".json").read_bytes()
    byte_loaded, arrays = load_panel_bytes(
        panel_bytes, metadata_bytes, path=byte_artifact.path,
        expected_canonical_sha256=byte_artifact.canonical_sha256,
        expected_artifact_sha256=hashlib.sha256(panel_bytes).hexdigest(),
    )
    assert byte_loaded == byte_artifact
    assert np.array_equal(arrays[0], X)


def test_paired_panel_mismatch_and_contact_counts(tmp_path: Path) -> None:
    artifact = persist_panel(
        tmp_path / "panel.npz",
        X=np.zeros((4, 32, 3)),
        G=np.ones((4, 32, 3)),
        order=np.arange(4),
        seed=0,
        transition=0,
        eval_ordinal=0,
    )
    with pytest.raises(RuntimeError, match="paired arm"):
        load_panel(artifact.path, expected_canonical_sha256="bad")
    q1 = torch.arange(32, dtype=torch.float32).repeat(4, 1)
    q2 = q1.clone()
    stats = selection_statistics(q1, q2, torch.full_like(q1, 1 / 32))
    assert sum(stats[f"contact_histogram_count_{i:02d}"] for i in range(32)) == 4


def test_worker_executes_disposable_one_step_selector_branches() -> None:
    worker_path = Path(__file__).parents[1] / "scripts" / "v2_counterfactual_worker.py"
    spec = importlib.util.spec_from_file_location("counterfactual_worker", worker_path)
    assert spec and spec.loader
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)

    class Goal:
        pass

    built_envs: list[dict] = []
    class FakeEnv:
        def __init__(self, **kwargs):
            self.n_envs = kwargs["n_envs"]
            # Record the resolved kwargs so the test also proves the worker
            # builds the CORRECTED adapter (reselection preflight, task A).
            self.kwargs = dict(kwargs)
            built_envs.append(dict(kwargs))
            self.env_resets: list[int] = []
        def reset(self, params, *, init_shape, seed):
            # `execute_branch` resets the scene before every batch; the stub
            # used to omit this, which made the test die on the first call
            # instead of exercising the selector branches it is named for.
            self.env_resets.append(int(seed))
        def get_centerline_batch(self):
            return np.zeros((self.n_envs, 32, 3), dtype=float)

    resets: list[tuple[int, int]] = []
    class FakeRunner:
        def __init__(self, env, *_args):
            self.env = env
        def begin_episodes(self, *, seed, episode_index, goals):
            resets.append((seed, episode_index))
        def step(self, p, delta, lift, *, rng):
            return {"d_before": np.full(len(p), 3.0), "d_after": np.asarray(p, dtype=float)}

    class FakeAgent:
        def __init__(self):
            self.calls: list[str] = []
        def select_actions(self, X, G, *, selector_operator, **_kwargs):
            self.calls.append(selector_operator)
            p = np.array([1, 2][:len(X)]) if selector_operator == "q1" else np.array([1, 0][:len(X)])
            return p, np.zeros((len(X), 3)), ["low"] * len(X)

    val_pairs = [("a", Goal()), ("b", Goal())]
    # The worker resolves `sim` fail-closed (reselection preflight): the
    # request always carries the training run's config snapshot, so the
    # fixture must carry a corrected `sim` block too.  A stub config that
    # omitted `sim` used to be silently defaulted into the pre-correction
    # adapter -- exactly the failure mode the resolver now refuses.
    cf_config = {
        "run": {"n_envs": 2},
        "eval": {"t2_episodes_per_goal": 1},
        "sim": {"move_v_max": 0.15, "move_hold_max_steps": 2000},
    }
    worker.goal_curve = lambda goal, length_m: np.zeros((32, 3))
    agent = FakeAgent()
    request = {"seed": 5, "transition": 9, "total_budget": 10, "development_episode_index_start": 90001}
    q1_p, q1_progress, q1_starts, q1_ids, q1_hashes = worker.execute_branch(
        "q1", agent, cf_config, request,
        val_pairs, FakeEnv, FakeRunner)
    qmin_p, qmin_progress, qmin_starts, qmin_ids, qmin_hashes = worker.execute_branch(
        "qmin", agent, cf_config, request,
        val_pairs, FakeEnv, FakeRunner)
    assert agent.calls == ["q1", "qmin"]
    assert resets == [(505, 90001), (505, 90001)]
    assert np.array_equal(q1_starts, qmin_starts)
    assert np.array_equal(q1_progress, [2.0, 1.0])
    assert np.array_equal(qmin_progress, [2.0, 3.0])
    assert np.array_equal(q1_p == qmin_p, [True, False])
    assert np.array_equal(q1_ids, ["a", "b"])
    assert np.array_equal(q1_ids, qmin_ids)
    assert np.array_equal(q1_hashes, qmin_hashes)
    # The worker must build the CORRECTED adapter for BOTH branches: a
    # counterfactual measured in a pre-correction env would attribute the
    # physics difference to the selector.
    assert built_envs, "execute_branch built no environment"
    for kwargs in built_envs:
        assert kwargs["move_v_max"] == 0.15
        assert kwargs["move_hold_max_steps"] == 2000
        assert kwargs["at1h_counters"] is False
        assert "move_step_size" not in kwargs
        assert "move_hold_steps" not in kwargs


def test_panel_selector_uses_authenticated_panel_order_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    worker_path = Path(__file__).parents[1] / "scripts" / "v2_counterfactual_worker.py"
    spec = importlib.util.spec_from_file_location("counterfactual_worker_panel", worker_path)
    assert spec and spec.loader
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)

    artifact = persist_panel(
        tmp_path / "panel.npz",
        X=np.stack(
            [np.full((32, 3), 10.0), np.full((32, 3), 20.0)]
        ),
        G=np.stack(
            [np.full((32, 3), 1.0), np.full((32, 3), 2.0)]
        ),
        order=np.array([1, 0]),
        seed=4,
        transition=9,
        eval_ordinal=1,
    )

    class FakeAgent:
        def select_actions(self, X, G, *, selector_operator, **_kwargs):
            assert np.array_equal(X[:, 0, 0], [20.0, 10.0])
            assert np.array_equal(G[:, 0, 0], [2.0, 1.0])
            return np.arange(len(X)) + (selector_operator == "qmin"), None, None

    request = {
        "panel_sha256": artifact.canonical_sha256, "seed": 5, "transition": 9,
        "total_budget": 10,
    }
    panel, arrays = load_panel_bytes(
        artifact.path.read_bytes(),
        artifact.path.with_suffix(artifact.path.suffix + ".json").read_bytes(),
        path=artifact.path,
        expected_canonical_sha256=artifact.canonical_sha256,
        expected_artifact_sha256=artifact.artifact_sha256,
    )
    q1, qmin, order = worker.select_panel(FakeAgent(), arrays, request)
    assert np.array_equal(q1, [0, 1])
    assert np.array_equal(qmin, [1, 2])
    assert np.array_equal(order, [1, 0])
    artifact.path.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="invalid panel artifact|hash verification failed|SHA mismatch"):
        load_panel_bytes(
            artifact.path.read_bytes(),
            artifact.path.with_suffix(artifact.path.suffix + ".json").read_bytes(),
            path=artifact.path,
            expected_canonical_sha256=artifact.canonical_sha256,
            expected_artifact_sha256=artifact.artifact_sha256,
        )


def test_worker_descriptor_snapshot_survives_path_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker_path = Path(__file__).parents[1] / "scripts" / "v2_counterfactual_worker.py"
    spec = importlib.util.spec_from_file_location("counterfactual_worker_snapshot", worker_path)
    assert spec and spec.loader
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    source = tmp_path / "input"
    replacement = tmp_path / "replacement"
    source.write_bytes(b"authenticated")
    replacement.write_bytes(b"replacement")
    real_fdopen = worker.os.fdopen

    def replace_after_open(fd, *args, **kwargs):
        handle = real_fdopen(fd, *args, **kwargs)
        replacement.replace(source)
        return handle

    monkeypatch.setattr(worker.os, "fdopen", replace_after_open)
    assert worker.read_regular_bytes(source) == b"authenticated"


def test_counterfactual_reporting_failure_cannot_escape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_path = Path(__file__).parents[1] / "scripts" / "p1_train.py"
    spec = importlib.util.spec_from_file_location("p1_train_counterfactual", train_path)
    assert spec and spec.loader
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    run = train.TrainingRun.__new__(train.TrainingRun)
    run.task = "t2"

    def diagnostic_failure(*_args, **_kwargs):
        raise RuntimeError("diagnostic setup failed")

    monkeypatch.setattr(run, "_run_counterfactual_worker_unsafe", diagnostic_failure)
    monkeypatch.setattr(run, "_record_counterfactual_failure", diagnostic_failure)
    monkeypatch.setattr(run, "save_run_summary", diagnostic_failure)
    run._run_counterfactual_worker(Path("checkpoint"), 1)
    assert run.counterfactual_diagnostic["status"] == "failed"
    assert run.counterfactual_diagnostic["downstream_ready"] is False
