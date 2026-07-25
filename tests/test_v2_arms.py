"""V2-DEV cadence, selection-objective, diagnostics, and BGT contracts (CPU)."""

from __future__ import annotations

import copy
import json
import types

import numpy as np
import pytest
import torch

from pathlib import Path
from dgcc.rl.selection import (
    CONTACT_WEIGHT_BETA,
    compare_selection_snapshots,
    contact_softmax_weights,
    kendall_rank_tau,
    selection_snapshot,
)
from dgcc.rl.sprint_arms import SprintTD3Agent, create_sprint_agent
from dgcc.rl.td3 import TD3Agent, TD3Config
from dgcc.rl.v2_arms import (
    BGTAgent,
    SelectionWeightedTD3Agent,
    selection_weighted_actor_loss,
)
from dgcc.rl import v2_arms

K = 32


def curve(offset: float = 0.0) -> np.ndarray:
    t = np.linspace(-0.5, 0.5, K)
    return np.column_stack((t + offset, 0.1 * np.sin(np.pi * t), np.zeros(K)))

def admitted_bgt(tmp_path: Path, *, margin: float = 1.0e9, onset: int = 100) -> BGTAgent:
    config = TD3Config()
    code_manifest = tmp_path / "code-manifest.json"
    code_manifest.write_text('{"closure":"full-governed-launch"}\n', encoding="utf-8")
    code_manifest_sha256 = v2_arms._sha256_file(code_manifest)
    lineage = {
        "development_split_path": "/development/t2.json",
        "development_split_sha256": "56" * 32,
        "development_split_role": "development_t2_split",
        "checkpoint_sha256": "34" * 32,
        "panel_sha256": "45" * 32,
        "config_sha256": v2_arms._sha256_json(config.to_dict()),
        "code_sha256": code_manifest_sha256,
    }
    identities = {
        "rank_calibration_sha256": "12" * 32,
        "gpu_latency_sha256": "23" * 32,
        "margin": margin,
        "onset_transition": onset,
        "development_lineage": lineage,
        "checkpoint_sha256": lineage["checkpoint_sha256"],
        "panel_sha256": lineage["panel_sha256"],
        "config_sha256": lineage["config_sha256"],
        "code_sha256": lineage["code_sha256"],
    }
    manifest = {
        "schema_version": 1,
        "bgt_admitted": True,
        **identities,
        "reason": "rank and approved synchronized-GPU latency gates passed",
    }
    manifest["admission_sha256"] = v2_arms._sha256_json(identities)
    path = tmp_path / "bgt-manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return BGTAgent(
        config,
        manifest_path=path,
        expected_manifest_sha256=v2_arms._sha256_file(path),
        checkpoint_sha256=identities["checkpoint_sha256"],
        panel_sha256=identities["panel_sha256"],
        code_manifest_sha256=code_manifest_sha256,
    )
def test_bgt_rejects_single_file_hash_instead_of_code_manifest(tmp_path: Path) -> None:
    admitted_bgt(tmp_path)
    path = tmp_path / "bgt-manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["code_sha256"] = v2_arms._sha256_file(Path(v2_arms.__file__))
    manifest["development_lineage"]["code_sha256"] = manifest["code_sha256"]
    material = {key: manifest[key] for key in v2_arms.BGT_ADMISSION_MATERIAL_KEYS}
    manifest["admission_sha256"] = v2_arms._sha256_json(material)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="identities are incompatible"):
        BGTAgent(
            TD3Config(),
            manifest_path=path,
            expected_manifest_sha256=v2_arms._sha256_file(path),
            checkpoint_sha256=manifest["checkpoint_sha256"],
            panel_sha256=manifest["panel_sha256"],
            code_manifest_sha256=v2_arms._sha256_file(tmp_path / "code-manifest.json"),
        )


def batch(size: int = 3, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "X_before": np.stack([curve(0.001 * i) for i in range(size)]),
        "X_after": np.stack([curve(0.001 * i + 0.01) for i in range(size)]),
        "goal_curve": np.stack([curve(0.03 + 0.001 * i) for i in range(size)]),
        "p": rng.integers(0, K, size=size),
        "delta": rng.uniform(-0.03, 0.03, size=(size, 3)),
        "lift": rng.integers(0, 2, size=size),
        "reward": rng.normal(size=size),
        "done": np.zeros(size, dtype=bool),
        "truncated": np.zeros(size, dtype=bool),
    }


def module_state(agent: TD3Agent) -> list[torch.Tensor]:
    modules = (
        agent.encoder,
        agent.critic,
        agent.actor,
        agent.encoder_target,
        agent.critic_target,
        agent.actor_target,
    )
    return [
        value.detach().clone()
        for module in modules
        for value in module.state_dict().values()
    ]


def kendall_tau_b_reference(left: list[float], right: list[float]) -> float:
    """Independent pair-count reference for the tau-b contract."""
    concordant = discordant = tied_left = tied_right = 0
    for i in range(len(left)):
        for j in range(i + 1, len(left)):
            left_sign = (left[i] > left[j]) - (left[i] < left[j])
            right_sign = (right[i] > right[j]) - (right[i] < right[j])
            if left_sign == 0 and right_sign == 0:
                continue
            if left_sign == 0:
                tied_left += 1
            elif right_sign == 0:
                tied_right += 1
            elif left_sign == right_sign:
                concordant += 1
            else:
                discordant += 1
    denominator = np.sqrt(
        (concordant + discordant + tied_left)
        * (concordant + discordant + tied_right)
    )
    return 0.0 if denominator == 0 else (concordant - discordant) / denominator


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    (
        ([1, 2, 3], [1, 2, 3], 1.0),
        ([1, 2, 3], [3, 2, 1], -1.0),
        ([1, 1, 1], [2, 2, 2], 0.0),
        ([1, 1, 1], [1, 2, 3], 0.0),
        ([1, 1, 2], [1, 2, 3], 0.8164965809277261),
        ([1, 1, 2], [2, 2, 1], -1.0),
    ),
)
def test_kendall_rank_tau_b_matches_committed_pair_count_reference(
    left: list[float], right: list[float], expected: float
) -> None:
    actual = kendall_rank_tau(
        torch.tensor([left], dtype=torch.float64),
        torch.tensor([right], dtype=torch.float64),
    )
    reference = kendall_tau_b_reference(left, right)
    assert reference == pytest.approx(expected, rel=0.0, abs=0.0)
    assert actual.item() == pytest.approx(reference, rel=0.0, abs=1e-15)


def test_contact_softmax_constant_tolerance_boundary_is_explicit() -> None:
    scores = torch.tensor(
        [[1.0, 1.0 + 8.0 * np.finfo(np.float64).eps]], dtype=torch.float64
    )
    uniform = torch.full_like(scores, 0.5)
    torch.testing.assert_close(
        contact_softmax_weights(scores), uniform, rtol=0.0, atol=0.0
    )

    above_tolerance = torch.tensor(
        [[1.0, 1.0 + 9.0 * np.finfo(np.float64).eps]], dtype=torch.float64
    )
    assert not torch.equal(contact_softmax_weights(above_tolerance), uniform)

def test_td3_default_policy_delay_is_native_one() -> None:
    assert TD3Config().policy_delay == 1


def test_actor_update_restores_critic_requires_grad_and_discards_critic_grads() -> None:
    agent = TD3Agent(TD3Config(policy_noise=0.0))
    critic_params = list(agent.critic.parameters())
    critic_params[0].requires_grad_(False)
    expected_requires_grad = [parameter.requires_grad for parameter in critic_params]

    agent.actor_update(batch(size=2))

    assert [parameter.requires_grad for parameter in critic_params] == expected_requires_grad
    assert all(parameter.grad is None for parameter in critic_params)


def test_checkpoint_is_evaluation_only_and_blocks_training_continuation(tmp_path) -> None:
    path = TD3Agent(TD3Config(policy_delay=2)).save_checkpoint(tmp_path / "td3.pt")
    restored = TD3Agent()
    with pytest.raises(ValueError, match="evaluation-only"):
        restored.load_checkpoint(path)

    restored.load_checkpoint(path, eval_only=True)
    assert restored.fresh_restart_only
    assert restored.to_dict()["resume"]["fresh_restart_only"]
def test_target_batch_requires_explicit_truncated_before_feature_work() -> None:
    agent = TD3Agent()
    with pytest.raises(ValueError, match="truncated"):
        agent.compute_target({"reward": np.zeros(1), "done": np.zeros(1, dtype=bool)})


def test_evaluation_checkpoint_rejects_contract_tampering_before_model_mutation(tmp_path) -> None:
    source = TD3Agent()
    path = source.save_checkpoint(tmp_path / "td3.pt")
    payload = torch.load(path, weights_only=False)
    payload["length_m"] = source.length_m + 1.0
    torch.save(payload, path)

    restored = TD3Agent()
    before = module_state(restored)
    with pytest.raises(ValueError, match="length contract"):
        restored.load_checkpoint(path, eval_only=True)
    for actual, expected in zip(module_state(restored), before, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)



def test_policy_delay_two_updates_actor_and_targets_every_second_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = TD3Agent(TD3Config(policy_delay=2))
    calls = {"critic": 0, "actor": 0, "target": 0}
    monkeypatch.setattr(agent, "features", lambda *args, **kwargs: torch.empty(0))

    def critic_update(self, _batch, **_kwargs):
        calls["critic"] += 1
        return {"critic_loss": 1.0}

    def actor_update(self, _batch, **_kwargs):
        calls["actor"] += 1
        return {"actor_loss": 2.0}

    def target_update(self):
        calls["target"] += 1

    monkeypatch.setattr(agent, "critic_update", types.MethodType(critic_update, agent))
    monkeypatch.setattr(agent, "actor_update", types.MethodType(actor_update, agent))
    monkeypatch.setattr(
        agent, "soft_update_targets", types.MethodType(target_update, agent)
    )

    empty_batch = {"X_before": None, "goal_curve": None}
    flags = [agent.update(empty_batch)["actor_updated"] for _ in range(5)]
    assert flags == [0.0, 1.0, 0.0, 1.0, 0.0]
    assert calls == {"critic": 5, "actor": 2, "target": 2}
    assert agent.update_count == 5
    assert agent.actor_update_count == agent.target_update_count == 2


def test_policy_delay_one_matches_native_update_sequence() -> None:
    torch.manual_seed(12)
    agent = TD3Agent(TD3Config(policy_delay=1, policy_noise=0.0), device="cpu")
    native = copy.deepcopy(agent)
    sample = batch(size=2, seed=4)

    agent.update(sample)
    feats = native.features(sample["X_before"], sample["goal_curve"])
    native.critic_update(sample, feats_before=feats)
    native.actor_update(sample, feats_before=feats)
    native.soft_update_targets()

    for actual, expected in zip(module_state(agent), module_state(native), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert agent.actor_update_count == agent.target_update_count == 1


def test_evaluation_checkpoint_discards_cadence_counts(tmp_path) -> None:
    agent = TD3Agent(TD3Config(policy_delay=2))
    agent.update_count = 7
    agent.actor_update_count = 3
    agent.target_update_count = 3
    path = agent.save_checkpoint(tmp_path / "cadence.pt")
    payload = torch.load(path, weights_only=False)
    assert "critic_optimizer" not in payload
    assert "update_count" not in payload

    restored = TD3Agent(TD3Config(policy_delay=2))
    restored.load_checkpoint(path, eval_only=True)
    assert (restored.update_count, restored.actor_update_count, restored.target_update_count) == (
        0,
        0,
        0,
    )


def test_v2_checkpoint_is_evaluation_only(tmp_path) -> None:
    agent = SelectionWeightedTD3Agent(
        TD3Config(), weight_operator="qmin", eval_operator="qmin"
    )
    path = agent.save_checkpoint(tmp_path / "v2.pt")
    restored = SelectionWeightedTD3Agent(
        TD3Config(), weight_operator="qmin", eval_operator="qmin"
    )
    with pytest.raises(ValueError, match="evaluation-only"):
        restored.load_checkpoint(path)

    restored.load_checkpoint(path, eval_only=True)
    assert restored.fresh_restart_only
    with pytest.raises(RuntimeError, match="fresh_restart_only"):
        restored.update(batch())


def test_sprint_evaluation_checkpoint_rejects_legacy_payload(tmp_path) -> None:
    path = SprintTD3Agent(TD3Config(policy_delay=2)).save_checkpoint(tmp_path / "sprint.pt")
    legacy_payload = torch.load(path, weights_only=False)
    legacy_payload.pop("sprint_arm")
    legacy_path = tmp_path / "legacy.pt"
    torch.save(legacy_payload, legacy_path)

    with pytest.raises(ValueError, match="incompatible sprint"):
        SprintTD3Agent().load_checkpoint(legacy_path, eval_only=True)
    with pytest.raises(ValueError, match="legacy checkpoint migration"):
        SprintTD3Agent().load_checkpoint(
            legacy_path,
            eval_only=True,
            migration_mode="legacy_response_head_eval",
        )


def test_constant_scores_use_uniform_detached_weights() -> None:
    scores = torch.full((2, K), 12345.0, requires_grad=True)
    weights = contact_softmax_weights(scores)
    torch.testing.assert_close(weights, torch.full_like(weights, 1.0 / K))
    assert not weights.requires_grad


def test_weight_path_has_no_critic_gradient() -> None:
    q1 = torch.tensor([[3.0, 2.0, 1.0]], requires_grad=True)
    q2 = torch.tensor([[0.3, 0.2, 0.1]], requires_grad=True)
    loss, weights = selection_weighted_actor_loss(
        q1, q2, weight_operator="q1", eval_operator="qmin"
    )
    loss.backward()
    torch.testing.assert_close(q1.grad, torch.zeros_like(q1))
    torch.testing.assert_close(q2.grad, -weights)
    assert not weights.requires_grad


@pytest.mark.parametrize(
    ("weight_operator", "eval_operator"),
    (("qmin", "qmin"), ("q1", "qmin"), ("q1", "q1")),
)
def test_operator_combinations_match_manual_formula(
    weight_operator, eval_operator
) -> None:
    q1 = torch.tensor([[0.3, 0.1, -0.2], [0.4, 0.0, 0.2]])
    q2 = torch.tensor([[0.2, 0.0, -0.1], [0.1, 0.3, 0.2]])
    loss, weights = selection_weighted_actor_loss(
        q1,
        q2,
        weight_operator=weight_operator,
        eval_operator=eval_operator,
    )
    qmin = torch.minimum(q1, q2)
    scores = q1 if weight_operator == "q1" else qmin
    values = q1 if eval_operator == "q1" else qmin
    expected_weights = torch.softmax(scores / CONTACT_WEIGHT_BETA, dim=1)
    torch.testing.assert_close(weights, expected_weights)
    torch.testing.assert_close(loss, -(expected_weights * values).sum(dim=1).mean())


def test_uniform_fallback_matches_native_all_contact_objective() -> None:
    q1 = torch.full((2, K), 2.0)
    q2 = torch.arange(2 * K, dtype=torch.float32).reshape(2, K) / 100.0
    loss, weights = selection_weighted_actor_loss(
        q1, q2, weight_operator="q1", eval_operator="qmin"
    )
    qmin = torch.minimum(q1, q2)
    torch.testing.assert_close(weights, torch.full_like(weights, 1.0 / K))
    torch.testing.assert_close(loss, -qmin.mean())


def test_policy_delay_one_plus_uniform_weights_matches_native_update() -> None:
    config = TD3Config(policy_delay=1, policy_noise=0.0)
    torch.manual_seed(44)
    native = SprintTD3Agent(config, arm="v1")
    torch.manual_seed(44)
    candidate = SelectionWeightedTD3Agent(
        config, weight_operator="q1", eval_operator="qmin"
    )

    def make_constant_q1(agent):
        original = agent._q_all_candidates

        def q_candidates(self, head, h, u):
            if head is self.critic.q1:
                return torch.full(h.shape[:2], 100.0, dtype=h.dtype, device=h.device)
            return original(head, h, u)

        agent._q_all_candidates = types.MethodType(q_candidates, agent)

    make_constant_q1(native)
    make_constant_q1(candidate)
    sample = batch(size=2, seed=9)
    native.update(sample)
    candidate.update(sample)
    for actual, expected in zip(
        module_state(candidate), module_state(native), strict=True
    ):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_beta_contact_is_locked_and_metadata_never_uses_contact_tau() -> None:
    agent = SelectionWeightedTD3Agent(
        TD3Config(), weight_operator="qmin", eval_operator="qmin"
    )
    assert agent.to_dict()["v2_arm"]["beta_contact"] == 0.015363
    assert "tau" not in agent.to_dict()["v2_arm"]
    with pytest.raises(ValueError, match="charter-locked"):
        SelectionWeightedTD3Agent(
            TD3Config(),
            weight_operator="qmin",
            eval_operator="qmin",
            beta_contact=0.025,
        )


def test_selection_actor_update_reuses_one_actor_and_twin_critic_forward() -> None:
    agent = SelectionWeightedTD3Agent(
        TD3Config(), weight_operator="qmin", eval_operator="qmin"
    )
    calls = {"actor": 0, "q1": 0, "q2": 0}
    hooks = [
        agent.actor.register_forward_hook(
            lambda *_: calls.__setitem__("actor", calls["actor"] + 1)
        ),
        agent.critic.q1.register_forward_hook(
            lambda *_: calls.__setitem__("q1", calls["q1"] + 1)
        ),
        agent.critic.q2.register_forward_hook(
            lambda *_: calls.__setitem__("q2", calls["q2"] + 1)
        ),
    ]
    try:
        stats = agent.actor_update(batch(size=2))
    finally:
        for hook in hooks:
            hook.remove()
    assert calls == {"actor": 1, "q1": 1, "q2": 1}
    assert stats["contact_weight_neff"] > 1.0


def test_factory_maps_all_three_operator_candidates() -> None:
    expected = {
        "v2-dmm": ("qmin", "qmin"),
        "v2-d1m": ("q1", "qmin"),
        "v2-d11": ("q1", "q1"),
    }
    for arm, operators in expected.items():
        agent = create_sprint_agent(arm, TD3Config())
        assert isinstance(agent, SelectionWeightedTD3Agent)
        assert (agent.weight_operator, agent.eval_operator) == operators
        assert agent.candidate_id == arm


def test_snapshot_comparison_is_exact_on_identical_panel() -> None:
    q1 = torch.arange(2 * K, dtype=torch.float32).reshape(2, K)
    q2 = q1.flip(1)
    weights = contact_softmax_weights(q1)
    snapshot = selection_snapshot(q1, q2, weights).cpu()
    stats = compare_selection_snapshots(snapshot, snapshot)
    assert stats == pytest.approx(
        {
            "soft_weight_js_to_previous_checkpoint": 0.0,
            "soft_weight_cosine_to_previous_checkpoint": 1.0,
            "top8_contact_overlap": 1.0,
            "hard_q1_churn": 0.0,
            "hard_qmin_churn": 0.0,
        }
    )


def test_selection_panel_exposes_all_lift_diagnostics() -> None:
    agent = SelectionWeightedTD3Agent(
        TD3Config(), weight_operator="qmin", eval_operator="qmin"
    )
    X = np.stack([curve(), curve(0.01)])
    G = np.stack([curve(0.03), curve(0.04)])
    stats, snapshot = agent.selection_panel(X, G)
    required = {
        "lift_near_threshold_all_045_055",
        "lift_near_threshold_selected_045_055",
        "lift_flip_rate_under_plusminus_002",
        "q_continuous_lift_minus_q_hard_lift",
        "q_at_lift_1_minus_q_at_lift_0",
        "selected_lift_entropy",
    }
    assert required <= stats.keys()
    assert snapshot.weights.shape == (2, K)


def test_bgt_is_fail_closed_without_admission_manifest() -> None:
    with pytest.raises(ValueError, match="admission required"):
        create_sprint_agent("v2-bgt", TD3Config())


def test_counterfactual_qmin_selector_uses_clipped_twin_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = TD3Agent(TD3Config())
    q1 = torch.zeros((2, K))
    q2 = torch.zeros((2, K))
    q1[:, 0] = 10.0
    q1[:, 1] = 5.0
    q2[:, 0] = -10.0
    q2[:, 1] = 4.0
    monkeypatch.setattr(agent, "features", lambda X, G: torch.zeros((len(X), K, 256)))
    agent.encoder = torch.nn.Identity()
    monkeypatch.setattr(
        agent.actor, "forward", lambda h: torch.zeros((*h.shape[:2], 4))
    )

    def q_candidates(self, head, _h, _u):
        return q1 if head is self.critic.q1 else q2

    monkeypatch.setattr(
        agent, "_q_all_candidates", types.MethodType(q_candidates, agent)
    )
    X = np.stack([curve(), curve(0.01)])
    G = np.stack([curve(0.03), curve(0.04)])
    p_q1, *_ = agent.select_actions(
        X,
        G,
        step=0,
        total_budget=1,
        rng=np.random.default_rng(0),
        deterministic=True,
        selector_operator="q1",
    )
    p_qmin, *_ = agent.select_actions(
        X,
        G,
        step=0,
        total_budget=1,
        rng=np.random.default_rng(0),
        deterministic=True,
        selector_operator="qmin",
    )
    np.testing.assert_array_equal(p_q1, [0, 0])
    np.testing.assert_array_equal(p_qmin, [1, 1])


def test_admitted_bgt_selects_from_fresh_seeded_initialization(tmp_path: Path) -> None:
    torch.manual_seed(314)
    agent = admitted_bgt(tmp_path)
    X = np.stack([curve(), curve(0.01)])
    G = np.stack([curve(0.03), curve(0.04)])
    p, delta, lift = agent.select_actions(
        X,
        G,
        step=100,
        total_budget=300_000,
        rng=np.random.default_rng(9),
        deterministic=True,
    )
    assert p.shape == (2,)
    assert delta.shape == (2, 3)
    assert len(lift) == 2

def test_production_bgt_rejects_force_all_even_via_internal_helper(tmp_path: Path) -> None:
    agent = admitted_bgt(tmp_path)
    X = np.stack([curve(), curve(0.01)])
    G = np.stack([curve(0.03), curve(0.04)])
    with pytest.raises(ValueError, match="restricted"):
        agent._select_actions_impl(
            X, G, step=100, total_budget=300_000, rng=np.random.default_rng(5),
            deterministic=True, benchmark_force_all_eligible=True,
        )


def test_bgt_rejects_rehashed_manifest_with_extra_lineage_field(
    tmp_path: Path,
) -> None:
    admitted_bgt(tmp_path)
    path = tmp_path / "bgt-manifest.json"
    forged = json.loads(path.read_text(encoding="utf-8"))
    forged["development_lineage"]["unapproved"] = "field"
    material = {
        key: forged[key] for key in v2_arms.BGT_ADMISSION_MATERIAL_KEYS
    }
    forged["admission_sha256"] = v2_arms._sha256_json(material)
    path.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(ValueError, match="development lineage"):
        BGTAgent(
            TD3Config(),
            manifest_path=path,
            expected_manifest_sha256=v2_arms._sha256_file(path),
            checkpoint_sha256=forged["checkpoint_sha256"],
            panel_sha256=forged["panel_sha256"],
            code_manifest_sha256=forged["code_sha256"],
        )


def test_bgt_evaluation_checkpoint_uses_separate_optional_byte_pin(
    tmp_path: Path,
) -> None:
    source = admitted_bgt(tmp_path)
    checkpoint = source.save_checkpoint(tmp_path / "bgt.pt")
    checkpoint_sha256 = v2_arms._sha256_file(checkpoint)
    restored = admitted_bgt(tmp_path)
    with pytest.raises(ValueError, match="only be loaded for evaluation"):
        restored.load_checkpoint(checkpoint)
    with pytest.raises(ValueError, match="bytes do not match"):
        restored.load_checkpoint(
            checkpoint,
            eval_only=True,
            expected_checkpoint_sha256="ab" * 32,
        )
    restored.load_checkpoint(
        checkpoint,
        eval_only=True,
        expected_checkpoint_sha256=checkpoint_sha256,
    )
    assert restored.fresh_restart_only

@pytest.mark.parametrize(
    "value",
    (
        np.asarray(["false"]),
        np.asarray([2]),
        np.asarray([0.5]),
        np.asarray([np.inf]),
    ),
)
def test_target_batch_rejects_nonbinary_done_and_truncated_values(value: np.ndarray) -> None:
    sample = batch(size=1)
    sample["done"] = value
    with pytest.raises(ValueError, match="done"):
        TD3Agent().compute_target(sample)
    sample = batch(size=1)
    sample["truncated"] = value
    with pytest.raises(ValueError, match="truncated"):
        TD3Agent().compute_target(sample)


def test_eval_only_agent_rejects_direct_target_mutation(tmp_path: Path) -> None:
    path = TD3Agent().save_checkpoint(tmp_path / "td3.pt")
    agent = TD3Agent()
    agent.load_checkpoint(path, eval_only=True)
    with pytest.raises(RuntimeError, match="fresh_restart_only"):
        agent.soft_update_targets()