"""V2 selection-weighted actor and response-gated behavior candidates."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from dgcc.models.networks import DELTA_SCALE, K_NODES, build_node_features
from dgcc.phi.dct import Phi_DCT
from dgcc.rl.selection import (
    CONTACT_WEIGHT_BETA,
    ContactOperator,
    contact_softmax_weights,
    goal_relative_progress,
    lift_statistics,
    selection_snapshot,
    selection_statistics,
)
from dgcc.rl.sprint_arms import SprintTD3Agent
from dgcc.rl.td3 import TD3Config, epsilon_schedule

V2SelectionArm = Literal["v2-dmm", "v2-d1m", "v2-d11"]
V2Arm = Literal["v2-dmm", "v2-d1m", "v2-d11", "v2-bgt"]

_OPERATOR_COMBINATIONS: dict[
    V2SelectionArm, tuple[ContactOperator, ContactOperator]
] = {
    "v2-dmm": ("qmin", "qmin"),
    "v2-d1m": ("q1", "qmin"),
    "v2-d11": ("q1", "q1"),
}


def _candidate_id(
    weight_operator: ContactOperator, eval_operator: ContactOperator
) -> str:
    matches = [
        candidate
        for candidate, operators in _OPERATOR_COMBINATIONS.items()
        if operators == (weight_operator, eval_operator)
    ]
    if not matches:
        raise ValueError(
            "unsupported selector-evaluator combination; expected DMM, D1M, or D11"
        )
    return matches[0]


def selection_weighted_actor_loss(
    q1: torch.Tensor,
    q2: torch.Tensor,
    *,
    weight_operator: ContactOperator,
    eval_operator: ContactOperator,
    beta_contact: float = CONTACT_WEIGHT_BETA,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one of the three chartered actor losses and detached weights."""

    if q1.shape != q2.shape or q1.ndim != 2:
        raise ValueError("q1 and q2 must have the same shape (B, K)")
    if weight_operator not in {"q1", "qmin"} or eval_operator not in {"q1", "qmin"}:
        raise ValueError("operators must be 'q1' or 'qmin'")
    qmin = torch.minimum(q1, q2)
    scores = q1 if weight_operator == "q1" else qmin
    values = q1 if eval_operator == "q1" else qmin
    weights = contact_softmax_weights(scores, beta_contact=beta_contact)
    return -(weights * values).sum(dim=1).mean(), weights


class SelectionWeightedTD3Agent(SprintTD3Agent):
    """V1 response supervision with a dense selection-weighted actor objective."""

    v2_schema_version = 1

    def __init__(
        self,
        config: TD3Config | None = None,
        *,
        weight_operator: ContactOperator,
        eval_operator: ContactOperator,
        beta_contact: float = CONTACT_WEIGHT_BETA,
        device: str | torch.device = "cpu",
        **kwargs: Any,
    ) -> None:
        super().__init__(config, arm="v1", device=device, **kwargs)
        self.weight_operator = weight_operator
        self.eval_operator = eval_operator
        self.candidate_id = _candidate_id(weight_operator, eval_operator)
        self.beta_contact = float(beta_contact)
        if self.beta_contact != CONTACT_WEIGHT_BETA:
            raise ValueError(
                f"beta_contact is charter-locked at {CONTACT_WEIGHT_BETA:.3f} Q units"
            )

    def contact_weights(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        qmin = torch.minimum(q1, q2)
        score = q1 if self.weight_operator == "q1" else qmin
        return contact_softmax_weights(score, beta_contact=self.beta_contact)

    def actor_update(
        self,
        batch: dict[str, np.ndarray],
        *,
        feats_before: torch.Tensor | None = None,
    ) -> dict[str, float]:
        feats = (
            self.features(
                batch["X_before"], batch["goal_curve"], batch.get("flip_before")
            )
            if feats_before is None
            else feats_before
        )
        with torch.no_grad():
            h = self.encoder(feats)
        u_all = self.actor(h)
        q1 = self._q_all_candidates(self.critic.q1, h, u_all)
        q2 = self._q_all_candidates(self.critic.q2, h, u_all)
        loss, weights = selection_weighted_actor_loss(
            q1,
            q2,
            weight_operator=self.weight_operator,
            eval_operator=self.eval_operator,
            beta_contact=self.beta_contact,
        )
        qmin = torch.minimum(q1, q2)
        values = q1 if self.eval_operator == "q1" else qmin
        self._assert_finite_loss(loss, "selection-weighted actor loss")

        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = self._clip_and_check_grads(list(self.actor.parameters()), "actor")
        self.actor_optimizer.step()
        self.critic_optimizer.zero_grad(set_to_none=True)

        stats = selection_statistics(q1.detach(), q2.detach(), weights)
        return {
            "actor_loss": float(loss.detach()),
            "actor_grad_norm": grad_norm,
            "weighted_actor_q": float((weights * values.detach()).sum(dim=1).mean()),
            "uniform_actor_q": float(values.detach().mean()),
            **stats,
        }

    @torch.no_grad()
    def selection_panel(
        self, X: np.ndarray, G_curve: np.ndarray, *, include_lift: bool = True
    ) -> tuple[dict[str, float], Any]:
        """Evaluate all V2 diagnostics on an ordered frozen development panel."""

        h = self.encoder(self.features(X, G_curve))
        u_all = self.actor(h)
        q1 = self._q_all_candidates(self.critic.q1, h, u_all)
        q2 = self._q_all_candidates(self.critic.q2, h, u_all)
        weights = self.contact_weights(q1, q2)
        selected = q1.argmax(dim=1)
        stats = selection_statistics(q1, q2, weights, selected=selected)
        if include_lift:
            stats.update(lift_statistics(self, h, u_all, q1, q2, selected))
        return stats, selection_snapshot(q1, q2, weights).cpu()

    def to_dict(self) -> dict[str, Any]:
        metadata = super().to_dict()
        metadata["v2_arm"] = {
            "schema_version": self.v2_schema_version,
            "candidate_id": self.candidate_id,
            "weight_operator": self.weight_operator,
            "eval_operator": self.eval_operator,
            "beta_contact": self.beta_contact,
            "deployment_operator": "q1",
        }
        return metadata

    def load_checkpoint(self, path: Path | str) -> None:
        payload = torch.load(Path(path), map_location=self.device, weights_only=False)
        metadata = payload.get("metadata", {}).get("v2_arm")
        if metadata is None:
            raise ValueError(
                "selection-weighted agent requires v2_arm checkpoint metadata"
            )
        expected = self.to_dict()["v2_arm"]
        if metadata != expected:
            raise ValueError(
                f"incompatible V2 checkpoint metadata: {metadata!r} != {expected!r}"
            )
        super().load_checkpoint(path)


class BGTAgent(SprintTD3Agent):
    """Q1-margin-gated, deterministic top-two response tie-break candidate."""

    v2_schema_version = 1

    def __init__(
        self,
        config: TD3Config | None = None,
        *,
        margin: float,
        onset_transition: int,
        calibration_sha256: str,
        device: str | torch.device = "cpu",
        **kwargs: Any,
    ) -> None:
        super().__init__(config, arm="v1", device=device, **kwargs)
        self.candidate_id = "v2-bgt"
        self.bgt_margin = float(margin)
        self.bgt_onset_transition = int(onset_transition)
        self.calibration_sha256 = str(calibration_sha256)
        if not np.isfinite(self.bgt_margin) or self.bgt_margin < 0:
            raise ValueError(
                "BGT margin must be a finite non-negative R5 calibration result"
            )
        if self.bgt_onset_transition < 0:
            raise ValueError("BGT onset_transition must be non-negative")
        if len(self.calibration_sha256) != 64:
            raise ValueError("BGT requires a 64-character R5 calibration SHA-256")

    @torch.no_grad()
    def select_actions(
        self,
        X: np.ndarray,
        G_curve: np.ndarray,
        *,
        step: int,
        total_budget: int,
        rng: np.random.Generator,
        deterministic: bool = False,
        return_info: bool = False,
        selector_operator: str = "behavior",
    ) -> (
        tuple[np.ndarray, np.ndarray, list[str]]
        | tuple[np.ndarray, np.ndarray, list[str], dict[str, np.ndarray]]
    ):
        if selector_operator in {"q1", "qmin"}:
            return super().select_actions(
                X,
                G_curve,
                step=step,
                total_budget=total_budget,
                rng=rng,
                deterministic=deterministic,
                return_info=return_info,
                selector_operator=selector_operator,
            )
        if selector_operator != "behavior":
            raise ValueError("selector_operator must be 'behavior', 'q1', or 'qmin'")
        features, flips = build_node_features(X, G_curve, self.length_m)
        feature_tensor = torch.as_tensor(
            features, dtype=torch.float32, device=self.device
        )
        h = self.encoder(feature_tensor)
        u_all = self.actor(h)
        q1 = self._q_all_candidates(self.critic.q1, h, u_all)
        top = torch.topk(q1, 2, dim=1)
        greedy = top.indices[:, 0]
        normalized_margin = (top.values[:, 0] - top.values[:, 1]) / q1.std(
            dim=1, unbiased=False
        ).clamp_min(1e-12)

        predicted_progress = torch.full(
            (h.shape[0], 2), float("nan"), dtype=h.dtype, device=self.device
        )
        eligible = (normalized_margin < self.bgt_margin) & (
            step >= self.bgt_onset_transition
        )
        if eligible.any():
            eligible_rows = torch.nonzero(eligible, as_tuple=False).flatten()
            eligible_top = top.indices[eligible_rows]
            row_grid = eligible_rows[:, None].expand(-1, 2)
            top_h = h[row_grid, eligible_top]
            top_u = u_all[row_grid, eligible_top]
            prediction = self.f_resp(
                top_h.reshape(-1, 256), top_u.reshape(-1, 4)
            ).reshape(-1, 2, 24)
            aligned_goals = np.asarray(G_curve).copy()
            aligned_goals[np.asarray(flips, dtype=bool)] = aligned_goals[
                np.asarray(flips, dtype=bool), ::-1
            ]
            residual = torch.as_tensor(
                np.stack(
                    [
                        Phi_DCT(goal) - Phi_DCT(state)
                        for state, goal in zip(X, aligned_goals, strict=True)
                    ]
                ),
                dtype=h.dtype,
                device=self.device,
            )
            residual_pairs = residual[eligible_rows, None, :].expand(-1, 2, -1)
            progress = goal_relative_progress(
                residual_pairs.reshape(-1, 24), prediction.reshape(-1, 24)
            ).reshape(-1, 2)
            predicted_progress[eligible_rows] = progress
            chosen = progress.argmax(dim=1)
            greedy[eligible_rows] = eligible_top[
                torch.arange(len(eligible_rows), device=self.device), chosen
            ]

        batch = feature_tensor.shape[0]
        p = greedy.cpu().numpy().copy()
        if not deterministic:
            eps = epsilon_schedule(step, total_budget, self.config)
            explore = rng.random(batch) < eps
            p[explore] = rng.integers(0, K_NODES, size=int(explore.sum()))

        rows = torch.arange(batch, device=self.device)
        u = (
            u_all[rows, torch.as_tensor(p, dtype=torch.long, device=self.device)]
            .cpu()
            .numpy()
        )
        delta = u[:, :3].copy()
        if not deterministic:
            delta += rng.normal(0.0, self.config.exploration_u_sigma, size=delta.shape)
            delta = np.clip(delta, -DELTA_SCALE, DELTA_SCALE)
        lift = ["high" if value > 0.5 else "low" for value in u[:, 3]]
        if return_info:
            return (
                p.astype(int),
                delta.astype(float),
                lift,
                {
                    "q1_candidates": q1.cpu().numpy(),
                    "u_executed": np.column_stack([delta, u[:, 3]]),
                    "bgt_top2": top.indices.cpu().numpy(),
                    "bgt_normalized_margin": normalized_margin.cpu().numpy(),
                    "bgt_eligible": eligible.cpu().numpy(),
                    "bgt_predicted_progress": predicted_progress.cpu().numpy(),
                },
            )
        return p.astype(int), delta.astype(float), lift

    def to_dict(self) -> dict[str, Any]:
        metadata = super().to_dict()
        metadata["v2_arm"] = {
            "schema_version": self.v2_schema_version,
            "candidate_id": self.candidate_id,
            "margin": self.bgt_margin,
            "onset_transition": self.bgt_onset_transition,
            "calibration_sha256": self.calibration_sha256,
            "deployment_operator": "q1_with_response_top2_tiebreak",
        }
        return metadata

    def load_checkpoint(self, path: Path | str) -> None:
        payload = torch.load(Path(path), map_location=self.device, weights_only=False)
        metadata = payload.get("metadata", {}).get("v2_arm")
        if metadata != self.to_dict()["v2_arm"]:
            raise ValueError("incompatible or missing V2-BGT checkpoint metadata")
        super().load_checkpoint(path)


def create_v2_agent(
    arm: V2Arm,
    config: TD3Config | None = None,
    *,
    device: str | torch.device = "cpu",
    aux_weight: float = 1.0,
    beta_contact: float = CONTACT_WEIGHT_BETA,
    bgt_margin: float | None = None,
    bgt_onset_transition: int | None = None,
    bgt_calibration_sha256: str | None = None,
    **kwargs: Any,
) -> SprintTD3Agent:
    if arm in _OPERATOR_COMBINATIONS:
        weight_operator, eval_operator = _OPERATOR_COMBINATIONS[arm]
        return SelectionWeightedTD3Agent(
            config,
            weight_operator=weight_operator,
            eval_operator=eval_operator,
            beta_contact=beta_contact,
            aux_weight=aux_weight,
            device=device,
            **kwargs,
        )
    if arm == "v2-bgt":
        if (
            bgt_margin is None
            or bgt_onset_transition is None
            or bgt_calibration_sha256 is None
        ):
            raise ValueError(
                "V2-BGT is fail-closed until all R5 calibration fields are pinned"
            )
        return BGTAgent(
            config,
            margin=bgt_margin,
            onset_transition=bgt_onset_transition,
            calibration_sha256=bgt_calibration_sha256,
            aux_weight=aux_weight,
            device=device,
            **kwargs,
        )
    raise ValueError(f"unknown V2 arm {arm!r}")


__all__ = [
    "BGTAgent",
    "SelectionWeightedTD3Agent",
    "V2Arm",
    "V2SelectionArm",
    "create_v2_agent",
    "selection_weighted_actor_loss",
]
