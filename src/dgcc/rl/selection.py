"""Selection-weighting and frozen-panel diagnostics shared by V2 arms."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch

ContactOperator = Literal["q1", "qmin"]
CONTACT_WEIGHT_BETA = 0.015363


def contact_softmax_weights(
    scores: torch.Tensor, *, beta_contact: float = CONTACT_WEIGHT_BETA
) -> torch.Tensor:
    """Return detached dense contact weights with a numerical-constant fallback."""

    if scores.ndim != 2 or scores.shape[1] < 2:
        raise ValueError("scores must have shape (B, K) with K >= 2")
    if beta_contact <= 0 or not math.isfinite(beta_contact):
        raise ValueError("beta_contact must be finite and positive")
    detached = scores.detach()
    if not torch.isfinite(detached).all():
        raise ValueError("contact scores must be finite")

    row_max = detached.max(dim=1, keepdim=True).values
    row_min = detached.min(dim=1, keepdim=True).values
    scale = detached.abs().max(dim=1, keepdim=True).values.clamp_min(1.0)
    tolerance = 8.0 * torch.finfo(detached.dtype).eps * scale
    constant = row_max - row_min <= tolerance

    weights = torch.softmax(detached / beta_contact, dim=1)
    uniform = torch.full_like(weights, 1.0 / weights.shape[1])
    return torch.where(constant, uniform, weights)


def uniform_contact_weights(scores: torch.Tensor) -> torch.Tensor:
    if scores.ndim != 2 or scores.shape[1] < 2:
        raise ValueError("scores must have shape (B, K) with K >= 2")
    return torch.full_like(scores.detach(), 1.0 / scores.shape[1])


def kendall_rank_tau(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Return per-row Kendall tau-b, with a zero result for zero denominators."""
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("rank tensors must have the same shape (B, K)")
    left_diff = left[:, :, None] - left[:, None, :]
    right_diff = right[:, :, None] - right[:, None, :]
    upper = torch.triu(
        torch.ones(left.shape[1], left.shape[1], dtype=torch.bool, device=left.device),
        diagonal=1,
    )
    left_sign = torch.sign(left_diff[:, upper])
    right_sign = torch.sign(right_diff[:, upper])
    products = left_sign * right_sign
    concordant = (products > 0).sum(dim=1)
    discordant = (products < 0).sum(dim=1)
    tied_left = ((left_sign == 0) & (right_sign != 0)).sum(dim=1)
    tied_right = ((right_sign == 0) & (left_sign != 0)).sum(dim=1)
    numerator = concordant - discordant
    denominator = torch.sqrt(
        (concordant + discordant + tied_left).to(torch.float64)
        * (concordant + discordant + tied_right).to(torch.float64)
    )
    return torch.where(
        denominator > 0,
        numerator.to(torch.float64) / denominator,
        torch.zeros_like(denominator),
    )


def _normalized_histogram(indices: torch.Tensor, contacts: int) -> torch.Tensor:
    counts = torch.bincount(indices, minlength=contacts).to(torch.float64)
    return counts / counts.sum().clamp_min(1.0)


@dataclass(frozen=True)
class SelectionSnapshot:
    """Per-state tensors retained for same-panel checkpoint comparison."""

    q1_selected: torch.Tensor
    qmin_selected: torch.Tensor
    weights: torch.Tensor
    top8: torch.Tensor
    contact_histogram: torch.Tensor
    contact_histogram_counts: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.int64)
    )

    def cpu(self) -> SelectionSnapshot:
        return SelectionSnapshot(
            q1_selected=self.q1_selected.detach().cpu(),
            qmin_selected=self.qmin_selected.detach().cpu(),
            weights=self.weights.detach().cpu(),
            top8=self.top8.detach().cpu(),
            contact_histogram=self.contact_histogram.detach().cpu(),
            contact_histogram_counts=self.contact_histogram_counts.detach().cpu(),
        )


def selection_snapshot(
    q1: torch.Tensor, q2: torch.Tensor, weights: torch.Tensor
) -> SelectionSnapshot:
    if q1.shape != q2.shape or q1.shape != weights.shape or q1.ndim != 2:
        raise ValueError("q1, q2, and weights must have the same shape (B, K)")
    qmin = torch.minimum(q1, q2)
    q1_selected = q1.argmax(dim=1)
    return SelectionSnapshot(
        q1_selected=q1_selected,
        qmin_selected=qmin.argmax(dim=1),
        weights=weights.detach(),
        top8=torch.topk(weights.detach(), min(8, weights.shape[1]), dim=1).indices,
        contact_histogram=_normalized_histogram(q1_selected, q1.shape[1]),
        contact_histogram_counts=torch.bincount(
            q1_selected, minlength=q1.shape[1]
        ).to(torch.int64),
    )


def selection_statistics(
    q1: torch.Tensor,
    q2: torch.Tensor,
    weights: torch.Tensor,
    *,
    selected: torch.Tensor | None = None,
) -> dict[str, float]:
    """Compute scalar selector/weight diagnostics without changing model state."""

    snapshot = selection_snapshot(q1, q2, weights)
    qmin = torch.minimum(q1, q2)
    selected = snapshot.q1_selected if selected is None else selected
    rows = torch.arange(q1.shape[0], device=q1.device)
    entropy = -(weights * weights.clamp_min(1e-30).log()).sum(dim=1)
    q1_margin = torch.topk(q1, 2, dim=1).values.diff(dim=1).neg().squeeze(1)
    qmin_margin = torch.topk(qmin, 2, dim=1).values.diff(dim=1).neg().squeeze(1)
    histogram = snapshot.contact_histogram
    histogram_entropy = -(
        histogram[histogram > 0] * histogram[histogram > 0].log()
    ).sum() / math.log(q1.shape[1])
    statistics: dict[str, float] = {
        "contact_weight_entropy": float(entropy.mean()),
        "contact_weight_neff": float((1.0 / weights.square().sum(dim=1)).mean()),
        "contact_weight_top1_mass": float(weights.max(dim=1).values.mean()),
        "q1_qmin_argmax_agreement": float(
            (snapshot.q1_selected == snapshot.qmin_selected).to(torch.float32).mean()
        ),
        "q1_qmin_rank_tau": float(kendall_rank_tau(q1, qmin).mean()),
        "q1_top1_top2_margin": float(q1_margin.mean()),
        "qmin_top1_top2_margin": float(qmin_margin.mean()),
        "selected_contact_weight": float(weights[rows, selected].mean()),
        "contact_histogram_entropy": float(histogram_entropy),
        "contact_histogram_max_share": float(histogram.max()),
    }
    statistics.update(
        {
            f"contact_histogram_count_{index:02d}": int(count)
            for index, count in enumerate(snapshot.contact_histogram_counts.tolist())
        }
    )
    return statistics


def _js_divergence(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    midpoint = 0.5 * (left + right)
    left_term = torch.where(
        left > 0, left * (left / midpoint.clamp_min(1e-30)).log(), 0.0
    )
    right_term = torch.where(
        right > 0, right * (right / midpoint.clamp_min(1e-30)).log(), 0.0
    )
    return 0.5 * (left_term.sum(dim=1) + right_term.sum(dim=1))


def compare_selection_snapshots(
    current: SelectionSnapshot, previous: SelectionSnapshot
) -> dict[str, float]:
    """Compare two checkpoints evaluated on the exact same ordered state panel."""

    if current.weights.shape != previous.weights.shape:
        raise ValueError("selection snapshots must use the same ordered state panel")
    current_weights = current.weights.to(torch.float64)
    previous_weights = previous.weights.to(torch.float64)
    cosine = torch.nn.functional.cosine_similarity(
        current_weights, previous_weights, dim=1
    )
    overlaps = []
    for current_top, previous_top in zip(current.top8, previous.top8, strict=True):
        overlaps.append(
            len(set(current_top.tolist()) & set(previous_top.tolist())) / 8.0
        )
    return {
        "soft_weight_js_to_previous_checkpoint": float(
            _js_divergence(current_weights, previous_weights).mean()
        ),
        "soft_weight_cosine_to_previous_checkpoint": float(cosine.mean()),
        "top8_contact_overlap": float(np.mean(overlaps)),
        "hard_q1_churn": float(
            (current.q1_selected != previous.q1_selected).to(torch.float32).mean()
        ),
        "hard_qmin_churn": float(
            (current.qmin_selected != previous.qmin_selected).to(torch.float32).mean()
        ),
    }


def goal_relative_progress(
    residual: torch.Tensor, predicted_delta: torch.Tensor
) -> torch.Tensor:
    """Normalized squared-residual reduction used by the BGT tie-break."""

    if residual.shape != predicted_delta.shape or residual.ndim != 2:
        raise ValueError("residual and predicted_delta must have matching shape (B, D)")
    before = residual.square().sum(dim=1)
    after = (residual - predicted_delta).square().sum(dim=1)
    return (before - after) / before.clamp_min(1e-12)


def lift_statistics(
    agent: Any,
    h: torch.Tensor,
    u_all: torch.Tensor,
    q1: torch.Tensor,
    q2: torch.Tensor,
    selected: torch.Tensor,
) -> dict[str, float]:
    """Compute the six preregistered lift diagnostics on a validation panel."""

    if u_all.ndim != 3 or u_all.shape[:2] != q1.shape or q1.shape != q2.shape:
        raise ValueError("incompatible actor/critic candidate tensors")
    rows = torch.arange(u_all.shape[0], device=u_all.device)
    lift = u_all[..., 3]
    selected_lift = lift[rows, selected]
    near_all = (lift >= 0.45) & (lift <= 0.55)
    near_selected = (selected_lift >= 0.45) & (selected_lift <= 0.55)
    flip_sensitive = ((lift - 0.02) > 0.5) != ((lift + 0.02) > 0.5)

    hard_u = u_all.clone()
    hard_u[..., 3] = (lift > 0.5).to(lift.dtype)
    one_u = u_all.clone()
    one_u[..., 3] = 1.0
    zero_u = u_all.clone()
    zero_u[..., 3] = 0.0
    qmin_continuous = torch.minimum(q1, q2)
    qmin_hard = torch.minimum(
        agent._q_all_candidates(agent.critic.q1, h, hard_u),
        agent._q_all_candidates(agent.critic.q2, h, hard_u),
    )
    qmin_one = torch.minimum(
        agent._q_all_candidates(agent.critic.q1, h, one_u),
        agent._q_all_candidates(agent.critic.q2, h, one_u),
    )
    qmin_zero = torch.minimum(
        agent._q_all_candidates(agent.critic.q1, h, zero_u),
        agent._q_all_candidates(agent.critic.q2, h, zero_u),
    )
    selected_entropy = -(
        selected_lift * selected_lift.clamp_min(1e-30).log()
        + (1.0 - selected_lift) * (1.0 - selected_lift).clamp_min(1e-30).log()
    )
    return {
        "lift_near_threshold_all_045_055": float(near_all.to(torch.float32).mean()),
        "lift_near_threshold_selected_045_055": float(
            near_selected.to(torch.float32).mean()
        ),
        "lift_flip_rate_under_plusminus_002": float(
            flip_sensitive.to(torch.float32).mean()
        ),
        "q_continuous_lift_minus_q_hard_lift": float(
            (qmin_continuous - qmin_hard).mean()
        ),
        "q_at_lift_1_minus_q_at_lift_0": float((qmin_one - qmin_zero).mean()),
        "selected_lift_entropy": float(selected_entropy.mean()),
    }


__all__ = [
    "CONTACT_WEIGHT_BETA",
    "ContactOperator",
    "SelectionSnapshot",
    "compare_selection_snapshots",
    "contact_softmax_weights",
    "goal_relative_progress",
    "kendall_rank_tau",
    "lift_statistics",
    "selection_snapshot",
    "selection_statistics",
    "uniform_contact_weights",
]
