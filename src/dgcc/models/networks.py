"""P1 §6 baseline architecture: encoder, twin per-point critic, per-point actor.

Module boundaries are deliberate — this is the common backbone for P2 probing
and P3 variants (P1.md §6):

    * :func:`build_node_features` — the §6 encoder INPUT CONTRACT:
      per-node feature = (x_i ∈ R³, arc-length coordinate σ_i, goal
      correspondence residual g_i − x_i) where the correspondence uses the
      canonical flip convention of
      :func:`dgcc.goals.distance.canonical_shape_flip` (M5R2). Goal
      conditioning enters ONLY through this input.
    * :class:`Encoder` — shared MLP(64) → 3-layer dilated 1D CNN (k=5,
      dilation 1/2/4, 128ch) + global max-pool broadcast → per-node embedding
      h_i ∈ R^256 (local 128 ⊕ global 128).
      LATENT HOOK [P2: "encoder per-node h_i"]: the forward output ``h``.
    * :class:`TwinCritic` — two independent per-point MLP(256, 256) heads on
      [h_p, u] → Q_i(s, g, p, u).
      LATENT HOOK [P2: "critic trunk mid-layer"]: ``forward(...,
      return_hidden=True)`` exposes each head's post-activation hidden layers.
    * :class:`Actor` — per-point MLP(256, 128): h_i → u_i = (Δ ∈ R³ tanh·0.15,
      lift logit).

Do NOT add response heads, auxiliary losses, or extra conditioning paths here
(P3 scope).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from dgcc.goals.distance import canonical_shape_flip
from dgcc.tasks.domain import P1_LENGTH_M

K_NODES = 32
NODE_FEATURE_DIM = 7  # x(3) + sigma(1) + goal residual(3)
U_DIM = 4  # delta(3) + lift(1)
DELTA_SCALE = 0.15
LOCAL_DIM = 128
GLOBAL_DIM = 128
EMBED_DIM = LOCAL_DIM + GLOBAL_DIM  # 256


def arc_length_coordinates(X: np.ndarray) -> np.ndarray:
    """Return normalized cumulative arc-length coordinates σ_i ∈ [0, 1]."""

    diffs = np.linalg.norm(np.diff(X, axis=-2), axis=-1)
    cumulative = np.concatenate(
        [np.zeros((*diffs.shape[:-1], 1)), np.cumsum(diffs, axis=-1)], axis=-1
    )
    total = cumulative[..., -1:]
    total = np.where(total <= 0.0, 1.0, total)
    return cumulative / total


def goal_residual_flips(
    X: np.ndarray, G_curve: np.ndarray, length_m: float = P1_LENGTH_M
) -> np.ndarray:
    """Return the per-sample canonical flip decisions for (X, G) batches.

    Routes through :func:`dgcc.goals.distance.canonical_shape_flip` (the M5R2
    fixed orientation convention) — no reimplementation.  The goal is passed
    as a mapping whose template is the goal curve itself; ``DualGoal``
    normalization reconstructs the same curve for the coefficient comparison.
    """

    x = np.asarray(X, dtype=float)
    g = np.asarray(G_curve, dtype=float)
    batched = x.ndim == 3
    if not batched:
        x = x[None]
        g = g[None]
    flips = np.zeros(x.shape[0], dtype=bool)
    for i in range(x.shape[0]):
        goal_map = {"shape_template": g[i], "anchor": g[i].mean(axis=0)}
        flips[i] = canonical_shape_flip(x[i], goal_map, length_m)
    return flips if batched else flips[:1]


def build_node_features(
    X: np.ndarray,
    G_curve: np.ndarray,
    length_m: float = P1_LENGTH_M,
    *,
    flips: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the §6 encoder input features for a batch.

    Args:
        X: ``(B, 32, 3)`` or ``(32, 3)`` current centerlines (canonical nodes).
        G_curve: matching goal curves.
        flips: optional precomputed canonical flip decisions (else computed
            via :func:`goal_residual_flips`).

    Returns:
        ``(features, flips)`` with features ``(B, 32, 7)``:
        ``[x_i, σ_i, g_i − x_i]`` where the goal correspondence is index-wise
        under the canonical flip decision.
    """

    x = np.asarray(X, dtype=float)
    g = np.asarray(G_curve, dtype=float)
    batched = x.ndim == 3
    if not batched:
        x = x[None]
        g = g[None]
    if flips is None:
        flips = goal_residual_flips(x, g, length_m)
    flips = np.asarray(flips, dtype=bool)

    g_aligned = np.where(flips[:, None, None], g[:, ::-1, :], g)
    sigma = arc_length_coordinates(x)[..., None]
    features = np.concatenate([x, sigma, g_aligned - x], axis=-1)
    if not batched:
        return features, flips
    return features, flips


class Encoder(nn.Module):
    """§6 encoder: shared MLP(64) → dilated 1D CNN ×3 → local⊕global h_i."""

    def __init__(self) -> None:
        super().__init__()
        self.point_mlp = nn.Sequential(nn.Linear(NODE_FEATURE_DIM, 64), nn.ReLU())
        self.conv = nn.Sequential(
            nn.Conv1d(64, LOCAL_DIM, kernel_size=5, dilation=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(LOCAL_DIM, LOCAL_DIM, kernel_size=5, dilation=2, padding=4),
            nn.ReLU(),
            nn.Conv1d(LOCAL_DIM, LOCAL_DIM, kernel_size=5, dilation=4, padding=8),
            nn.ReLU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """(B, 32, 7) → (B, 32, 256).

        LATENT HOOK [P2 latent_api: "encoder per-node h_i"]: the return value.
        """

        z = self.point_mlp(features)  # (B, 32, 64)
        local = self.conv(z.transpose(1, 2))  # (B, 128, 32)
        global_pool = local.max(dim=2).values  # (B, 128)
        broadcast = global_pool[:, :, None].expand(-1, -1, local.shape[2])
        h = torch.cat([local, broadcast], dim=1).transpose(1, 2)
        return h  # (B, 32, 256)


class _QHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(EMBED_DIM + U_DIM, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, 1)
        self.act = nn.ReLU()

    def forward(
        self, h_p: torch.Tensor, u: torch.Tensor, *, return_hidden: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        z = torch.cat([h_p, u], dim=-1)
        hid1 = self.act(self.fc1(z))
        hid2 = self.act(self.fc2(hid1))
        q = self.out(hid2).squeeze(-1)
        if return_hidden:
            # LATENT HOOK [P2 latent_api: "critic trunk mid-layer"].
            return q, {"trunk_hidden1": hid1, "trunk_hidden2": hid2}
        return q


class TwinCritic(nn.Module):
    """Twin per-point critics Q1/Q2 on [h_p, u] (P1 §6/§7)."""

    def __init__(self) -> None:
        super().__init__()
        self.q1 = _QHead()
        self.q2 = _QHead()

    def forward(self, h_p: torch.Tensor, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(h_p, u), self.q2(h_p, u)


class Actor(nn.Module):
    """Per-point actor MLP(256, 128): h_i → (Δ tanh·0.15, lift logit)."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EMBED_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, U_DIM),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h (..., 256) → u (..., 4) = [Δ ∈ [-0.15, 0.15]³, lift ∈ (0, 1)]."""

        raw = self.net(h)
        delta = torch.tanh(raw[..., :3]) * DELTA_SCALE
        lift = torch.sigmoid(raw[..., 3:4])
        return torch.cat([delta, lift], dim=-1)


def parameter_count(*modules: nn.Module) -> int:
    """Total trainable parameters across modules (spec guideline ~1-2M)."""

    return sum(p.numel() for m in modules for p in m.parameters() if p.requires_grad)


__all__ = [
    "Actor",
    "DELTA_SCALE",
    "EMBED_DIM",
    "Encoder",
    "K_NODES",
    "NODE_FEATURE_DIM",
    "TwinCritic",
    "U_DIM",
    "arc_length_coordinates",
    "build_node_features",
    "goal_residual_flips",
    "parameter_count",
]
