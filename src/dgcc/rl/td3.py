"""P1 §7 TD3-style off-policy learner with double-Q decoupling.

Target semantics (consensus plan C4/F4 — EXACT, target networks only):

    y = r + γ · min(Q_target_1, Q_target_2)(s′, g, p′*, ũ′),  γ = 0.95
    p′* = argmax_p Q_target_1(s′, g, p, u_target(s′, p))       # selection: Q_target_1
    ũ′  = u_target(s′, p′*) + clip(N(0, 0.05), ±0.1)           # target policy smoothing

Online critics NEVER appear in the target computation.  The actor loss is the
all-candidate objective L_actor = −E[(1/K) Σ_p Q_min(s, g, p, u_θ(s, p))];
actor gradients flow ONLY through u (p is a discrete index and the encoder
trunk is detached in the actor pass — the trunk is trained by the critic
loss, which keeps P2's latent semantics critic-grounded).

Training-level NaN covenant (global rule 6): non-finite loss or gradients
raise :class:`TrainingNaNError` BEFORE any optimizer step; the caller must
halt the run, preserve the last checkpoint, and report factually.  Silent
continuation is forbidden.

Forbidden here (P1 scope): response heads/aux losses (P3), HER, distributed
training, custom CUDA.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from dgcc.models.networks import (
    Actor,
    DELTA_SCALE,
    Encoder,
    K_NODES,
    TwinCritic,
    build_node_features,
)
from dgcc.tasks.domain import P1_LENGTH_M


class TrainingNaNError(RuntimeError):
    """Raised when loss/gradients go non-finite (halt + report; no silent continue)."""


@dataclass
class TD3Config:
    """§7 start values. Adjustable tier — changes require STEP_LOG entries."""

    gamma: float = 0.95
    tau: float = 0.005
    lr: float = 3.0e-4
    batch_size: int = 256
    replay_capacity: int = 500_000
    utd: int = 1
    warmup_transitions: int = 5_000
    grad_clip: float = 10.0
    policy_noise: float = 0.05
    noise_clip: float = 0.1
    exploration_u_sigma: float = 0.03
    eps_p_start: float = 1.0
    eps_p_end: float = 0.1
    eps_p_fraction: float = 0.3

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Pure target-computation pieces (unit-testable with hand-built tables)
# ---------------------------------------------------------------------------


def select_p_star(q1_target_candidates: torch.Tensor) -> torch.Tensor:
    """p′* selection over all K candidates using Q_target_1 ONLY (§7)."""

    if q1_target_candidates.ndim != 2:
        raise ValueError("q1_target_candidates must have shape (B, K)")
    return q1_target_candidates.argmax(dim=1)


def smooth_target_u(u: torch.Tensor, noise: torch.Tensor, *, noise_clip: float) -> torch.Tensor:
    """ũ′ = u + clip(noise, ±noise_clip), then clamp to the valid u box."""

    smoothed = u + noise.clamp(-noise_clip, noise_clip)
    delta = smoothed[..., :3].clamp(-DELTA_SCALE, DELTA_SCALE)
    lift = smoothed[..., 3:4].clamp(0.0, 1.0)
    return torch.cat([delta, lift], dim=-1)


def td_target(
    reward: torch.Tensor, done: torch.Tensor, gamma: float, q_min: torch.Tensor
) -> torch.Tensor:
    """y = r + γ·(1 − done)·min_i Q_target_i(s′, g, p′*, ũ′)."""

    return reward + gamma * (1.0 - done.to(reward.dtype)) * q_min


def epsilon_schedule(step: int, total_budget: int, config: TD3Config) -> float:
    """ε-greedy over p: eps_p_start → eps_p_end linearly over the first
    ``eps_p_fraction`` of the training budget (§7)."""

    horizon = max(1, int(total_budget * config.eps_p_fraction))
    frac = min(1.0, max(0.0, step / horizon))
    return float(config.eps_p_start + (config.eps_p_end - config.eps_p_start) * frac)


def u_tensor(delta: np.ndarray, lift: np.ndarray, device: torch.device) -> torch.Tensor:
    """Assemble the 4-dim u = [Δ, lift∈{0,1}] tensor from executed actions."""

    d = torch.as_tensor(np.asarray(delta), dtype=torch.float32, device=device)
    l = torch.as_tensor(np.asarray(lift), dtype=torch.float32, device=device).reshape(-1, 1)
    return torch.cat([d, l], dim=-1)


class TD3Agent:
    """Encoder + twin critic + per-point actor with §7 training semantics."""

    def __init__(
        self,
        config: TD3Config | None = None,
        *,
        device: str | torch.device = "cpu",
        length_m: float = P1_LENGTH_M,
    ) -> None:
        self.config = config or TD3Config()
        self.device = torch.device(device)
        self.length_m = float(length_m)

        self.encoder = Encoder().to(self.device)
        self.critic = TwinCritic().to(self.device)
        self.actor = Actor().to(self.device)
        self.encoder_target = copy.deepcopy(self.encoder).requires_grad_(False)
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False)
        self.actor_target = copy.deepcopy(self.actor).requires_grad_(False)

        # Encoder is trained by the critic loss only (see module docstring).
        self.critic_optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.critic.parameters()),
            lr=self.config.lr,
        )
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.lr)
        self.update_count = 0

    # ------------------------------------------------------------------
    # Feature/embedding helpers
    # ------------------------------------------------------------------

    def features(self, X: np.ndarray, G_curve: np.ndarray) -> torch.Tensor:
        feats, _ = build_node_features(X, G_curve, self.length_m)
        return torch.as_tensor(feats, dtype=torch.float32, device=self.device)

    @staticmethod
    def _flat_nodes(h: torch.Tensor) -> torch.Tensor:
        return h.reshape(-1, h.shape[-1])

    def _q_all_candidates(
        self, critic_head: nn.Module, h: torch.Tensor, u_all: torch.Tensor
    ) -> torch.Tensor:
        batch, k = h.shape[0], h.shape[1]
        q = critic_head(self._flat_nodes(h), u_all.reshape(batch * k, -1))
        return q.reshape(batch, k)

    # ------------------------------------------------------------------
    # §7 decoupled double-Q target (target networks ONLY)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_target(
        self,
        batch: dict[str, np.ndarray],
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        feats_next = self.features(batch["X_after"], batch["goal_curve"])
        h_next = self.encoder_target(feats_next)  # (B, K, 256)
        u_next_all = self.actor_target(h_next)  # (B, K, 4)

        q1_candidates = self._q_all_candidates(self.critic_target.q1, h_next, u_next_all)
        p_star = select_p_star(q1_candidates)
        arange = torch.arange(h_next.shape[0], device=self.device)
        h_star = h_next[arange, p_star]
        u_star = u_next_all[arange, p_star]

        noise = (
            torch.randn(u_star.shape, generator=generator, device=self.device)
            * self.config.policy_noise
        )
        u_tilde = smooth_target_u(u_star, noise, noise_clip=self.config.noise_clip)

        q1_t, q2_t = self.critic_target(h_star, u_tilde)
        q_min = torch.minimum(q1_t, q2_t)
        reward = torch.as_tensor(batch["reward"], dtype=torch.float32, device=self.device)
        done = torch.as_tensor(batch["done"], dtype=torch.bool, device=self.device)
        return td_target(reward, done, self.config.gamma, q_min)

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------

    def critic_update(
        self, batch: dict[str, np.ndarray], *, generator: torch.Generator | None = None
    ) -> dict[str, float]:
        y = self.compute_target(batch, generator=generator)

        feats = self.features(batch["X_before"], batch["goal_curve"])
        h = self.encoder(feats)
        arange = torch.arange(h.shape[0], device=self.device)
        p = torch.as_tensor(batch["p"], dtype=torch.long, device=self.device)
        h_p = h[arange, p]
        u = u_tensor(batch["delta"], batch["lift"], self.device)

        q1, q2 = self.critic(h_p, u)
        loss = nn.functional.mse_loss(q1, y) + nn.functional.mse_loss(q2, y)
        self._assert_finite_loss(loss, "critic loss")

        self.critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = self._clip_and_check_grads(
            list(self.encoder.parameters()) + list(self.critic.parameters()), "critic"
        )
        self.critic_optimizer.step()
        return {
            "critic_loss": float(loss.detach().cpu()),
            "critic_grad_norm": grad_norm,
            "target_mean": float(y.mean().cpu()),
            "q1_mean": float(q1.detach().mean().cpu()),
            "q2_mean": float(q2.detach().mean().cpu()),
        }

    def actor_update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        feats = self.features(batch["X_before"], batch["goal_curve"])
        with torch.no_grad():
            h = self.encoder(feats)  # trunk detached: actor grads flow via u only
        u_all = self.actor(h)  # (B, K, 4)
        q1 = self._q_all_candidates(self.critic.q1, h, u_all)
        q2 = self._q_all_candidates(self.critic.q2, h, u_all)
        q_min = torch.minimum(q1, q2)
        loss = -q_min.mean()  # (1/K) Σ_p over all candidates, all samples
        self._assert_finite_loss(loss, "actor loss")

        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = self._clip_and_check_grads(list(self.actor.parameters()), "actor")
        self.actor_optimizer.step()
        # Discard critic/encoder grads produced by the actor objective.
        self.critic_optimizer.zero_grad(set_to_none=True)
        return {"actor_loss": float(loss.detach().cpu()), "actor_grad_norm": grad_norm}

    def soft_update_targets(self) -> None:
        tau = self.config.tau
        for target, online in (
            (self.encoder_target, self.encoder),
            (self.critic_target, self.critic),
            (self.actor_target, self.actor),
        ):
            for tp, op in zip(target.parameters(), online.parameters(), strict=True):
                tp.data.mul_(1.0 - tau).add_(op.data, alpha=tau)

    def update(
        self, batch: dict[str, np.ndarray], *, generator: torch.Generator | None = None
    ) -> dict[str, float]:
        """One §7 update: critic (+encoder), actor, target soft update."""

        stats = self.critic_update(batch, generator=generator)
        stats.update(self.actor_update(batch))
        self.soft_update_targets()
        self.update_count += 1
        return stats

    # ------------------------------------------------------------------
    # Action selection (§7 exploration / deterministic eval)
    # ------------------------------------------------------------------

    def select_actions(
        self,
        X: np.ndarray,
        G_curve: np.ndarray,
        *,
        step: int,
        total_budget: int,
        rng: np.random.Generator,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Return (p, delta, lift) for a batch of states.

        Exploration: ε-greedy over p (linear 1.0→0.1 over the first 30% of
        the budget) + Gaussian noise σ=0.03 m on the delta.  Deterministic
        eval: argmax p by online Q1 with noise-free u.
        """

        with torch.no_grad():
            feats = self.features(X, G_curve)
            h = self.encoder(feats)
            u_all = self.actor(h)
            q1 = self._q_all_candidates(self.critic.q1, h, u_all)
            greedy_p = q1.argmax(dim=1).cpu().numpy()

        batch = feats.shape[0]
        p = greedy_p.copy()
        if not deterministic:
            eps = epsilon_schedule(step, total_budget, self.config)
            explore = rng.random(batch) < eps
            p[explore] = rng.integers(0, K_NODES, size=int(explore.sum()))

        arange = torch.arange(batch, device=self.device)
        u = u_all[arange, torch.as_tensor(p, dtype=torch.long, device=self.device)]
        u = u.cpu().numpy()
        delta = u[:, :3].copy()
        if not deterministic:
            delta = delta + rng.normal(0.0, self.config.exploration_u_sigma, size=delta.shape)
            delta = np.clip(delta, -DELTA_SCALE, DELTA_SCALE)
        lift = ["high" if value > 0.5 else "low" for value in u[:, 3]]
        return p.astype(int), delta.astype(float), lift

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": self.config.to_dict(),
                "update_count": self.update_count,
                "encoder": self.encoder.state_dict(),
                "critic": self.critic.state_dict(),
                "actor": self.actor.state_dict(),
                "encoder_target": self.encoder_target.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
            },
            target,
        )
        return target

    def load_checkpoint(self, path: Path | str) -> None:
        payload = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.update_count = int(payload["update_count"])
        self.encoder.load_state_dict(payload["encoder"])
        self.critic.load_state_dict(payload["critic"])
        self.actor.load_state_dict(payload["actor"])
        self.encoder_target.load_state_dict(payload["encoder_target"])
        self.critic_target.load_state_dict(payload["critic_target"])
        self.actor_target.load_state_dict(payload["actor_target"])
        self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        self.actor_optimizer.load_state_dict(payload["actor_optimizer"])

    # ------------------------------------------------------------------
    # Training-level NaN covenant (global rule 6)
    # ------------------------------------------------------------------

    @staticmethod
    def _assert_finite_loss(loss: torch.Tensor, name: str) -> None:
        if not torch.isfinite(loss).all():
            raise TrainingNaNError(f"non-finite {name}; halt run and report with last checkpoint")

    def _clip_and_check_grads(self, params: list[torch.Tensor], name: str) -> float:
        grad_norm = torch.nn.utils.clip_grad_norm_(params, self.config.grad_clip)
        if not torch.isfinite(grad_norm):
            raise TrainingNaNError(
                f"non-finite {name} gradient norm; halt run and report with last checkpoint"
            )
        return float(grad_norm)


__all__ = [
    "TD3Agent",
    "TD3Config",
    "TrainingNaNError",
    "epsilon_schedule",
    "select_p_star",
    "smooth_target_u",
    "td_target",
    "u_tensor",
]
