"""P1 §7 TD3-style off-policy learner with double-Q decoupling.

Target semantics (P1.md §7 + verdict F1 — target networks only, timeout
truncation bootstraps):

    terminal = done & ~truncated
    y = r + γ · (1 − terminal) · min(Q_target_1, Q_target_2)(s′, g, p′*, ũ′)
    p′* = argmax_p Q_target_1(s′, g, p, u_target(s′, p))       # selection: Q_target_1
    ũ′  = u_target(s′, p′*) + clip(N(0, 0.05), ±0.1)           # target policy smoothing

Online critics NEVER appear in the target computation.  Verdict S1 makes Huber
loss the PRIMARY residual-containment mechanism for both critics.  Verdict S3
adds only a derived TD-target clamp backstop from the immutable reward envelope;
a nonzero steady clamp-hit rate is pre-registered evidence for the
intrinsic-explosion antithesis.  The actor loss is the all-candidate objective
L_actor = −E[(1/K) Σ_p Q_min(s, g, p, u_θ(s, p))]; actor gradients flow ONLY
through u (p is a discrete index and the encoder trunk is detached in the actor
pass — the trunk is trained by the critic loss, which keeps P2's latent
semantics critic-grounded).

Training-level NaN covenant (global rule 6): non-finite loss or gradients
raise :class:`TrainingNaNError` BEFORE any optimizer step; the caller must
halt the run, preserve the last checkpoint, and report factually.  Silent
continuation is forbidden.

Forbidden here (P1 scope): response heads/aux losses (P3), HER, distributed
training, custom CUDA.
"""

from __future__ import annotations

import copy
import hashlib
import os
import stat
from decimal import Decimal
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from dgcc.models.networks import (
    Actor,
    DELTA_SCALE,
    Encoder,
    K_NODES,
    TwinCritic,
    build_node_features,
)
from dgcc.tasks.domain import P1_LENGTH_M, RewardConstants


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
    policy_delay: int = 1
    warmup_transitions: int = 5_000
    grad_clip: float = 10.0
    huber_delta: float = 1.0
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


def derive_td_target_bound(
    constants: RewardConstants, gamma: float, d_max: float = 7.0
) -> dict[str, float]:
    """Derive the S3 TD-target clamp from P1 reward constants.

    F3/S3 fixes the goal-node norm bound at 4.0 m, giving ``D_max=7.0`` for
    the reward envelope.  The bound is a backstop; S1 Huber loss is the primary
    residual-containment mechanism.
    """

    alpha = Decimal(str(constants.alpha))
    c_step = Decimal(str(constants.c_step))
    r_succ = Decimal(str(constants.r_succ))
    distance = Decimal(str(d_max))
    discount_gap = Decimal("1") - Decimal(str(gamma))
    if distance <= 0:
        raise ValueError("d_max must be positive")
    if discount_gap <= 0:
        raise ValueError("gamma must be less than 1")

    r_max = alpha * distance - c_step + r_succ
    r_min = -alpha * distance - c_step
    v_max = r_max / discount_gap
    return {
        "d_max": float(distance),
        "r_min": float(r_min),
        "r_max": float(r_max),
        "v_max": float(v_max),
    }


def td_target(
    reward: torch.Tensor,
    done: torch.Tensor,
    gamma: float,
    q_min: torch.Tensor,
    *,
    truncated: torch.Tensor,
    v_max: float | None = None,
) -> torch.Tensor:
    """P1.md §7 TD target with verdict F1 timeout masking and S3 clamp.

    ``terminal = done & ~truncated``; timeout truncations bootstrap, while true
    success terminations zero the bootstrap term.  Training batches must provide
    ``truncated`` explicitly; treating a missing field as false changes targets.
    """
    if reward.shape != done.shape or reward.shape != truncated.shape or reward.shape != q_min.shape:
        raise ValueError("reward, done, truncated, and q_min must have identical shapes")
    terminal = done.to(torch.bool) & ~truncated.to(torch.bool)
    y = reward + gamma * (1.0 - terminal.to(reward.dtype)) * q_min
    if v_max is not None:
        y = torch.clamp(y, -float(v_max), float(v_max))
    return y


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
        reward_constants: RewardConstants | None = None,
    ) -> None:
        self.config = config or TD3Config()
        if self.config.policy_delay < 1:
            raise ValueError("policy_delay must be at least 1")
        self.device = torch.device(device)
        self.length_m = float(length_m)
        self.reward_constants = reward_constants or RewardConstants()
        self.td_target_bound = derive_td_target_bound(
            self.reward_constants, self.config.gamma
        )
        self.last_clamp_hit_frac = 0.0

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
        self.actor_update_count = 0
        self.target_update_count = 0
        self.fresh_restart_only = False

    def _assert_trainable(self) -> None:
        """Reject every direct optimizer or target mutation after eval-only loading."""
        if self.fresh_restart_only:
            raise RuntimeError(
                "checkpoint was loaded eval-only and is fresh_restart_only; "
                "create a new agent before training"
            )

    # ------------------------------------------------------------------
    # Feature/embedding helpers
    # ------------------------------------------------------------------

    def features(
        self,
        X: np.ndarray,
        G_curve: np.ndarray,
        flips: np.ndarray | None = None,
    ) -> torch.Tensor:
        """§6 input features; ``flips`` accepts replay-cached decisions.

        Note (M1 gate LOW, deliberate deviation record): the flip decision for
        s′ features is computed from X_after itself, whereas the M5R2
        MEASUREMENT convention fixes the flip once per transition from
        X_before.  For encoder inputs each state is aligned by its own
        canonical decision so the goal-conditioning is well-defined per state;
        near the flip boundary the residual channel may change orientation
        between s and s′ — accepted and documented, not a metric-path change.
        """

        feats, _ = build_node_features(X, G_curve, self.length_m, flips=flips)
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
        """Compute the P1.md §7/F1 target and apply the S3 derived clamp."""
        self._validate_target_batch(batch)
        feats_next = self.features(
            batch["X_after"], batch["goal_curve"], batch.get("flip_after")
        )
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
        done = torch.as_tensor(np.asarray(batch["done"], dtype=bool), dtype=torch.bool, device=self.device)
        truncated = torch.as_tensor(
            np.asarray(batch["truncated"], dtype=bool), dtype=torch.bool, device=self.device
        )
        unclamped = td_target(
            reward,
            done,
            self.config.gamma,
            q_min,
            truncated=truncated,
        )
        v_max = float(self.td_target_bound["v_max"])
        hits = unclamped.abs() >= v_max
        y = torch.clamp(unclamped, -v_max, v_max)
        self.last_clamp_hit_frac = float(hits.to(torch.float32).mean().cpu())
        return y

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------

    def critic_update(
        self,
        batch: dict[str, np.ndarray],
        *,
        generator: torch.Generator | None = None,
        feats_before: torch.Tensor | None = None,
    ) -> dict[str, float]:
        self._assert_trainable()
        y = self.compute_target(batch, generator=generator)

        feats = (
            self.features(batch["X_before"], batch["goal_curve"], batch.get("flip_before"))
            if feats_before is None
            else feats_before
        )
        h = self.encoder(feats)
        arange = torch.arange(h.shape[0], device=self.device)
        p = torch.as_tensor(batch["p"], dtype=torch.long, device=self.device)
        h_p = h[arange, p]
        u = u_tensor(batch["delta"], batch["lift"], self.device)

        q1, q2 = self.critic(h_p, u)
        loss = F.huber_loss(q1, y, delta=self.config.huber_delta) + F.huber_loss(
            q2, y, delta=self.config.huber_delta
        )
        self._assert_finite_loss(loss, "critic loss")

        self.critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = self._clip_and_check_grads(
            list(self.encoder.parameters()) + list(self.critic.parameters()), "critic"
        )
        self.critic_optimizer.step()
        with torch.no_grad():
            td_error = (q1 - y).abs()
        return {
            "critic_loss": float(loss.detach().cpu()),
            "critic_grad_norm": grad_norm,
            "target_mean": float(y.mean().cpu()),
            "td_target_clamp_hit_frac": self.last_clamp_hit_frac,
            "q1_mean": float(q1.detach().mean().cpu()),
            "q2_mean": float(q2.detach().mean().cpu()),
            "q1_std": float(q1.detach().std(unbiased=False).cpu()),
            "td_error_mean": float(td_error.mean().cpu()),
            "td_error_p95": float(td_error.quantile(0.95).cpu()),
            "td_error_max": float(td_error.max().cpu()),
        }

    def actor_update(
        self,
        batch: dict[str, np.ndarray],
        *,
        feats_before: torch.Tensor | None = None,
    ) -> dict[str, float]:
        self._assert_trainable()
        feats = (
            self.features(batch["X_before"], batch["goal_curve"], batch.get("flip_before"))
            if feats_before is None
            else feats_before
        )
        with torch.no_grad():
            h = self.encoder(feats)  # trunk detached: actor grads flow via u only
        u_all = self.actor(h)  # (B, K, 4)
        critic_params = list(self.critic.parameters())
        requires_grad = [parameter.requires_grad for parameter in critic_params]
        try:
            for parameter in critic_params:
                parameter.requires_grad_(False)
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
        finally:
            for parameter, was_required in zip(critic_params, requires_grad, strict=True):
                parameter.requires_grad_(was_required)
            # Discard critic/encoder grads produced by the actor objective.
            self.critic_optimizer.zero_grad(set_to_none=True)
        return {"actor_loss": float(loss.detach().cpu()), "actor_grad_norm": grad_norm}

    def soft_update_targets(self) -> None:
        self._assert_trainable()
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
        """Update critic every step and actor/targets at ``policy_delay`` cadence."""
        self._assert_trainable()

        feats_before = self.features(
            batch["X_before"], batch["goal_curve"], batch.get("flip_before")
        )
        stats = self.critic_update(batch, generator=generator, feats_before=feats_before)
        actor_due = (self.update_count + 1) % self.config.policy_delay == 0
        stats["actor_updated"] = float(actor_due)
        if actor_due:
            stats.update(self.actor_update(batch, feats_before=feats_before))
            self.actor_update_count += 1
            self.soft_update_targets()
            self.target_update_count += 1
        self.update_count += 1
        return stats
    @staticmethod
    def _validate_target_batch(batch: dict[str, np.ndarray]) -> None:
        """Reject target-contract violations before an update can mutate state."""
        missing = {"reward", "done", "truncated"} - batch.keys()
        if missing:
            raise ValueError(
                "training batch lacks required target fields: " + ", ".join(sorted(missing))
            )
        shapes = {
            name: np.asarray(batch[name]).shape for name in ("reward", "done", "truncated")
        }
        if len(set(shapes.values())) != 1 or len(shapes["reward"]) != 1:
            raise ValueError(
                "reward, done, and truncated must be one-dimensional with identical shapes: "
                + ", ".join(f"{name}={shape}" for name, shape in shapes.items())
            )
        reward = np.asarray(batch["reward"])
        for name in ("done", "truncated"):
            value = np.asarray(batch[name])
            if value.dtype.kind == "b":
                continue
            if value.dtype.kind not in "iuIf" or not np.isfinite(value).all() or not np.isin(value, (0, 1)).all():
                raise ValueError(f"{name} must contain only boolean or numeric 0/1 values")
        if not np.isfinite(reward).all():
            raise TrainingNaNError("reward is non-finite")
        if "X_after" in batch and np.asarray(batch["X_after"]).shape[0] != shapes["reward"][0]:
            raise ValueError("X_after batch length must match reward, done, and truncated")

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
        return_info: bool = False,
        selector_operator: str = "q1",
    ) -> tuple[np.ndarray, np.ndarray, list[str]] | tuple[np.ndarray, np.ndarray, list[str], dict[str, np.ndarray]]:
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
            if selector_operator == "q1":
                selector_scores = q1
            elif selector_operator == "qmin":
                q2 = self._q_all_candidates(self.critic.q2, h, u_all)
                selector_scores = torch.minimum(q1, q2)
            else:
                raise ValueError("selector_operator must be 'q1' or 'qmin'")
            greedy_p = selector_scores.argmax(dim=1).cpu().numpy()

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
        if return_info:
            info = {
                "q1_candidates": q1.cpu().numpy(),
                "selector_operator": np.asarray([selector_operator] * batch),
                "u_executed": np.column_stack([delta, u[:, 3]]),
            }
            return p.astype(int), delta.astype(float), lift, info
        return p.astype(int), delta.astype(float), lift

    @torch.no_grad()
    def selection_panel(
        self, X: np.ndarray, G_curve: np.ndarray, *, include_lift: bool = True
    ) -> tuple[dict[str, float], Any]:
        """Evaluate common selector diagnostics on an ordered development panel."""

        from dgcc.rl.selection import (
            lift_statistics,
            selection_snapshot,
            selection_statistics,
            uniform_contact_weights,
        )

        h = self.encoder(self.features(X, G_curve))
        u_all = self.actor(h)
        q1 = self._q_all_candidates(self.critic.q1, h, u_all)
        q2 = self._q_all_candidates(self.critic.q2, h, u_all)
        weights = uniform_contact_weights(q1)
        selected = q1.argmax(dim=1)
        stats = selection_statistics(q1, q2, weights, selected=selected)
        if include_lift:
            stats.update(lift_statistics(self, h, u_all, q1, q2, selected))
        return stats, selection_snapshot(q1, q2, weights).cpu()


    @torch.no_grad()
    def q_min_executed(
        self,
        X: np.ndarray,
        G_curve: np.ndarray,
        p: np.ndarray,
        delta: np.ndarray,
        lift: np.ndarray,
    ) -> np.ndarray:
        """min(Q1, Q2) of the ONLINE critics for executed actions.

        Used by §8 diagnostics for the overestimation gap
        (Q(s, a) vs realized discounted return on eval episodes).
        """

        feats = self.features(X, G_curve)
        h = self.encoder(feats)
        arange = torch.arange(h.shape[0], device=self.device)
        h_p = h[arange, torch.as_tensor(np.asarray(p), dtype=torch.long, device=self.device)]
        u = u_tensor(delta, lift, self.device)
        q1, q2 = self.critic(h_p, u)
        return torch.minimum(q1, q2).cpu().numpy()

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Agent metadata echoed in evaluation-only checkpoints and diagnostics."""
        return {
            "config": self.config.to_dict(),
            "length_m": self.length_m,
            "reward_constants": asdict(self.reward_constants),
            "td_target_bound": dict(self.td_target_bound),
            "resume": {
                "fresh_restart_only": True,
                "reason": "run, replay, RNG, and environment state are not serialized",
            },
        }

    def save_checkpoint(self, path: Path | str) -> Path:
        """Save model weights for evaluation only, never as a training resume."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "checkpoint_schema": 1,
                "checkpoint_type": "evaluation_only",
                "config": self.config.to_dict(),
                "length_m": self.length_m,
                "reward_constants": asdict(self.reward_constants),
                "td_target_bound": dict(self.td_target_bound),
                "metadata": self.to_dict(),
                "encoder": self.encoder.state_dict(),
                "critic": self.critic.state_dict(),
                "actor": self.actor.state_dict(),
                "encoder_target": self.encoder_target.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_target": self.actor_target.state_dict(),
            },
            target,
        )
        return target

    def _read_evaluation_checkpoint(
        self,
        path: Path | str,
        *,
        expected_sha256: str | None = None,
    ) -> Any:
        checkpoint = Path(path)
        try:
            fd = os.open(checkpoint, os.O_RDONLY | os.O_NOFOLLOW)
            handle = os.fdopen(fd, "rb")
        except OSError as error:
            raise ValueError("checkpoint path is missing or unsafe") from error
        with handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("checkpoint path is not a regular file")
            if expected_sha256 is not None:
                digest = hashlib.sha256()
                while chunk := handle.read(1 << 20):
                    digest.update(chunk)
                if digest.hexdigest() != expected_sha256:
                    raise ValueError(
                        "evaluation checkpoint bytes do not match SHA-256 pin"
                    )
                handle.seek(0)
            return torch.load(
                handle, map_location=self.device, weights_only=True
            )

    def _load_evaluation_checkpoint_payload(self, payload: Any) -> None:
        self._validate_evaluation_checkpoint(payload)

        # Validate every state mapping before loading any module, avoiding partial
        # model mutation from a malformed later state mapping.
        model_state = (
            ("encoder", self.encoder),
            ("critic", self.critic),
            ("actor", self.actor),
            ("encoder_target", self.encoder_target),
            ("critic_target", self.critic_target),
            ("actor_target", self.actor_target),
        )
        for name, module in model_state:
            saved_state = payload[name]
            expected_state = module.state_dict()
            if not isinstance(saved_state, dict) or saved_state.keys() != expected_state.keys():
                raise ValueError(f"checkpoint {name} state does not match the evaluation model")
            if any(
                not isinstance(value, torch.Tensor)
                or value.shape != expected_state[key].shape
                for key, value in saved_state.items()
            ):
                raise ValueError(f"checkpoint {name} state does not match the evaluation model")

        for name, module in model_state:
            module.load_state_dict(payload[name])
        self.fresh_restart_only = True
        self.update_count = 0
        self.actor_update_count = 0
        self.target_update_count = 0

    def load_checkpoint(self, path: Path | str, *, eval_only: bool = False) -> None:
        """Load an evaluation-only checkpoint.

        Checkpoints intentionally exclude run, replay, RNG, and environment state,
        so all continuation attempts are rejected, including payloads with
        resume-looking keys.
        """
        if not eval_only:
            raise ValueError(
                "all checkpoints are evaluation-only; non-eval training continuation is forbidden"
            )
        self._load_evaluation_checkpoint_payload(
            self._read_evaluation_checkpoint(path)
        )

    def _validate_evaluation_checkpoint(self, payload: Any) -> None:
        """Validate immutable evaluation contracts before mutating model state."""
        if not isinstance(payload, dict):
            raise ValueError("checkpoint payload must be a mapping")
        required = {
            "checkpoint_schema",
            "checkpoint_type",
            "config",
            "length_m",
            "reward_constants",
            "td_target_bound",
            "metadata",
            "encoder",
            "critic",
            "actor",
            "encoder_target",
            "critic_target",
            "actor_target",
        }
        missing = sorted(required - payload.keys())
        if missing:
            raise ValueError("checkpoint lacks required evaluation state: " + ", ".join(missing))
        if payload["checkpoint_schema"] != 1 or payload["checkpoint_type"] != "evaluation_only":
            raise ValueError("checkpoint is not schema-1 evaluation-only")
        metadata = payload["metadata"]
        resume = metadata.get("resume") if isinstance(metadata, dict) else None
        if not isinstance(resume, dict) or resume.get("fresh_restart_only") is not True:
            raise ValueError("checkpoint must explicitly prohibit training continuation")
        saved_config = payload["config"]
        if not isinstance(saved_config, dict) or saved_config.get("gamma") != self.config.gamma:
            raise ValueError("checkpoint target-discount contract does not match this agent")
        if payload["length_m"] != self.length_m:
            raise ValueError("checkpoint length contract does not match this agent")
        if payload["reward_constants"] != asdict(self.reward_constants):
            raise ValueError("checkpoint reward contract does not match this agent")
        if payload["td_target_bound"] != self.td_target_bound:
            raise ValueError("checkpoint target contract does not match this agent")

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
    "derive_td_target_bound",
    "epsilon_schedule",
    "select_p_star",
    "smooth_target_u",
    "td_target",
    "u_tensor",
]
