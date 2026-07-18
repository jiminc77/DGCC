"""Sprint TD3 arms implemented as an adapter over the frozen TD3 baseline."""

from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from dgcc.phi.dct import Phi_DCT
from dgcc.rl.td3 import TD3Agent, TD3Config, u_tensor

SprintArm = Literal["bb", "v1"]


class ResponseHead(nn.Module):
    """V1 response predictor: ``[h_p, u]`` to DCT displacement."""

    def __init__(self) -> None:
        super().__init__()
        self.input = nn.Linear(260, 256)
        self.hidden = nn.Linear(256, 256)
        self.output = nn.Linear(256, 24)

    def z_resp(self, h_p: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        if h_p.ndim != 2 or h_p.shape[1] != 256:
            raise ValueError(f"h_p must have shape (B, 256), got {tuple(h_p.shape)}")
        if u.ndim != 2 or u.shape != (h_p.shape[0], 4):
            raise ValueError(f"u must have shape (B, 4), got {tuple(u.shape)}")
        return F.relu(self.hidden(F.relu(self.input(torch.cat((h_p, u), dim=1)))))

    def forward(self, h_p: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.output(self.z_resp(h_p, u))


def delta_m_from_batch(batch: dict[str, np.ndarray]) -> np.ndarray:
    """Return the contract target ``Phi_DCT(X_after) - Phi_DCT(X_before)``."""
    before = np.asarray(batch["X_before"])
    after = np.asarray(batch["X_after"])
    if before.ndim != 3 or after.shape != before.shape:
        raise ValueError("X_before and X_after must have matching shape (B, 32, 3)")
    return np.stack([Phi_DCT(xa) - Phi_DCT(xb) for xb, xa in zip(before, after, strict=True)])


class SprintTD3Agent(TD3Agent):
    """TD3 V1 adapter; baseline modules and target paths remain unchanged."""

    schema_version = 2

    def __init__(
        self,
        config: TD3Config | None = None,
        *,
        arm: SprintArm = "v1",
        aux_weight: float = 1.0,
        device: str | torch.device = "cpu",
        **kwargs: Any,
    ) -> None:
        if arm != "v1":
            raise ValueError("SprintTD3Agent only implements arm='v1'")
        # This must remain first: TD3Agent consumes precisely the baseline RNG sequence.
        super().__init__(config, device=device, **kwargs)
        self.arm: SprintArm = arm
        self.aux_weight = float(aux_weight)
        if self.aux_weight < 0:
            raise ValueError("aux_weight must be non-negative")

        # Construct and initialize outside the global RNG stream.  The derived seed
        # makes V1 initialization reproducible without perturbing callers' RNG state.
        derived_seed = (torch.initial_seed() ^ 0x535052494E54) & ((1 << 63) - 1)
        with torch.random.fork_rng(devices=[]):
            self.f_resp = ResponseHead().to(self.device)
            generator = torch.Generator(device=self.device).manual_seed(derived_seed)
            self._initialize_response_head(generator)
        self.critic_optimizer.add_param_group({"params": list(self.f_resp.parameters()), "lr": self.config.lr})

    def _initialize_response_head(self, generator: torch.Generator) -> None:
        with torch.no_grad():
            for module in self.f_resp.modules():
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, a=5**0.5, generator=generator)
                    bound = 1 / module.weight.shape[1] ** 0.5
                    nn.init.uniform_(module.bias, -bound, bound, generator=generator)

    def critic_update(
        self,
        batch: dict[str, np.ndarray],
        *,
        generator: torch.Generator | None = None,
        feats_before: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Baseline critic loss plus the V1 DCT-response auxiliary loss."""
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
        q_loss = F.huber_loss(q1, y, delta=self.config.huber_delta) + F.huber_loss(
            q2, y, delta=self.config.huber_delta
        )
        # Do not evaluate f_resp at lambda=0: this preserves exact BB parity.
        if self.aux_weight == 0.0:
            aux_loss = torch.zeros((), device=self.device)
            loss = q_loss
        else:
            target = torch.as_tensor(delta_m_from_batch(batch), dtype=torch.float32, device=self.device)
            aux_loss = F.mse_loss(self.f_resp(h_p, u), target)
            loss = q_loss + self.aux_weight * aux_loss
        self._assert_finite_loss(loss, "sprint critic loss")
        self.critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = self._clip_and_check_grads(
            list(self.encoder.parameters()) + list(self.critic.parameters()) + list(self.f_resp.parameters()),
            "sprint critic",
        )
        self.critic_optimizer.step()
        with torch.no_grad():
            td_error = (q1 - y).abs()
        return {
            "critic_loss": float(q_loss.detach().cpu()),
            "aux_loss": float(aux_loss.detach().cpu()),
            "loss": float(loss.detach().cpu()),
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

    def to_dict(self) -> dict[str, Any]:
        metadata = super().to_dict()
        metadata["sprint_arm"] = {"schema_version": self.schema_version, "arm": self.arm, "aux_weight": self.aux_weight}
        return metadata

    def save_checkpoint(self, path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": self.config.to_dict(),
                "reward_constants": asdict(self.reward_constants),
                "td_target_bound": dict(self.td_target_bound),
                "metadata": self.to_dict(),
                "update_count": self.update_count,
                "encoder": self.encoder.state_dict(), "critic": self.critic.state_dict(), "actor": self.actor.state_dict(),
                "encoder_target": self.encoder_target.state_dict(), "critic_target": self.critic_target.state_dict(), "actor_target": self.actor_target.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(), "actor_optimizer": self.actor_optimizer.state_dict(),
                "sprint_arm": {"schema_version": self.schema_version, "arm": self.arm, "aux_weight": self.aux_weight, "f_resp": self.f_resp.state_dict()},
            },
            target,
        )
        return target

    def load_checkpoint(self, path: Path | str) -> None:
        payload = torch.load(Path(path), map_location=self.device, weights_only=False)
        sprint = payload.get("sprint_arm")
        if sprint is None:
            # Legacy BB payload: preserve its optimizer state and retain freshly
            # initialized V1 parameters, which have no legacy counterpart.
            legacy = copy.deepcopy(payload)
            legacy_optimizer = legacy["critic_optimizer"]
            legacy_optimizer["param_groups"].append(copy.deepcopy(self.critic_optimizer.state_dict()["param_groups"][-1]))
            payload = legacy
        else:
            if sprint.get("schema_version") != self.schema_version or sprint.get("arm") != self.arm:
                raise ValueError("incompatible sprint checkpoint")
            self.f_resp.load_state_dict(sprint["f_resp"])
            self.aux_weight = float(sprint.get("aux_weight", self.aux_weight))
        self.update_count = int(payload["update_count"])
        self.encoder.load_state_dict(payload["encoder"]); self.critic.load_state_dict(payload["critic"]); self.actor.load_state_dict(payload["actor"])
        self.encoder_target.load_state_dict(payload["encoder_target"]); self.critic_target.load_state_dict(payload["critic_target"]); self.actor_target.load_state_dict(payload["actor_target"])
        self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        self.actor_optimizer.load_state_dict(payload["actor_optimizer"])


def create_sprint_agent(
    arm: str,
    config: TD3Config | None = None,
    *,
    device: str | torch.device = "cpu",
    aux_weight: float = 1.0,
    **kwargs: Any,
) -> TD3Agent:
    """Create a sprint arm; BB deliberately returns the unmodified baseline."""
    if arm == "bb":
        return TD3Agent(config, device=device, **kwargs)
    if arm == "v1":
        return SprintTD3Agent(config, arm="v1", aux_weight=aux_weight, device=device, **kwargs)
    if arm in {"matched", "random"}:
        raise NotImplementedError(f"sprint arm {arm!r} is not implemented yet")
    raise ValueError(f"unknown sprint arm {arm!r}")


__all__ = ["ResponseHead", "SprintTD3Agent", "create_sprint_agent", "delta_m_from_batch"]
