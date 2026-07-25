"""Sprint TD3 arms implemented as an adapter over the frozen TD3 baseline."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from dgcc.phi.dct import Phi_DCT
from dgcc.rl.selection import CONTACT_WEIGHT_BETA
from dgcc.rl.td3 import TD3Agent, TD3Config, select_p_star, u_tensor

SprintArm = Literal["bb", "bb-d2", "v1", "v1-d2", "matched", "random", "v2-dmm", "v2-d1m", "v2-d11", "v2-bgt"]
MATCHED_PROJECTION_SEED = 20260719
RANDOM_TARGET_SEED = 20260718


def matched_projection(seed: int = MATCHED_PROJECTION_SEED) -> torch.Tensor:
    """Create the fixed 24×256 Gaussian-QR projection used by the matched arm."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    matrix = torch.randn((256, 24), generator=generator)
    q, _ = torch.linalg.qr(matrix, mode="reduced")
    return q.transpose(0, 1).contiguous()
def random_target(seed: int = RANDOM_TARGET_SEED) -> torch.Tensor:
    """Create the fixed 24-channel Gaussian target used by the random arm."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(24, generator=generator)


class _RandomTarget(nn.Module):
    """Non-trainable holder for the fixed random target buffer."""

    def __init__(self, seed: int, device: torch.device) -> None:
        super().__init__()
        self.register_buffer("target", random_target(seed).to(device))


class _MatchedProjection(nn.Module):
    """Non-trainable holder so P is a module buffer, not an optimizer parameter."""

    def __init__(self, seed: int, device: torch.device) -> None:
        super().__init__()
        self.register_buffer("P", matched_projection(seed).to(device))

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
        projection_seed: int = MATCHED_PROJECTION_SEED,
        target_seed: int = RANDOM_TARGET_SEED,
        device: str | torch.device = "cpu",
        **kwargs: Any,
    ) -> None:
        if arm not in {"v1", "matched", "random"}:
            raise ValueError("SprintTD3Agent only implements arm='v1', arm='matched', or arm='random'")
        # This must remain first: TD3Agent consumes precisely the baseline RNG sequence.
        super().__init__(config, device=device, **kwargs)
        self.arm: SprintArm = arm
        self.aux_weight = float(aux_weight)
        if self.aux_weight < 0:
            raise ValueError("aux_weight must be non-negative")
        self.projection_seed = int(projection_seed)
        self.target_seed = int(target_seed)
        self.matched_projection = (
            _MatchedProjection(self.projection_seed, self.device) if arm == "matched" else None
        )
        self.random_target_buffer = (
            _RandomTarget(self.target_seed, self.device) if arm == "random" else None
        )

        # Construct and initialize outside the global RNG stream.  The derived seed
        # makes head initialization reproducible without perturbing callers' RNG state.
        derived_seed = (torch.initial_seed() ^ 0x535052494E54) & ((1 << 63) - 1)
        with torch.random.fork_rng(devices=[]):
            self.f_resp = ResponseHead().to(self.device)
            generator = torch.Generator(device=self.device).manual_seed(derived_seed)
            self._initialize_response_head(generator)
        self.critic_optimizer.add_param_group({"params": list(self.f_resp.parameters()), "lr": self.config.lr})

    @property
    def projection(self) -> torch.Tensor:
        """Fixed matched-dimension projection P."""
        if self.matched_projection is None:
            raise AttributeError("projection is only defined for arm='matched'")
        return self.matched_projection.P
    @property
    def random_target(self) -> torch.Tensor:
        """Fixed random 24-channel target used by the random arm."""
        if self.random_target_buffer is None:
            raise AttributeError("random_target is only defined for arm='random'")
        return self.random_target_buffer.target

    def _initialize_response_head(self, generator: torch.Generator) -> None:
        with torch.no_grad():
            for module in self.f_resp.modules():
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, a=5**0.5, generator=generator)
                    bound = 1 / module.weight.shape[1] ** 0.5
                    nn.init.uniform_(module.bias, -bound, bound, generator=generator)
    @torch.no_grad()
    def matched_target(self, batch: dict[str, np.ndarray]) -> torch.Tensor:
        """Return sg[P h_target(s′, p′*)] using baseline Q1 target selection."""
        feats_next = self.features(
            batch["X_after"], batch["goal_curve"], batch.get("flip_after")
        )
        h_next = self.encoder_target(feats_next)
        u_next_all = self.actor_target(h_next)
        q1_candidates = self._q_all_candidates(self.critic_target.q1, h_next, u_next_all)
        p_star = select_p_star(q1_candidates)
        arange = torch.arange(h_next.shape[0], device=self.device)
        return (self.projection @ h_next[arange, p_star].unsqueeze(-1)).squeeze(-1).detach()


    def critic_update(
        self,
        batch: dict[str, np.ndarray],
        *,
        generator: torch.Generator | None = None,
        feats_before: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Baseline critic loss plus the V1 DCT-response auxiliary loss."""
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
        q_loss = F.huber_loss(q1, y, delta=self.config.huber_delta) + F.huber_loss(
            q2, y, delta=self.config.huber_delta
        )
        # Do not evaluate f_resp at lambda=0: this preserves exact BB parity.
        if self.aux_weight == 0.0:
            aux_loss = torch.zeros((), device=self.device)
            loss = q_loss
        else:
            if self.arm == "v1":
                target = torch.as_tensor(
                    delta_m_from_batch(batch), dtype=torch.float32, device=self.device
                )
            elif self.arm == "matched":
                target = self.matched_target(batch)
            else:
                target = self.random_target.expand(h_p.shape[0], -1)
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
        stats = {
            "critic_loss": float(q_loss.detach().cpu()),
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
        if self.aux_weight == 0.0:
            return stats
        return {
            **stats,
            "aux_loss": float(aux_loss.detach().cpu()),
            "loss": float(loss.detach().cpu()),
        }

    def to_dict(self) -> dict[str, Any]:
        metadata = super().to_dict()
        sprint_arm = {
            "schema_version": self.schema_version,
            "arm": self.arm,
            "aux_weight": self.aux_weight,
        }
        if self.arm == "matched":
            sprint_arm["projection_seed"] = self.projection_seed
        if self.arm == "random":
            sprint_arm["target_seed"] = self.target_seed
        metadata["sprint_arm"] = sprint_arm
        return metadata

    def save_checkpoint(self, path: Path | str) -> Path:
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
                "encoder": self.encoder.state_dict(), "critic": self.critic.state_dict(), "actor": self.actor.state_dict(),
                "encoder_target": self.encoder_target.state_dict(), "critic_target": self.critic_target.state_dict(), "actor_target": self.actor_target.state_dict(),
                "sprint_arm": {
                    **{
                        "schema_version": self.schema_version,
                        "arm": self.arm,
                        "aux_weight": self.aux_weight,
                        "f_resp": self.f_resp.state_dict(),
                    },
                    **(
                        {"projection_seed": self.projection_seed}
                        if self.arm == "matched"
                        else {"target_seed": self.target_seed}
                        if self.arm == "random"
                        else {}
                    ),
                },
            },
            target,
        )
        return target

    def _load_sprint_checkpoint_payload(
        self,
        payload: Any,
        *,
        migration_mode: Literal["legacy_response_head_eval"] | None = None,
    ) -> None:
        if migration_mode is not None:
            raise ValueError(
                "legacy checkpoint migration is not supported for evaluation-only checkpoints"
            )
        if not isinstance(payload, dict):
            raise ValueError("checkpoint payload must be a mapping")
        sprint = payload.get("sprint_arm")
        if (
            not isinstance(sprint, dict)
            or sprint.get("schema_version") != self.schema_version
            or sprint.get("arm") != self.arm
            or "f_resp" not in sprint
        ):
            raise ValueError("incompatible sprint evaluation checkpoint")
        expected_response = self.f_resp.state_dict()
        saved_response = sprint["f_resp"]
        if not isinstance(saved_response, dict) or saved_response.keys() != expected_response.keys():
            raise ValueError("sprint response-head state does not match the evaluation model")
        if any(
            not isinstance(value, torch.Tensor) or value.shape != expected_response[key].shape
            for key, value in saved_response.items()
        ):
            raise ValueError("sprint response-head state does not match the evaluation model")
        aux_weight = sprint.get("aux_weight")
        if (
            isinstance(aux_weight, bool)
            or not isinstance(aux_weight, (int, float))
            or not np.isfinite(aux_weight)
            or aux_weight < 0
        ):
            raise ValueError("sprint checkpoint has invalid auxiliary-weight contract")
        if self.arm == "matched" and type(sprint.get("projection_seed")) is not int:
            raise ValueError("matched sprint checkpoint lacks projection_seed")
        if self.arm == "random" and type(sprint.get("target_seed")) is not int:
            raise ValueError("random sprint checkpoint lacks target_seed")

        self._load_evaluation_checkpoint_payload(payload)
        self.f_resp.load_state_dict(saved_response)
        self.aux_weight = float(aux_weight)
        if self.arm == "matched":
            self.projection_seed = int(sprint["projection_seed"])
            self.matched_projection = _MatchedProjection(
                self.projection_seed, self.device
            )
        if self.arm == "random":
            self.target_seed = int(sprint["target_seed"])
            self.random_target_buffer = _RandomTarget(
                self.target_seed, self.device
            )

    def load_checkpoint(
        self,
        path: Path | str,
        *,
        eval_only: bool = False,
        migration_mode: Literal["legacy_response_head_eval"] | None = None,
    ) -> None:
        """Load an evaluation-only sprint checkpoint."""
        if not eval_only:
            raise ValueError(
                "all checkpoints are evaluation-only; non-eval training continuation is forbidden"
            )
        self._load_sprint_checkpoint_payload(
            self._read_evaluation_checkpoint(path),
            migration_mode=migration_mode,
        )

def create_sprint_agent(
    arm: str,
    config: TD3Config | None = None,
    *,
    device: str | torch.device = "cpu",
    aux_weight: float = 1.0,
    projection_seed: int = MATCHED_PROJECTION_SEED,
    target_seed: int = RANDOM_TARGET_SEED,
    beta_contact: float = CONTACT_WEIGHT_BETA,
    bgt_manifest_path: Path | str | None = None,
    bgt_expected_manifest_sha256: str | None = None,
    bgt_checkpoint_sha256: str | None = None,
    bgt_panel_sha256: str | None = None,
    **kwargs: Any,
) -> TD3Agent:
    """Create a sprint or V2 arm from the shared training-driver seam."""
    if arm in {"bb-d2", "v1-d2"} and (
        config is None or config.policy_delay != 2
    ):
        raise ValueError(f"{arm} requires policy_delay == 2")
    if arm in {"bb", "bb-d2"}:
        return TD3Agent(config, device=device, **kwargs)
    if arm in {"v1", "v1-d2", "matched", "random"}:
        sprint_arm: SprintArm = "v1" if arm == "v1-d2" else arm
        return SprintTD3Agent(
            config,
            arm=sprint_arm,
            aux_weight=aux_weight,
            projection_seed=projection_seed,
            target_seed=target_seed,
            device=device,
            **kwargs,
        )
    if arm in {"v2-dmm", "v2-d1m", "v2-d11", "v2-bgt"}:
        from dgcc.rl.v2_arms import create_v2_agent

        return create_v2_agent(
            arm,
            config,
            device=device,
            aux_weight=aux_weight,
            beta_contact=beta_contact,
            bgt_manifest_path=bgt_manifest_path,
            bgt_expected_manifest_sha256=bgt_expected_manifest_sha256,
            bgt_checkpoint_sha256=bgt_checkpoint_sha256,
            bgt_panel_sha256=bgt_panel_sha256,
            **kwargs,
        )
    raise ValueError(f"unknown sprint arm {arm!r}")


__all__ = [
    "MATCHED_PROJECTION_SEED",
    "RANDOM_TARGET_SEED",
    "ResponseHead",
    "SprintTD3Agent",
    "create_sprint_agent",
    "delta_m_from_batch",
    "matched_projection",
    "random_target",
]
