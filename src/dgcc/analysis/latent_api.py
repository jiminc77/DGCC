"""P1-M5 frozen-critic latent extraction API (P2 handover interface — P1.md M5).

This is the interface P2 probing consumes.  Contract highlights:

* Checkpoints load FROZEN: all modules in eval mode with
  ``requires_grad_(False)``; :meth:`FrozenLatentExtractor.parameter_sha256`
  gives a digest for before/after immutability checks.
* Extraction routes through the EXACT training code paths — features via
  :meth:`TD3Agent.features` (§6 input contract incl. canonical flip),
  encoder forward, and ``_QHead.forward(..., return_hidden=True)`` (the
  pre-planned latent hooks in :mod:`dgcc.models.networks`).  No
  reimplementation of any math.
* Layer names and shapes are pinned in :data:`LATENT_SPEC`; the same names
  appear in the output HDF5 of ``scripts/extract_latents.py`` and in
  ``docs/latent_api.md``.  Changing a name/shape is an interface break and
  requires updating all three in one commit.

Forbidden here (P1 scope): probes (ridge/MLP regressions), δm prediction,
latent interpretation claims.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dgcc.rl.td3 import TD3Agent, TD3Config, u_tensor
from dgcc.tasks.domain import RewardConstants

#: Latent name → shape contract ("B" = batch).  Synchronized with
#: docs/latent_api.md and the extract_latents.py output layout.
LATENT_SPEC: dict[str, tuple[Any, ...]] = {
    "encoder_h": ("B", 32, 256),  # per-node h_i (local 128 ⊕ global 128)
    "h_p": ("B", 256),  # selected-node embedding h_p
    "q1_trunk_hidden1": ("B", 256),  # critic Q1 post-LN post-ReLU layer 1
    "q1_trunk_hidden2": ("B", 256),  # critic Q1 post-LN post-ReLU layer 2
    "q2_trunk_hidden1": ("B", 256),
    "q2_trunk_hidden2": ("B", 256),
    "q1": ("B",),  # final Q heads on [h_p, u]
    "q2": ("B",),
    "q_min": ("B",),  # min(q1, q2)
    "flip_before": ("B",),  # canonical flip decision used for the features
}


def lift_to_float(lift: Any) -> np.ndarray:
    """Normalize lift inputs ("high"/"low" strings or 0/1 numerics) → float array."""

    arr = np.asarray(lift)
    if arr.dtype.kind in ("U", "S", "O"):
        return np.asarray([1.0 if str(v) == "high" else 0.0 for v in arr], dtype=np.float64)
    return arr.astype(np.float64)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FrozenLatentExtractor:
    """Frozen-critic latent extraction over arbitrary (s, g, p, u) batches."""

    def __init__(self, agent: TD3Agent, *, ckpt_path: Path, ckpt_sha256: str) -> None:
        self.agent = agent
        self.ckpt_path = Path(ckpt_path)
        self.ckpt_sha256 = ckpt_sha256
        self._enforce_frozen()

    def _frozen_modules(self):
        agent = self.agent
        return (
            agent.encoder,
            agent.critic,
            agent.actor,
            agent.encoder_target,
            agent.critic_target,
            agent.actor_target,
        )

    def _enforce_frozen(self) -> None:
        """(Re-)apply the frozen contract: eval mode + requires_grad False.

        Called at construction AND at every extract() so external
        ``.train()``/``requires_grad_`` tampering cannot silently leak into
        an extraction (QA finding, G006 review cycle).
        """

        for module in self._frozen_modules():
            module.eval()
            module.requires_grad_(False)

    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls, path: Path | str, *, device: str | torch.device = "cpu"
    ) -> "FrozenLatentExtractor":
        """Load a TD3 checkpoint frozen, reconstructing config/reward constants
        from the checkpoint payload itself (no external config file)."""

        ckpt_path = Path(path)
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        config = TD3Config(**payload["config"])
        constants = RewardConstants(**payload["reward_constants"])
        agent = TD3Agent(config, device=device, reward_constants=constants)
        agent.load_checkpoint(ckpt_path)
        return cls(agent, ckpt_path=ckpt_path, ckpt_sha256=sha256_file(ckpt_path))

    # ------------------------------------------------------------------

    def parameter_sha256(self) -> str:
        """Deterministic digest of all module state dicts (frozen-guarantee probe)."""

        digest = hashlib.sha256()
        agent = self.agent
        for name, module in (
            ("encoder", agent.encoder),
            ("critic", agent.critic),
            ("actor", agent.actor),
            ("encoder_target", agent.encoder_target),
            ("critic_target", agent.critic_target),
            ("actor_target", agent.actor_target),
        ):
            digest.update(name.encode())
            state = module.state_dict()
            for key in sorted(state):
                digest.update(key.encode())
                digest.update(state[key].detach().cpu().numpy().tobytes())
        return digest.hexdigest()

    def metadata(self) -> dict[str, Any]:
        """Provenance echoed into extraction outputs."""

        return {
            "ckpt_path": str(self.ckpt_path),
            "ckpt_sha256": self.ckpt_sha256,
            "agent": self.agent.to_dict(),
            "update_count": self.agent.update_count,
            "latent_spec": {k: list(v) for k, v in LATENT_SPEC.items()},
        }

    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract(
        self,
        X: np.ndarray,
        G_curve: np.ndarray,
        p: np.ndarray,
        delta: np.ndarray,
        lift: Any,
    ) -> dict[str, np.ndarray]:
        """Extract all :data:`LATENT_SPEC` tensors for one batch.

        Args:
            X: ``(B, 32, 3)`` centerlines (s).
            G_curve: ``(B, 32, 3)`` world-frame goal curves (g).
            p: ``(B,)`` selected node indices.
            delta: ``(B, 3)`` executed deltas.
            lift: ``(B,)`` "high"/"low" strings or 0/1 numerics.
        """

        self._enforce_frozen()
        agent = self.agent
        from dgcc.models.networks import goal_residual_flips

        x = np.asarray(X, dtype=float)
        g = np.asarray(G_curve, dtype=float)
        flips = goal_residual_flips(x, g, agent.length_m)
        feats = agent.features(x, g, flips)
        h = agent.encoder(feats)  # (B, 32, 256)
        arange = torch.arange(h.shape[0], device=agent.device)
        p_idx = torch.as_tensor(np.asarray(p), dtype=torch.long, device=agent.device)
        h_p = h[arange, p_idx]
        u = u_tensor(np.asarray(delta, dtype=float), lift_to_float(lift), agent.device)

        q1, hid1 = agent.critic.q1(h_p, u, return_hidden=True)
        q2, hid2 = agent.critic.q2(h_p, u, return_hidden=True)

        def npy(t: torch.Tensor) -> np.ndarray:
            return t.detach().cpu().numpy()

        out = {
            "encoder_h": npy(h),
            "h_p": npy(h_p),
            "q1_trunk_hidden1": npy(hid1["trunk_hidden1"]),
            "q1_trunk_hidden2": npy(hid1["trunk_hidden2"]),
            "q2_trunk_hidden1": npy(hid2["trunk_hidden1"]),
            "q2_trunk_hidden2": npy(hid2["trunk_hidden2"]),
            "q1": npy(q1),
            "q2": npy(q2),
            "q_min": npy(torch.minimum(q1, q2)),
            "flip_before": np.asarray(flips, dtype=bool),
        }
        for name, shape in LATENT_SPEC.items():
            expected = tuple(shape[1:])
            actual = out[name].shape[1:]
            if actual != expected:
                raise AssertionError(f"latent {name} shape {actual} != contract {expected}")
        return out


__all__ = ["FrozenLatentExtractor", "LATENT_SPEC", "lift_to_float", "sha256_file"]
