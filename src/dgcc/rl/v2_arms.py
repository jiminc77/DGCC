"""V2 selection-weighted actor and response-gated behavior candidates."""

from __future__ import annotations

import hashlib
import json
import os
import stat
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


class BGTAdmissionRequiredError(ValueError):
    """Raised when a BGT candidate lacks authenticated admission evidence."""

_OPERATOR_COMBINATIONS: dict[
    V2SelectionArm, tuple[ContactOperator, ContactOperator]
] = {
    "v2-dmm": ("qmin", "qmin"),
    "v2-d1m": ("q1", "qmin"),
    "v2-d11": ("q1", "q1"),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_pinned_json(path: Path, expected_sha256: str) -> Any:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise OSError("not a regular file")
            chunks: list[bytes] = []
            digest = hashlib.sha256()
            while chunk := os.read(fd, 1 << 20):
                digest.update(chunk)
                chunks.append(chunk)
        finally:
            os.close(fd)
    except OSError as error:
        raise BGTAdmissionRequiredError(
            "BGT admission required: manifest is missing or unsafe"
        ) from error
    if digest.hexdigest() != expected_sha256:
        raise BGTAdmissionRequiredError(
            "BGT admission required: manifest does not match pin"
        )
    try:
        return json.loads(b"".join(chunks))
    except json.JSONDecodeError as error:
        raise BGTAdmissionRequiredError(
            "BGT admission required: manifest is unreadable"
        ) from error


def _sha256_json(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _is_pinned_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
        and len(set(value)) > 1
    )


_BGT_DEVELOPMENT_LINEAGE_KEYS = (
    "development_split_path",
    "development_split_sha256",
    "development_split_role",
    "checkpoint_sha256",
    "panel_sha256",
    "config_sha256",
    "code_sha256",
)
BGT_ADMISSION_MATERIAL_KEYS = (
    "rank_calibration_sha256",
    "gpu_latency_sha256",
    "margin",
    "onset_transition",
    "development_lineage",
    "checkpoint_sha256",
    "panel_sha256",
    "config_sha256",
    "code_sha256",
)
BGT_ADMISSION_MANIFEST_KEYS = frozenset(
    (
        "schema_version",
        "bgt_admitted",
        *BGT_ADMISSION_MATERIAL_KEYS,
        "admission_sha256",
        "reason",
    )
)


def bgt_admission_material(value: dict[str, Any]) -> dict[str, Any]:
    """Canonical closed-schema material shared by admission producer and runtime."""
    if not isinstance(value, dict) or set(value) != set(BGT_ADMISSION_MATERIAL_KEYS):
        raise BGTAdmissionRequiredError(
            "BGT admission required: admission material is not closed schema"
        )
    return {key: value[key] for key in BGT_ADMISSION_MATERIAL_KEYS}


def validate_bgt_development_lineage(value: Any) -> dict[str, Any]:
    digest_keys = tuple(
        key for key in _BGT_DEVELOPMENT_LINEAGE_KEYS if key.endswith("_sha256")
    )
    if (
        not isinstance(value, dict)
        or set(value) != set(_BGT_DEVELOPMENT_LINEAGE_KEYS)
        or value.get("development_split_role") != "development_t2_split"
        or not isinstance(value.get("development_split_path"), str)
        or not value["development_split_path"]
        or any(not _is_pinned_sha256(value.get(key)) for key in digest_keys)
    ):
        raise BGTAdmissionRequiredError(
            "BGT admission required: invalid development lineage"
        )
    return value


def _validated_bgt_manifest_bytes(
    manifest_bytes: bytes,
    expected_manifest_sha256: str,
    checkpoint_sha256: str,
    panel_sha256: str,
    config_sha256: str,
    code_manifest_sha256: str,
) -> dict[str, Any]:
    """Validate exact authenticated admission-manifest bytes."""
    if not _is_pinned_sha256(expected_manifest_sha256):
        raise BGTAdmissionRequiredError(
            "BGT admission required: expected manifest SHA-256 is invalid"
        )
    if not isinstance(manifest_bytes, bytes) or (
        hashlib.sha256(manifest_bytes).hexdigest() != expected_manifest_sha256
    ):
        raise BGTAdmissionRequiredError(
            "BGT admission required: manifest does not match pin"
        )
    try:
        manifest = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BGTAdmissionRequiredError(
            "BGT admission required: manifest is unreadable"
        ) from error
    if not isinstance(manifest, dict) or set(manifest) != BGT_ADMISSION_MANIFEST_KEYS:
        raise BGTAdmissionRequiredError(
            "BGT admission required: manifest is not closed schema"
        )
    material = bgt_admission_material(
        {key: manifest[key] for key in BGT_ADMISSION_MATERIAL_KEYS}
    )
    lineage = validate_bgt_development_lineage(material["development_lineage"])
    required = {
        "schema_version": 1,
        "bgt_admitted": True,
        "checkpoint_sha256": checkpoint_sha256,
        "panel_sha256": panel_sha256,
        "config_sha256": config_sha256,
        "code_sha256": code_manifest_sha256,
        "reason": "rank and approved synchronized-GPU latency gates passed",
    }
    if any(manifest.get(key) != value for key, value in required.items()):
        raise BGTAdmissionRequiredError(
            "BGT admission required: manifest identities are incompatible"
        )
    for field in (
        "checkpoint_sha256",
        "panel_sha256",
        "config_sha256",
        "code_sha256",
    ):
        if lineage[field] != required[field]:
            raise BGTAdmissionRequiredError(
                "BGT admission required: development lineage is incompatible"
            )
    for field in (
        "rank_calibration_sha256",
        "gpu_latency_sha256",
        "admission_sha256",
        "checkpoint_sha256",
        "panel_sha256",
        "config_sha256",
        "code_sha256",
    ):
        if not _is_pinned_sha256(manifest.get(field)):
            raise BGTAdmissionRequiredError(f"BGT admission required: invalid {field}")
    margin = manifest.get("margin")
    onset = manifest.get("onset_transition")
    if (
        isinstance(margin, bool)
        or not isinstance(margin, (int, float))
        or not np.isfinite(margin)
        or margin < 0
    ):
        raise BGTAdmissionRequiredError("BGT admission required: invalid margin")
    if type(onset) is not int or onset < 0:
        raise BGTAdmissionRequiredError(
            "BGT admission required: invalid onset_transition"
        )
    if manifest["admission_sha256"] != _sha256_json(material):
        raise BGTAdmissionRequiredError("BGT admission required: admission hash is invalid")
    return manifest
def validate_bgt_manifest(
    manifest_path: Path | str,
    expected_manifest_sha256: str,
    checkpoint_sha256: str,
    panel_sha256: str,
    config_sha256: str,
    code_manifest_sha256: str,
) -> dict[str, Any]:
    """Offline/CPU validator for a locally pinned admission manifest."""
    return _validated_bgt_manifest_bytes(
        Path(manifest_path).read_bytes(),
        expected_manifest_sha256,
        checkpoint_sha256,
        panel_sha256,
        config_sha256,
        code_manifest_sha256,
    )


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
                f"beta_contact is charter-locked at {CONTACT_WEIGHT_BETA:.6f} Q units"
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
        self._assert_trainable()
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

    def load_checkpoint(
        self,
        path: Path | str,
        *,
        eval_only: bool = False,
        migration_mode: Literal["legacy_response_head_eval"] | None = None,
    ) -> None:
        if not eval_only:
            raise ValueError(
                "all checkpoints are evaluation-only; non-eval training continuation is forbidden"
            )
        payload = self._read_evaluation_checkpoint(path)
        if not isinstance(payload, dict):
            raise ValueError("checkpoint payload must be a mapping")
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
        self._load_sprint_checkpoint_payload(
            payload, migration_mode=migration_mode
        )


class BGTAgent(SprintTD3Agent):
    """Q1-margin-gated, deterministic top-two response tie-break candidate."""

    v2_schema_version = 1

    def __init__(
        self,
        config: TD3Config | None = None,
        *,
        manifest_bytes: bytes | None = None,
        manifest_path: Path | str | None = None,
        expected_manifest_sha256: str,
        checkpoint_sha256: str,
        panel_sha256: str,
        code_manifest_sha256: str,
        device: str | torch.device = "cpu",
        **kwargs: Any,
    ) -> None:
        super().__init__(config, arm="v1", device=device, **kwargs)
        config = config or TD3Config()
        if manifest_bytes is not None:
            manifest = _validated_bgt_manifest_bytes(
                manifest_bytes,
                expected_manifest_sha256,
                checkpoint_sha256,
                panel_sha256,
                _sha256_json(config.to_dict()),
                code_manifest_sha256,
            )
        elif manifest_path is not None:
            manifest = validate_bgt_manifest(
                manifest_path,
                expected_manifest_sha256,
                checkpoint_sha256,
                panel_sha256,
                _sha256_json(config.to_dict()),
                code_manifest_sha256,
            )
        else:
            raise BGTAdmissionRequiredError(
                "BGT admission required: authenticated manifest bytes are missing"
            )
        self.candidate_id = "v2-bgt"
        self.bgt_margin = float(manifest["margin"])
        self.bgt_onset_transition = int(manifest["onset_transition"])
        self.calibration_sha256 = manifest["admission_sha256"]
        self.manifest_sha256 = expected_manifest_sha256
        self.calibration_checkpoint_sha256 = checkpoint_sha256
        if not np.isfinite(self.bgt_margin) or self.bgt_margin < 0:
            raise ValueError("BGT manifest margin must be finite and non-negative")
        if self.bgt_onset_transition < 0:
            raise ValueError("BGT manifest onset_transition must be non-negative")
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
        return self._select_actions_impl(
            X,
            G_curve,
            step=step,
            total_budget=total_budget,
            rng=rng,
            deterministic=deterministic,
            return_info=return_info,
            selector_operator=selector_operator,
            benchmark_force_all_eligible=False,
        )


    @torch.no_grad()
    def _select_actions_impl(
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
        benchmark_force_all_eligible: bool = False,
    ) -> (
        tuple[np.ndarray, np.ndarray, list[str]]
        | tuple[np.ndarray, np.ndarray, list[str], dict[str, np.ndarray]]
    ):
        if benchmark_force_all_eligible and not isinstance(self, BenchmarkBGTAgent):
            raise BGTAdmissionRequiredError(
                "benchmark force-all eligibility is restricted to BenchmarkBGTAgent"
            )
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
        eligible = (
            torch.ones_like(normalized_margin, dtype=torch.bool)
            if benchmark_force_all_eligible
            else (normalized_margin < self.bgt_margin)
            & (step >= self.bgt_onset_transition)
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
            eligible_state = np.asarray(X)[eligible_rows.cpu().numpy()]
            eligible_goals = aligned_goals[eligible_rows.cpu().numpy()]
            residual = torch.as_tensor(
                np.stack(
                    [
                        Phi_DCT(goal) - Phi_DCT(state)
                        for state, goal in zip(eligible_state, eligible_goals, strict=True)
                    ]
                ),
                dtype=h.dtype,
                device=self.device,
            )
            residual_pairs = residual[:, None, :].expand(-1, 2, -1)
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

    def load_checkpoint(
        self,
        path: Path | str,
        *,
        eval_only: bool = False,
        expected_checkpoint_sha256: str | None = None,
        migration_mode: Literal["legacy_response_head_eval"] | None = None,
    ) -> None:
        if not eval_only:
            raise ValueError("BGT checkpoints may only be loaded for evaluation")
        if (
            expected_checkpoint_sha256 is not None
            and not _is_pinned_sha256(expected_checkpoint_sha256)
        ):
            raise ValueError("BGT evaluation checkpoint SHA-256 pin is invalid")
        payload = self._read_evaluation_checkpoint(
            path, expected_sha256=expected_checkpoint_sha256
        )
        if not isinstance(payload, dict):
            raise ValueError("checkpoint payload must be a mapping")
        metadata = payload.get("metadata", {}).get("v2_arm")
        if metadata != self.to_dict()["v2_arm"]:
            raise ValueError("incompatible or missing V2-BGT checkpoint metadata")
        self._load_sprint_checkpoint_payload(
            payload, migration_mode=migration_mode
        )
class BenchmarkBGTAgent(BGTAgent):
    """Benchmark-only BGT implementation with no admission or checkpoint API."""

    def __init__(
        self,
        config: TD3Config | None = None,
        *,
        margin: float,
        onset_transition: int,
        device: str | torch.device = "cpu",
        **kwargs: Any,
    ) -> None:
        SprintTD3Agent.__init__(self, config, arm="v1", device=device, **kwargs)
        if not isinstance(margin, (int, float)) or not np.isfinite(margin) or margin < 0:
            raise ValueError("benchmark BGT margin must be finite and non-negative")
        if not isinstance(onset_transition, int) or onset_transition < 0:
            raise ValueError("benchmark BGT onset_transition must be non-negative")
        self.candidate_id = "v2-bgt-benchmark-only"
        self.bgt_margin = float(margin)
        self.bgt_onset_transition = onset_transition
        self.calibration_sha256 = None
        self.manifest_sha256 = None

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
        benchmark_force_all_eligible: bool = False,
    ) -> (
        tuple[np.ndarray, np.ndarray, list[str]]
        | tuple[np.ndarray, np.ndarray, list[str], dict[str, np.ndarray]]
    ):
        return self._select_actions_impl(
            X,
            G_curve,
            step=step,
            total_budget=total_budget,
            rng=rng,
            deterministic=deterministic,
            return_info=return_info,
            selector_operator=selector_operator,
            benchmark_force_all_eligible=benchmark_force_all_eligible,
        )
    def load_checkpoint(self, path: Path | str) -> None:
        raise BGTAdmissionRequiredError(
            "benchmark-only BGT cannot load a production checkpoint"
        )

    def load_benchmark_checkpoint(self, path: Path | str) -> None:
        """Load weights solely for an in-process benchmark measurement."""
        SprintTD3Agent.load_checkpoint(self, path, eval_only=True)

    def to_dict(self) -> dict[str, Any]:
        raise BGTAdmissionRequiredError(
            "benchmark-only BGT cannot be serialized as a production agent"
        )


def create_v2_agent(
    arm: V2Arm,
    config: TD3Config | None = None,
    *,
    device: str | torch.device = "cpu",
    aux_weight: float = 1.0,
    beta_contact: float = CONTACT_WEIGHT_BETA,
    bgt_manifest_bytes: bytes | None = None,
    bgt_manifest_path: Path | str | None = None,
    bgt_expected_manifest_sha256: str | None = None,
    bgt_checkpoint_sha256: str | None = None,
    bgt_panel_sha256: str | None = None,
    bgt_code_manifest_sha256: str | None = None,
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
            bgt_manifest_bytes is None
            or bgt_expected_manifest_sha256 is None
            or bgt_checkpoint_sha256 is None
            or bgt_panel_sha256 is None
            or bgt_code_manifest_sha256 is None
        ):
            raise BGTAdmissionRequiredError(
                "BGT admission required: provide authenticated manifest bytes and independent pin"
            )
        return BGTAgent(
            config,
            manifest_bytes=bgt_manifest_bytes,
            expected_manifest_sha256=bgt_expected_manifest_sha256,
            checkpoint_sha256=bgt_checkpoint_sha256,
            panel_sha256=bgt_panel_sha256,
            code_manifest_sha256=bgt_code_manifest_sha256,
            aux_weight=aux_weight,
            device=device,
            **kwargs,
        )
    raise ValueError(f"unknown V2 arm {arm!r}")


__all__ = [
    "BGTAdmissionRequiredError",
    "BGTAgent",
    "BenchmarkBGTAgent",
    "BGT_ADMISSION_MANIFEST_KEYS",
    "BGT_ADMISSION_MATERIAL_KEYS",
    "SelectionWeightedTD3Agent",
    "V2Arm",
    "V2SelectionArm",
    "validate_bgt_manifest",
    "validate_bgt_development_lineage",
    "bgt_admission_material",
    "create_v2_agent",
    "selection_weighted_actor_loss",
]
