"""P1 §5 pinned task domain constants and numeric policy tiers.

Immutable tier (P1.md §2 rule 4 — issue #8 sign-off; NEVER change in P1):
    * ``EPS_SUCC_COEFF = 0.05`` — success threshold ε_succ = 0.05·L on the raw
      correspondence distance, equivalently ``D < 0.05`` on the
      length-normalized metric returned by
      :func:`dgcc.goals.distance.correspondence_l2`.
    * ``SETTLE_VEL_THRESHOLD = 1e-3`` and ``SETTLE_MAX_STEPS = 10000`` — every
      settle-bearing simulator call in P1 collection MUST pass these values
      (global rule 7; the DLOLabEnv per-call defaults are 5000 and are NOT
      relied upon).
    * grasp realism 5% failure — env-level, kept enabled.  Rev 6 restates the
      position-error half of this tier: it was written as "±1 node", but a node
      is a DISCRETIZATION unit whose physical size is ``L/(n-1)``, so "±1 node"
      silently rescales the modelled gripper error whenever the mesh is
      refined (±32.3 mm at n=32, ±15.9 mm at n=64).  The invariant is the
      PHYSICAL error, so it is now pinned as the fixed arc length ``L/31`` —
      numerically identical at n=32 (exactly ±1 vertex, byte-identical) and
      stable under refinement.  See `dgcc.envs.dlolab.GRASP_NOISE_ARC_FRACTION`
      and the Rev 6 change-history entry.
    * ``K = 32`` external nodes, M = 8 DCT modes (24ch) — P0 values.

Adjustable tier (record every change in STEP_LOG.md, locked at M6):
    * Reward constants α=10, c_step=0.1, R_succ=5 (:class:`RewardConstants`).

WARNING (executor trap): :class:`dgcc.envs.base.RopeParams` defaults to
``n_segments=50`` (the P0 interface default) and ``rope_mass_total_kg`` to the
0.032 kg historical base.  The P1 §5 common domain pins ``n_segments`` and the
rope mass explicitly — always construct rope params through
:func:`p1_rope_params`.
"""

from __future__ import annotations

from dataclasses import dataclass

from dgcc.envs.base import RopeParams

# --- P1 §5 common rope domain (immutable for P1; OOD domains forbidden) ---
P1_LENGTH_M = 1.0
# Rev 6 (owner confirmation 2026-08-03, pilot `dossier/pilot_report_nodecount.md`):
# the main line moves from 32 to 64 raw vertices.  The pilot measured that the
# 149.56-degree hinges that drove the AT-1H exceptions are a 32-node
# DISCRETIZATION artifact (1.417% of hinges at n=32; exactly 0.000% at n=64
# under mass control) while the step cost is 1.01x.  Because the policy acts on
# a K=32 arc-length resampling, this is a physics refinement, not an action- or
# observation-space change -- the Rev 6 C1 mapping layer is what makes that
# true (`dgcc.envs.dlolab.arc_length_vertex_index`).
P1_N_SEGMENTS = 32  # STAGE 1 (byte-identity proof); stage 2 sets 64
# Rev 6 (owner-specified): total rope mass.  Previously implicit -- the adapter
# fixed the PER-SEGMENT mass at 1 g, so the rope weighed n_segments grams and a
# discretization change was silently a mass change.  Mass is now declared once,
# here, and the segment mass is derived.  0.040 kg is a PHYSICS CHANGE from the
# historical 0.032 kg and is recorded as such in the change history; it is not
# the byte-identity baseline (that is 0.032 kg at n=32).
P1_ROPE_MASS_TOTAL_KG = 0.032  # STAGE 1 (byte-identity proof); stage 2 sets 0.040
P1_BEND_STIFFNESS = 1.0
P1_TWIST_STIFFNESS = 1.0
P1_FRICTION = 1.0
P1_RADIUS = 0.005

#: The four init-shape templates (risk #5: keep all four, uniform).
INIT_SHAPES: tuple[str, ...] = ("straight", "u_bend", "s_curve", "random_smooth")

# --- Episode protocol (P1 §5) ---
EPISODE_HORIZON = 10  # max primitives per episode (research plan §6.1, adopted at M0)

# --- Immutable numeric policy (issue #8 sign-off) ---
EPS_SUCC_COEFF = 0.05  # ε_succ = 0.05·L
SETTLE_VEL_THRESHOLD = 1.0e-3
SETTLE_MAX_STEPS = 10000  # global rule 7: ALL new P1 collection settles


@dataclass(frozen=True)
class RewardConstants:
    """Adjustable-tier reward constants (P1 start values).

    Any change requires a STEP_LOG.md entry [reason/old/new] and is finally
    locked at M6 (P1.md §2 rule 4).
    """

    alpha: float = 10.0
    c_step: float = 0.1
    r_succ: float = 5.0


def p1_rope_params() -> RopeParams:
    """Return the pinned P1 §5 common rope domain.

    Explicitly sets every field — in particular ``n_segments=32`` — so the
    P0 interface default (``n_segments=50``) can never leak in.
    """

    return RopeParams(
        length_m=P1_LENGTH_M,
        n_segments=P1_N_SEGMENTS,
        bend_stiffness=P1_BEND_STIFFNESS,
        twist_stiffness=P1_TWIST_STIFFNESS,
        friction=P1_FRICTION,
        radius=P1_RADIUS,
        rope_mass_total_kg=P1_ROPE_MASS_TOTAL_KG,
    )


def eps_succ(length_m: float) -> float:
    """Return ε_succ = 0.05·L in meters (raw-distance form, immutable)."""

    return EPS_SUCC_COEFF * float(length_m)


__all__ = [
    "EPISODE_HORIZON",
    "EPS_SUCC_COEFF",
    "INIT_SHAPES",
    "P1_BEND_STIFFNESS",
    "P1_FRICTION",
    "P1_LENGTH_M",
    "P1_N_SEGMENTS",
    "P1_RADIUS",
    "P1_ROPE_MASS_TOTAL_KG",
    "P1_TWIST_STIFFNESS",
    "RewardConstants",
    "SETTLE_MAX_STEPS",
    "SETTLE_VEL_THRESHOLD",
    "eps_succ",
    "p1_rope_params",
]
