"""P1-M5 analysis interfaces (P2 handover surface).

Only frozen-critic latent extraction lives here.  Probes, δm-prediction
experiments, and any latent INTERPRETATION are P2 scope and forbidden in P1.
"""

from dgcc.analysis.latent_api import (
    LATENT_SPEC,
    FrozenLatentExtractor,
    lift_to_float,
)

__all__ = ["FrozenLatentExtractor", "LATENT_SPEC", "lift_to_float"]
