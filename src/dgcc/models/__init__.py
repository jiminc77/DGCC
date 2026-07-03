"""P1 baseline networks (P1.md §6) — common backbone for P2/P3."""

from dgcc.models.networks import (
    Actor,
    DELTA_SCALE,
    EMBED_DIM,
    Encoder,
    K_NODES,
    NODE_FEATURE_DIM,
    TwinCritic,
    U_DIM,
    arc_length_coordinates,
    build_node_features,
    goal_residual_flips,
    parameter_count,
)

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
