"""P1 off-policy RL: replay schema v2 + §7 TD3 with double-Q decoupling."""

from dgcc.rl.replay import (
    PROVENANCE_FRESH,
    PROVENANCE_P0_REUSE,
    ReplayBuffer,
    ReplaySchemaError,
    SCHEMA_VERSION,
    goal_spec_hash,
    ingest_v1_transitions,
    read_v2_transitions,
    validate_v2_layout,
    write_v2_transitions,
)
from dgcc.rl.td3 import (
    TD3Agent,
    TD3Config,
    TrainingNaNError,
    epsilon_schedule,
    select_p_star,
    smooth_target_u,
    td_target,
)

__all__ = [
    "PROVENANCE_FRESH",
    "PROVENANCE_P0_REUSE",
    "ReplayBuffer",
    "ReplaySchemaError",
    "SCHEMA_VERSION",
    "TD3Agent",
    "TD3Config",
    "TrainingNaNError",
    "epsilon_schedule",
    "goal_spec_hash",
    "ingest_v1_transitions",
    "read_v2_transitions",
    "select_p_star",
    "smooth_target_u",
    "td_target",
    "validate_v2_layout",
    "write_v2_transitions",
]
