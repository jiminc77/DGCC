"""P1 task suite: T1/T2 goals, episode protocol, and reward (P1.md §5)."""

from dgcc.tasks.domain import (
    EPISODE_HORIZON,
    EPS_SUCC_COEFF,
    INIT_SHAPES,
    RewardConstants,
    SETTLE_MAX_STEPS,
    SETTLE_VEL_THRESHOLD,
    eps_succ,
    p1_rope_params,
)
from dgcc.tasks.episode import (
    BatchedEpisodeRunner,
    EpisodeConfig,
    build_batch_init_vertices,
    random_policy_actions,
)
from dgcc.tasks.reward import distance_to_goal, is_success, step_reward
from dgcc.tasks.t1 import T1_TASKS, sample_t1_goal
from dgcc.tasks.t2 import T2_FAMILIES, build_t2_goal, load_t2_payload, load_t2_split

__all__ = [
    "BatchedEpisodeRunner",
    "EPISODE_HORIZON",
    "EPS_SUCC_COEFF",
    "EpisodeConfig",
    "INIT_SHAPES",
    "RewardConstants",
    "SETTLE_MAX_STEPS",
    "SETTLE_VEL_THRESHOLD",
    "T1_TASKS",
    "T2_FAMILIES",
    "build_batch_init_vertices",
    "build_t2_goal",
    "distance_to_goal",
    "eps_succ",
    "is_success",
    "load_t2_payload",
    "load_t2_split",
    "p1_rope_params",
    "random_policy_actions",
    "sample_t1_goal",
    "step_reward",
]
