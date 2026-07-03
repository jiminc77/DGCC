"""P1-M1 exit check: 100-update training loop on a CPU dummy env, error-free.

Runs the real §7 components end-to-end — episode runner (immutable settle
budget kwargs captured by the dummy env), replay buffer, TD3 agent with
decoupled double-Q targets, exploration schedule, NaN covenant — against a
scripted CPU dynamics model.  No GPU, no Genesis, no learning claims: this is
a plumbing proof for the M1 exit ("더미 env에서 100 step 학습 루프 무오류
실행 로그").
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, TextIO

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dgcc.goals.dual_goal import DualGoal, goal_curve
from dgcc.models.networks import Actor, Encoder, TwinCritic, parameter_count
from dgcc.rl.replay import ReplayBuffer
from dgcc.rl.td3 import TD3Agent, TD3Config
from dgcc.tasks.domain import P1_LENGTH_M, p1_rope_params
from dgcc.tasks.episode import BatchedEpisodeRunner
from dgcc.tasks.t1 import sample_t1a_goal


class Tee:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


class DummyBatchEnv:
    """CPU stand-in for DLOLabEnv's batch surface with damped random dynamics."""

    def __init__(self, n_envs: int, seed: int) -> None:
        self.n_envs = int(n_envs)
        self._rng = np.random.default_rng(seed)
        self.state: np.ndarray | None = None

    def light_reset(self, vertices: np.ndarray, *, vel_threshold: float, max_steps: int) -> dict:
        assert max_steps == 10000 and vel_threshold == 1e-3, "immutable budget violated"
        verts = np.asarray(vertices, dtype=float)
        if verts.ndim == 2:
            verts = np.broadcast_to(verts, (self.n_envs, *verts.shape)).copy()
        self.state = verts.copy()
        return {
            "settle_converged": np.ones(self.n_envs, dtype=bool),
            "settle_steps": np.zeros(self.n_envs, dtype=int),
        }

    def get_centerline_batch(self) -> np.ndarray:
        assert self.state is not None
        return self.state.copy()

    def get_centerline_raw_batch(self) -> np.ndarray:
        return self.get_centerline_batch()

    def step_primitive_batch(
        self,
        p: np.ndarray,
        delta: np.ndarray,
        lift: list[str],
        *,
        vel_threshold: float,
        max_steps: int,
        rng: np.random.Generator | None = None,
    ) -> dict:
        assert max_steps == 10000 and vel_threshold == 1e-3, "immutable budget violated"
        assert self.state is not None
        x_before = self.state.copy()
        # Local pull around the grasped node with exponential falloff + damping.
        idx = np.arange(32)
        for env in range(self.n_envs):
            weight = np.exp(-np.abs(idx - int(p[env])) / 4.0)[:, None]
            self.state[env] = self.state[env] + weight * np.asarray(delta[env])[None, :]
        self.state += self._rng.normal(0.0, 1.0e-4, size=self.state.shape)
        return {
            "X_before": x_before,
            "X_after": self.state.copy(),
            "grasp_success": np.ones(self.n_envs, dtype=bool),
            "settle_steps": np.full(self.n_envs, 42, dtype=int),
            "info": {"settle_converged": np.ones(self.n_envs, dtype=bool)},
        }


def main() -> int:
    log_path = Path("outputs/reports/p1_m1_dummy_loop.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    original_stdout = sys.stdout
    with log_path.open("w", encoding="utf-8") as log_file:
        sys.stdout = Tee(original_stdout, log_file)  # type: ignore[assignment]
        try:
            seed = 0
            n_envs = 8
            updates_goal = 100
            # Demo-scale overrides (logged): warmup/batch shrunk so 100 real
            # updates happen inside the demo; all other §7 values default.
            config = TD3Config(warmup_transitions=64, batch_size=32)
            print(f"p1_m1_dummy_loop seed={seed} n_envs={n_envs} updates_goal={updates_goal}")
            print(f"td3 config: {config.to_dict()}")
            total_params = parameter_count(Encoder(), TwinCritic(), Actor())
            print(f"backbone parameter count (encoder+twin critic+actor): {total_params:,}")

            params = p1_rope_params()
            env = DummyBatchEnv(n_envs, seed)
            runner = BatchedEpisodeRunner(env, params)
            agent = TD3Agent(config, device="cpu")
            buffer = ReplayBuffer(config.replay_capacity)
            rng = np.random.default_rng(seed)

            episode_index = 0
            runner.begin_episodes(
                seed=seed,
                episode_index=episode_index,
                goal_fn=lambda i, x, r: sample_t1a_goal(x, r),
            )
            goal_curves = np.stack([goal_curve(g, P1_LENGTH_M) for g in runner.goals])

            updates_done = 0
            transitions = 0
            start = time.perf_counter()
            while updates_done < updates_goal:
                if runner.all_done():
                    episode_index += 1
                    runner.begin_episodes(
                        seed=seed,
                        episode_index=episode_index,
                        goal_fn=lambda i, x, r: sample_t1a_goal(x, r),
                    )
                    goal_curves = np.stack([goal_curve(g, P1_LENGTH_M) for g in runner.goals])

                p, delta, lift = agent.select_actions(
                    env.get_centerline_batch(),
                    goal_curves,
                    step=transitions,
                    total_budget=updates_goal * n_envs,
                    rng=rng,
                )
                record = runner.step(p, delta, lift, rng=rng)
                if record.get("discarded"):
                    print(f"transition batch discarded (NaN covenant): {record['reason']}")
                    continue
                active = record["active"]
                if active.any():
                    buffer.add_batch(
                        X_before=record["X_before"][active],
                        X_after=record["X_after"][active],
                        goal_curve=goal_curves[active],
                        p=np.asarray(p)[active],
                        delta=np.asarray(delta)[active],
                        lift=np.asarray([1 if v == "high" else 0 for v in lift])[active],
                        reward=record["reward"][active],
                        done=record["done"][active],
                    )
                    transitions += int(active.sum())

                if buffer.size >= config.warmup_transitions:
                    stats = agent.update(buffer.sample(config.batch_size, rng))
                    updates_done += 1
                    if updates_done % 10 == 0 or updates_done == 1:
                        print(
                            f"update {updates_done:03d} transitions={transitions} "
                            f"critic_loss={stats['critic_loss']:.4f} actor_loss={stats['actor_loss']:.4f} "
                            f"critic_grad={stats['critic_grad_norm']:.3f} actor_grad={stats['actor_grad_norm']:.3f} "
                            f"q1_mean={stats['q1_mean']:.3f} target_mean={stats['target_mean']:.3f}"
                        )

            wall = time.perf_counter() - start
            print(
                f"loop complete: updates={updates_done} transitions={transitions} "
                f"wall_s={wall:.1f} nan_incidents={runner.nan_incidents} — no errors"
            )
            return 0
        finally:
            sys.stdout = original_stdout


if __name__ == "__main__":
    raise SystemExit(main())
