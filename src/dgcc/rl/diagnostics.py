"""P1 §8 stability instrumentation — all runs, 25k-cadence auto plots.

Logged items (P1.md §8, research plan §6.6):
    * Q-value distribution statistics
    * overestimation gap: Q(s, a) vs realized discounted return on eval episodes
    * TD-error distribution
    * gradient norms
    * per-point argmax entropy (softmax over Q_i — DLO symmetry monitor)
    * replay statistics, active-step D statistics, and TD-target clamp hits
    * NaN/magnitude incident counters (global rule 6, env level)
    * per-template success decomposition plus lift/flip-flicker diagnostics

Every 25k collected transitions the logger renders a multi-panel dashboard to
``outputs/plots/p1_diag_<run_tag>_<transitions>.png`` and persists the full
scalar history JSON alongside the run metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PLOT_CADENCE = 25_000


def argmax_entropy(q_candidates: np.ndarray) -> float:
    """Mean softmax entropy of per-point Q values (nats)."""

    q = np.asarray(q_candidates, dtype=float)
    q = q - q.max(axis=1, keepdims=True)
    probs = np.exp(q)
    probs /= probs.sum(axis=1, keepdims=True)
    entropy = -(probs * np.log(np.clip(probs, 1e-12, None))).sum(axis=1)
    return float(entropy.mean())


class DiagnosticsLogger:
    """Accumulates §8 series keyed by collected-transition count."""

    def __init__(
        self,
        run_tag: str,
        *,
        plots_dir: Path | str = "outputs/plots",
        metrics_dir: Path | str = "outputs/metrics",
        cadence: int = PLOT_CADENCE,
    ) -> None:
        self.run_tag = str(run_tag)
        self.plots_dir = Path(plots_dir)
        self.metrics_dir = Path(metrics_dir)
        self.cadence = int(cadence)
        self._next_plot_at = self.cadence
        self.update_series: list[dict[str, float]] = []
        self.entropy_series: list[dict[str, float]] = []
        self.replay_series: list[dict[str, float]] = []
        self.eval_series: list[dict[str, Any]] = []
        self.nan_incidents = 0
        self.magnitude_incidents = 0
        self.nan_series: list[dict[str, float]] = []
        self.step_d_series: list[dict[str, float | str]] = []
        self.lift_dist_series: list[dict[str, float | str]] = []
        self.flip_flicker_series: list[dict[str, float | str]] = []
        self.plots_written: list[str] = []

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def log_update(self, transitions: int, stats: dict[str, float]) -> None:
        self.update_series.append({"transitions": float(transitions), **stats})

    def log_action_info(self, transitions: int, q1_candidates: np.ndarray) -> None:
        self.entropy_series.append(
            {
                "transitions": float(transitions),
                "argmax_entropy": argmax_entropy(q1_candidates),
                "q_candidates_mean": float(np.mean(q1_candidates)),
                "q_candidates_max": float(np.max(q1_candidates)),
            }
        )

    def log_replay(self, transitions: int, *, size: int, reward_mean: float, done_frac: float) -> None:
        self.replay_series.append(
            {
                "transitions": float(transitions),
                "replay_size": float(size),
                "replay_reward_mean": float(reward_mean),
                "replay_done_frac": float(done_frac),
            }
        )

    def log_step_d(
        self,
        transitions: int,
        d_values: np.ndarray,
        *,
        phase: str = "collect",
    ) -> None:
        values = np.asarray(d_values, dtype=float).reshape(-1)
        finite = values[np.isfinite(values)]
        self.step_d_series.append(
            {
                "transitions": float(transitions),
                "phase": str(phase),
                "d_mean": float(finite.mean()) if finite.size else float("nan"),
                "d_p50": float(np.quantile(finite, 0.50)) if finite.size else float("nan"),
                "d_p95": float(np.quantile(finite, 0.95)) if finite.size else float("nan"),
                "n_active": float(values.size),
            }
        )

    def log_nan_incidents(
        self,
        transitions: int,
        total_count: int,
        magnitude_count: int | None = None,
    ) -> None:
        self.nan_incidents = int(total_count)
        if magnitude_count is not None:
            self.magnitude_incidents = int(magnitude_count)
        self.nan_series.append(
            {
                "transitions": float(transitions),
                "nan_incidents": float(total_count),
                "magnitude_incidents": float(self.magnitude_incidents),
            }
        )

    def log_lift_dist(
        self,
        transitions: int,
        *,
        templates: list[str] | np.ndarray,
        lift: list[str] | np.ndarray,
        active: np.ndarray,
        phase: str = "collect",
    ) -> None:
        templates_arr = np.asarray(templates, dtype=object)
        lift_arr = np.asarray([str(value) for value in lift], dtype=object)
        active_arr = np.asarray(active, dtype=bool)
        for template in sorted({str(value) for value in templates_arr[active_arr]}):
            mask = active_arr & (templates_arr == template)
            n = int(mask.sum())
            n_high = int(np.count_nonzero(lift_arr[mask] == "high"))
            self.lift_dist_series.append(
                {
                    "transitions": float(transitions),
                    "phase": str(phase),
                    "template": template,
                    "n": float(n),
                    "n_high": float(n_high),
                    "frac_high": float(n_high / n) if n else float("nan"),
                }
            )

    def log_flip_flicker(
        self,
        transitions: int,
        rows: list[dict[str, float | int | str]],
        *,
        phase: str = "collect",
    ) -> None:
        for row in rows:
            self.flip_flicker_series.append(
                {
                    "transitions": float(transitions),
                    "phase": str(phase),
                    "template": str(row["template"]),
                    "flip_transitions": float(row.get("flip_transitions", 0.0)),
                    "n_active": float(row.get("n_active", 0.0)),
                    "n_tracked": float(row.get("n_tracked", 0.0)),
                    "active_transition_rate": float(row.get("active_transition_rate", float("nan"))),
                    "completed_episodes": float(row.get("completed_episodes", 0.0)),
                    "episode_flicker_rate_mean": float(
                        row.get("episode_flicker_rate_mean", float("nan"))
                    ),
                }
            )

    def log_eval(self, transitions: int, eval_result: dict[str, Any]) -> None:
        """Record one deterministic eval block (see driver for the fields)."""

        self.eval_series.append({"transitions": float(transitions), **eval_result})

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def history(self) -> dict[str, Any]:
        return {
            "run_tag": self.run_tag,
            "updates": self.update_series,
            "argmax_entropy": self.entropy_series,
            "replay": self.replay_series,
            "evals": self.eval_series,
            "nan_incidents": self.nan_series,
            "step_d": self.step_d_series,
            "lift_dist": self.lift_dist_series,
            "flip_flicker": self.flip_flicker_series,
            "plots_written": self.plots_written,
        }

    def save_history(self) -> Path:
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        path = self.metrics_dir / f"p1_diag_{self.run_tag}.json"
        path.write_text(json.dumps(self.history(), indent=1) + "\n", encoding="utf-8")
        return path

    def maybe_plot(self, transitions: int, *, force: bool = False) -> Path | None:
        if not force and transitions < self._next_plot_at:
            return None
        while self._next_plot_at <= transitions:
            self._next_plot_at += self.cadence
        return self.plot(transitions)

    def plot(self, transitions: int) -> Path:
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(3, 4, figsize=(22, 13))
        fig.suptitle(f"P1 §8 diagnostics — {self.run_tag} @ {transitions:,} transitions")

        def series(rows: list[dict], key: str) -> tuple[list[float], list[float]]:
            xs = [row["transitions"] for row in rows if key in row]
            ys = [row[key] for row in rows if key in row]
            return xs, ys

        def templated_series(
            rows: list[dict], key: str
        ) -> dict[str, tuple[list[float], list[float]]]:
            grouped: dict[str, tuple[list[float], list[float]]] = {}
            for row in rows:
                if key not in row:
                    continue
                value = row[key]
                if value is None or not np.isfinite(float(value)):
                    continue
                label = str(row.get("template", "all"))
                phase = str(row.get("phase", "collect"))
                if phase != "collect":
                    label = f"{label}:{phase}"
                grouped.setdefault(label, ([], []))
                grouped[label][0].append(float(row["transitions"]))
                grouped[label][1].append(float(value))
            return grouped

        ax = axes[0, 0]
        for key in ("q1_mean", "q2_mean", "target_mean"):
            ax.plot(*series(self.update_series, key), label=key, alpha=0.8)
        ax.set_title("Q-value stats")
        ax.legend(fontsize=7)

        ax = axes[0, 1]
        for key in ("td_error_mean", "td_error_p95"):
            ax.plot(*series(self.update_series, key), label=key, alpha=0.8)
        ax.set_title("TD error")
        ax.set_yscale("log")
        ax.legend(fontsize=7)

        ax = axes[0, 2]
        for key in ("critic_grad_norm", "actor_grad_norm"):
            ax.plot(*series(self.update_series, key), label=key, alpha=0.8)
        ax.set_title("Gradient norms")
        ax.set_yscale("log")
        ax.legend(fontsize=7)

        ax = axes[0, 3]
        ax.plot(*series(self.entropy_series, "argmax_entropy"))
        ax.axhline(np.log(32), color="grey", ls="--", lw=0.8, label="uniform ln(32)")
        ax.set_title("Per-point argmax entropy (nats)")
        ax.legend(fontsize=7)

        ax = axes[1, 0]
        xs, ys = series(self.eval_series, "overestimation_gap_mean")
        ax.plot(xs, ys, marker="o")
        ax.axhline(0.0, color="grey", ls="--", lw=0.8)
        ax.set_title("Overestimation gap (Q(s0,a0) − realized return)")

        ax = axes[1, 1]
        xs, ys = series(self.eval_series, "success_rate")
        ax.plot(xs, ys, marker="o", label="eval success")
        xs, ys = series(self.eval_series, "mean_return")
        ax2 = ax.twinx()
        ax2.plot(xs, ys, marker="s", color="tab:orange", label="eval return")
        ax.set_title("Eval success / return")
        ax.set_ylim(-0.05, 1.05)

        ax = axes[1, 2]
        templates: dict[str, tuple[list[float], list[float]]] = {}
        for row in self.eval_series:
            per_template = row.get("per_template_success", {})
            for name, value in per_template.items():
                templates.setdefault(name, ([], []))
                templates[name][0].append(row["transitions"])
                templates[name][1].append(value)
        for name, (xs, ys) in sorted(templates.items()):
            ax.plot(xs, ys, marker="o", label=name)
        ax.set_title("Per-template success (risk #5)")
        ax.set_ylim(-0.05, 1.05)
        if templates:
            ax.legend(fontsize=7)

        ax = axes[1, 3]
        ax.plot(*series(self.replay_series, "replay_size"), label="replay size")
        ax.legend(loc="upper left", fontsize=7)
        ax2 = ax.twinx()
        ax2.plot(*series(self.nan_series, "nan_incidents"), color="tab:red", label="NaN")
        ax2.plot(
            *series(self.nan_series, "magnitude_incidents"),
            color="tab:purple",
            label="magnitude",
        )
        ax2.legend(loc="lower right", fontsize=7)
        ax.set_title("Replay / incidents")

        ax = axes[2, 0]
        for key in ("d_mean", "d_p50", "d_p95"):
            ax.plot(*series(self.step_d_series, key), label=key, alpha=0.8)
        ax.set_title("Active-step D")
        ax.legend(fontsize=7)

        ax = axes[2, 1]
        ax.plot(*series(self.update_series, "td_target_clamp_hit_frac"))
        ax.set_title("TD-target clamp hit fraction")
        ax.set_ylim(-0.05, 1.05)

        ax = axes[2, 2]
        lift_series = templated_series(self.lift_dist_series, "frac_high")
        for name, (xs, ys) in sorted(lift_series.items()):
            ax.plot(xs, ys, marker=".", label=name, alpha=0.8)
        ax.set_title("Lift distribution by init template")
        ax.set_ylim(-0.05, 1.05)
        if lift_series:
            ax.legend(fontsize=7)

        ax = axes[2, 3]
        flicker_series = templated_series(self.flip_flicker_series, "active_transition_rate")
        for name, (xs, ys) in sorted(flicker_series.items()):
            ax.plot(xs, ys, marker=".", label=name, alpha=0.8)
        ax.set_title("Flip flicker by init template")
        ax.set_ylim(-0.05, 1.05)
        if flicker_series:
            ax.legend(fontsize=7)

        for row in axes:
            for axis in row:
                axis.set_xlabel("transitions")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        path = self.plots_dir / f"p1_diag_{self.run_tag}_{transitions:07d}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        self.plots_written.append(str(path))
        return path


__all__ = ["DiagnosticsLogger", "PLOT_CADENCE", "argmax_entropy"]
