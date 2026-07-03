"""Decompose stored G1 effect sizes by initial template.

This is a stats-only appendix helper: it reads the raw distance lists already
stored in ``outputs/metrics/g1_effect_size.json`` and never launches a simulator.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dgcc.utils.meta import get_git_commit_hash
from gate_g1 import (
    VALID_INIT_SHAPES,
    as_jsonable,
    bootstrap_ci_key,
    bootstrap_d_ci,
    bootstrap_d_ci_cluster,
    cohens_d,
    condition_label,
    describe_distribution,
    load_config,
    measurement_config,
    measurement_lists,
    output_paths,
    pair_key,
    params_from_config,
    parse_sequences,
    summarize_sequence_counts,
)

AXIS_ORDER = ("stiffness", "friction")
IID_SEED_OFFSET = 60_000
CLUSTER_SEED_OFFSET = 70_000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recompute the G1 per-init-template appendix decomposition from stored raw distances."
    )
    parser.add_argument("--config", default="configs/gate_g1.yaml", help="G1 YAML config path")
    parser.add_argument(
        "--metrics-json",
        default=None,
        help="stored G1 metrics JSON; defaults to config.outputs.metrics_json",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/metrics/g1_template_decomposition.json",
        help="appendix decomposition JSON path",
    )
    parser.add_argument(
        "--plot-png",
        default="outputs/plots/g1_template_decomposition.png",
        help="faceted appendix plot path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="bootstrap seed; omitted means reuse stored metrics cli_seed",
    )
    parser.add_argument(
        "--stats-only-style",
        action="store_true",
        help="document that this run is recompute-only from stored distance lists",
    )
    return parser


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(as_jsonable(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def records_for_template(
    records: Sequence[dict[str, Any]],
    *,
    template: str,
    sequence_ids: Sequence[str],
    label: str,
) -> list[dict[str, Any]]:
    allowed = set(sequence_ids)
    kept = [dict(record) for record in records if str(record.get("sequence_id")) in allowed]
    unexpected = sorted({str(record.get("init_shape")) for record in kept if str(record.get("init_shape")) != template})
    if unexpected:
        raise ValueError(f"{label} for {template} contained init_shape values {unexpected}")
    return kept


def attach_cluster_metadata(ci: dict[str, Any], *, sequence_clusters: int) -> dict[str, Any]:
    enriched = dict(ci)
    enriched["sequence_clusters"] = int(sequence_clusters)
    return enriched


def compute_template_pair_metrics(
    *,
    stored_pair: dict[str, Any],
    template: str,
    sequence_ids: Sequence[str],
    pair: tuple[float, float],
    init_seed_count: int,
    bootstrap_replicates: int,
    bootstrap_level: float,
    rng_iid: np.random.Generator,
    rng_cluster: np.random.Generator,
) -> dict[str, Any]:
    key = pair_key(pair)
    between_records = records_for_template(
        stored_pair["between_condition_distances"]["values"],
        template=template,
        sequence_ids=sequence_ids,
        label=f"{key} between-condition distances",
    )
    within_records = records_for_template(
        stored_pair["within_condition_noise_floor"]["values"],
        template=template,
        sequence_ids=sequence_ids,
        label=f"{key} within-condition noise floor",
    )

    expected_between = len(sequence_ids) * init_seed_count
    expected_within = len(sequence_ids) * (init_seed_count * (init_seed_count - 1) // 2) * 2
    if len(between_records) != expected_between:
        raise ValueError(
            f"{template} {key} expected {expected_between} between distances, got {len(between_records)}"
        )
    if len(within_records) != expected_within:
        raise ValueError(f"{template} {key} expected {expected_within} within distances, got {len(within_records)}")

    between_values = [float(record["distance"]) for record in between_records]
    within_values = [float(record["distance"]) for record in within_records]
    iid_ci_key = bootstrap_ci_key(bootstrap_level, "iid")
    cluster_ci_key = bootstrap_ci_key(bootstrap_level, "cluster")

    return {
        "condition_pair": [float(pair[0]), float(pair[1])],
        "sequence_ids": list(sequence_ids),
        "sequence_cluster_count": len(sequence_ids),
        "between_condition_distances": {
            "summary": describe_distribution(between_values),
            "values": between_records,
        },
        "within_condition_noise_floor": {
            "pooled_conditions": [float(pair[0]), float(pair[1])],
            "summary": describe_distribution(within_values),
            "values": within_records,
        },
        "cohens_d": cohens_d(between_values, within_values),
        iid_ci_key: bootstrap_d_ci(
            between_values,
            within_values,
            replicates=bootstrap_replicates,
            level=bootstrap_level,
            rng=rng_iid,
        ),
        cluster_ci_key: attach_cluster_metadata(
            bootstrap_d_ci_cluster(
                between_records,
                within_records,
                sequence_ids=sequence_ids,
                replicates=bootstrap_replicates,
                level=bootstrap_level,
                rng=rng_cluster,
            ),
            sequence_clusters=len(sequence_ids),
        ),
    }


def matrix_entry(entry: dict[str, Any], *, bootstrap_level: float) -> dict[str, Any]:
    iid_ci_key = bootstrap_ci_key(bootstrap_level, "iid")
    cluster_ci_key = bootstrap_ci_key(bootstrap_level, "cluster")
    return {
        "cohens_d": entry["cohens_d"],
        iid_ci_key: entry[iid_ci_key],
        cluster_ci_key: entry[cluster_ci_key],
        "between_n": entry["between_condition_distances"]["summary"]["n"],
        "within_n": entry["within_condition_noise_floor"]["summary"]["n"],
        "sequence_cluster_count": entry["sequence_cluster_count"],
    }


def spread_summary(
    *,
    stored_pair: dict[str, Any],
    template_entries: dict[str, dict[str, Any]],
    pair: tuple[float, float],
) -> dict[str, Any]:
    pooled_d = float(stored_pair["cohens_d"])
    template_ds = {template: float(entry["cohens_d"]) for template, entry in template_entries.items()}
    deltas = {template: value - pooled_d for template, value in template_ds.items()}
    min_template = min(template_ds, key=template_ds.__getitem__)
    max_template = max(template_ds, key=template_ds.__getitem__)
    max_abs_delta_template = max(deltas, key=lambda template: abs(deltas[template]))
    return {
        "condition_pair": [float(pair[0]), float(pair[1])],
        "pooled_cohens_d": pooled_d,
        "template_cohens_d": template_ds,
        "template_minus_pooled": deltas,
        "min_template": {"template": min_template, "cohens_d": template_ds[min_template]},
        "max_template": {"template": max_template, "cohens_d": template_ds[max_template]},
        "template_range": template_ds[max_template] - template_ds[min_template],
        "max_abs_delta_from_pooled": {
            "template": max_abs_delta_template,
            "delta": deltas[max_abs_delta_template],
        },
    }


def compute_decomposition(
    *,
    config: dict[str, Any],
    config_text: str,
    config_path: Path,
    metrics_path: Path,
    stored_payload: dict[str, Any],
    source_sha256: str,
    output_json: Path,
    plot_png: Path,
    seed: int | None,
    stats_only_style: bool,
) -> dict[str, Any]:
    base_params = params_from_config(config)
    sequences = parse_sequences(config, n_vertices=base_params.n_segments)
    init_seeds, stiffness_conditions, friction_conditions, pairs = measurement_lists(config)
    sequence_ids_by_template = {
        template: [sequence.id for sequence in sequences if sequence.init_shape == template]
        for template in VALID_INIT_SHAPES
    }
    for template, sequence_ids in sequence_ids_by_template.items():
        if len(sequence_ids) != 5:
            raise ValueError(f"expected 5 sequences for {template}, got {len(sequence_ids)}")

    measurement = measurement_config(config)
    bootstrap_replicates = int(measurement.get("bootstrap_replicates", 5000))
    bootstrap_level = float(measurement.get("bootstrap_ci", 0.95))
    stats_seed = int(stored_payload.get("cli_seed", 0) if seed is None else seed)
    rng_iid = np.random.default_rng(stats_seed + IID_SEED_OFFSET)
    rng_cluster = np.random.default_rng(stats_seed + CLUSTER_SEED_OFFSET)
    iid_ci_key = bootstrap_ci_key(bootstrap_level, "iid")
    cluster_ci_key = bootstrap_ci_key(bootstrap_level, "cluster")

    stored_axes = stored_payload.get("axes", {})
    axes: dict[str, Any] = {}
    d_matrix: dict[str, dict[str, dict[str, Any]]] = {
        template: {axis: {} for axis in AXIS_ORDER} for template in VALID_INIT_SHAPES
    }
    spreads: dict[str, dict[str, Any]] = {axis: {} for axis in AXIS_ORDER}

    for axis in AXIS_ORDER:
        if axis not in stored_axes:
            raise ValueError(f"stored metrics missing axis {axis}")
        stored_axis = stored_axes[axis]
        axis_conditions = stiffness_conditions if axis == "stiffness" else friction_conditions
        axis_payload: dict[str, Any] = {
            "conditions": [float(condition) for condition in axis_conditions],
            "pairwise": {},
        }
        for pair in pairs:
            key = pair_key(pair)
            stored_pairwise = stored_axis.get("pairwise", {})
            if key not in stored_pairwise:
                raise ValueError(f"stored {axis} metrics missing pair {key}")
            stored_pair = stored_pairwise[key]
            template_entries = {
                template: compute_template_pair_metrics(
                    stored_pair=stored_pair,
                    template=template,
                    sequence_ids=sequence_ids_by_template[template],
                    pair=pair,
                    init_seed_count=len(init_seeds),
                    bootstrap_replicates=bootstrap_replicates,
                    bootstrap_level=bootstrap_level,
                    rng_iid=rng_iid,
                    rng_cluster=rng_cluster,
                )
                for template in VALID_INIT_SHAPES
            }
            axis_payload["pairwise"][key] = {
                "condition_pair": [float(pair[0]), float(pair[1])],
                "pooled_reference": {
                    "cohens_d": stored_pair["cohens_d"],
                    iid_ci_key: stored_pair.get(iid_ci_key),
                    cluster_ci_key: stored_pair.get(cluster_ci_key),
                    "between_n": stored_pair["between_condition_distances"]["summary"]["n"],
                    "within_n": stored_pair["within_condition_noise_floor"]["summary"]["n"],
                },
                "templates": template_entries,
                "spread_vs_pooled": spread_summary(
                    stored_pair=stored_pair,
                    template_entries=template_entries,
                    pair=pair,
                ),
            }
            spreads[axis][key] = axis_payload["pairwise"][key]["spread_vs_pooled"]
            for template, entry in template_entries.items():
                d_matrix[template][axis][key] = matrix_entry(entry, bootstrap_level=bootstrap_level)
        axes[axis] = axis_payload

    return {
        "schema_version": 1,
        "gate": "G1",
        "artifact": "per_init_template_decomposition",
        "created_at": utc_now(),
        "commit_hash": get_git_commit_hash(Path.cwd()),
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_text.encode("utf-8")).hexdigest(),
        "source_metrics_json": str(metrics_path),
        "source_metrics_sha256": source_sha256,
        "outputs": {
            "decomposition_json": str(output_json),
            "plot_png": str(plot_png),
        },
        "recompute_only": True,
        "stats_only_style_flag": bool(stats_only_style),
        "no_new_simulation": True,
        "purpose": "Per-init-template decomposition of stored G1 raw distance lists for human inspection.",
        "small_n_caveat": {
            "template_sequence_clusters": 5,
            "text": "Each template-specific cluster bootstrap resamples 5 sequence clusters; each cell contains 15 between-condition distances and 30 pooled within-floor distances.",
        },
        "bootstrap": {
            "level": bootstrap_level,
            "replicates": bootstrap_replicates,
            "seed": stats_seed,
            "seed_derivation": {
                "source": "--seed when provided, otherwise stored metrics cli_seed",
                "iid": f"np.random.default_rng(seed + {IID_SEED_OFFSET}) consumed in deterministic axis→pair→template order",
                "cluster": f"np.random.default_rng(seed + {CLUSTER_SEED_OFFSET}) consumed in deterministic axis→pair→template order",
            },
            "iid_ci_key": iid_ci_key,
            "cluster_ci_key": cluster_ci_key,
        },
        "measurement_design": {
            "templates": list(VALID_INIT_SHAPES),
            "sequence_ids_by_template": sequence_ids_by_template,
            "sequence_counts_by_shape": summarize_sequence_counts(sequences),
            "init_seeds": [int(seed_value) for seed_value in init_seeds],
            "stiffness_multipliers": [float(value) for value in stiffness_conditions],
            "friction_multipliers": [float(value) for value in friction_conditions],
            "condition_pairs": [[float(a), float(b)] for a, b in pairs],
            "base_rope_params": asdict(base_params),
        },
        "d_matrix": d_matrix,
        "notable_spreads_vs_pooled": spreads,
        "axes": axes,
    }


def finite_ci_bounds(ci: dict[str, Any] | None) -> tuple[float, float] | None:
    if not ci or ci.get("low") is None or ci.get("high") is None:
        return None
    low = float(ci["low"])
    high = float(ci["high"])
    if not (np.isfinite(low) and np.isfinite(high)):
        return None
    return low, high


def draw_interval(
    ax: plt.Axes,
    x: float,
    ci: dict[str, Any] | None,
    *,
    color: str,
    alpha: float,
    linewidth: float,
    cap_width: float,
    label: str | None,
) -> None:
    bounds = finite_ci_bounds(ci)
    if bounds is None:
        return
    low, high = bounds
    ax.vlines(x, low, high, color=color, alpha=alpha, linewidth=linewidth, label=label)
    ax.hlines([low, high], x - cap_width / 2.0, x + cap_width / 2.0, color=color, alpha=alpha, linewidth=linewidth)


def plot_decomposition(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    templates = payload["measurement_design"]["templates"]
    pairs = [pair_key(pair) for pair in payload["measurement_design"]["condition_pairs"]]
    iid_ci_key = payload["bootstrap"]["iid_ci_key"]
    cluster_ci_key = payload["bootstrap"]["cluster_ci_key"]

    fig, axes = plt.subplots(
        len(AXIS_ORDER),
        len(pairs),
        figsize=(4.9 * len(pairs), 4.2 * len(AXIS_ORDER)),
        constrained_layout=True,
    )
    axes_array = np.asarray(axes).reshape(len(AXIS_ORDER), len(pairs))
    x_positions = np.arange(len(templates), dtype=float)
    x_labels = [template.replace("_", "\n") for template in templates]

    for row, axis in enumerate(AXIS_ORDER):
        for col, pair in enumerate(pairs):
            ax = axes_array[row, col]
            pair_payload = payload["axes"][axis]["pairwise"][pair]
            entries = pair_payload["templates"]
            d_values = np.asarray([float(entries[template]["cohens_d"]) for template in templates], dtype=float)
            pooled_d = float(pair_payload["pooled_reference"]["cohens_d"])

            for idx, template in enumerate(templates):
                entry = entries[template]
                draw_interval(
                    ax,
                    x_positions[idx] - 0.055,
                    entry[iid_ci_key],
                    color="#7aa6c2",
                    alpha=0.55,
                    linewidth=2.0,
                    cap_width=0.10,
                    label="iid 95% CI" if row == 0 and col == 0 and idx == 0 else None,
                )
                draw_interval(
                    ax,
                    x_positions[idx] + 0.055,
                    entry[cluster_ci_key],
                    color="#1f4e79",
                    alpha=0.90,
                    linewidth=2.4,
                    cap_width=0.12,
                    label="cluster 95% CI" if row == 0 and col == 0 and idx == 0 else None,
                )
            ax.scatter(x_positions, d_values, color="#0b1f33", s=28, zorder=3, label="template d" if row == 0 and col == 0 else None)
            ax.axhline(0.0, color="0.78", linewidth=0.9)
            ax.axhline(
                pooled_d,
                color="#b35c00",
                linestyle="--",
                linewidth=1.4,
                label="pooled d" if row == 0 and col == 0 else None,
            )
            ax.set_title(f"{axis} {pair.replace('_vs_', ' vs ')}")
            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_labels, fontsize=8)
            ax.grid(axis="y", color="0.90", linewidth=0.8)
            if col == 0:
                ax.set_ylabel("Cohen's d")

    handles, labels = axes_array[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=4, bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.suptitle("G1 per-init-template effect sizes from stored distance lists", y=1.08, fontsize=13)
    fig.text(
        0.5,
        -0.015,
        "Each template cell: 5 sequence clusters, 15 between-condition distances, 30 pooled within-floor distances.",
        ha="center",
        fontsize=9,
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(args.config)
    config, config_text = load_config(config_path)
    paths = output_paths(config)
    metrics_path = Path(args.metrics_json) if args.metrics_json is not None else paths["metrics_json"]
    output_json = Path(args.output_json)
    plot_png = Path(args.plot_png)

    if metrics_path.resolve() == output_json.resolve():
        raise ValueError("output JSON must differ from the source G1 metrics JSON")

    source_sha256 = sha256_file(metrics_path)
    stored_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload = compute_decomposition(
        config=config,
        config_text=config_text,
        config_path=config_path,
        metrics_path=metrics_path,
        stored_payload=stored_payload,
        source_sha256=source_sha256,
        output_json=output_json,
        plot_png=plot_png,
        seed=args.seed,
        stats_only_style=args.stats_only_style,
    )
    write_json(output_json, payload)
    plot_decomposition(payload, plot_png)
    print(f"source_metrics_json {metrics_path}")
    print(f"source_metrics_sha256 {source_sha256}")
    print(f"wrote metrics {output_json}")
    print(f"wrote plot {plot_png}")


if __name__ == "__main__":
    main()
