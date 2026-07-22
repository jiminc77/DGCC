import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generate_sprint_g6b_report.py"
spec = importlib.util.spec_from_file_location("g6b_report", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_mixed_grid_is_incremental_and_idempotent(tmp_path):
    metrics = tmp_path / "outputs" / "metrics"
    reports = tmp_path / "outputs" / "reports"
    tag = "sprint_t2_v1_s0"
    _json(metrics / f"p1_run_{tag}.json", {
        "transitions": 300032, "halt_reason": None, "initial_weights_sha256": "init-hash",
        "nan_incidents_env": 2, "magnitude_incidents_env": 3, "full_scene_rebuilds": 1,
        "evals": [{"wall_s": 4.0}, {"wall_s": 7.5}],
    })
    reports.mkdir(parents=True)
    (reports / f"p1_sprint_train_{tag}.log").write_text(
        f"run_tag={tag}\nnan_env=2 mag=3 rebuilds=1\nrun complete wall_h=1.25 nan_env=2 mag=3 rebuilds=1\n",
        encoding="utf-8")
    _json(metrics / "sprint_sel_t2_v1_s0.json", {"selected_ckpt": "model.pt"})
    _json(metrics / f"p1_v1_sprint_heldout_{tag}.json", {
        "summary": {"success_rate": 0.25, "mean_return": 1.5}, "ckpt_sha256": "ckpt-hash",
    })
    claim = metrics / f"p1_v1_sprint_heldout_{tag}_claim.json"
    _json(claim, {"claim": "one-shot"})

    first = module.generate(tmp_path)
    watch = metrics / "sprint_g6b_watch.json"
    report = reports / "sprint_g6b_report.md"
    first_watch, first_report = watch.read_bytes(), report.read_bytes()
    second = module.generate(tmp_path)

    assert first == second
    assert watch.read_bytes() == first_watch
    assert report.read_bytes() == first_report
    assert len(first["runs"]) == 17
    assert sum(row["status"] == "complete" for row in first["runs"]) == 1
    assert sum(row["status"] == "pending" for row in first["runs"]) == 16
    completed = next(row for row in first["runs"] if row["status"] == "complete")
    assert completed["eval_wall_max_s"] == 7.5
    assert completed["wall_h"] == 1.25
    assert completed["nan"] == 2
    assert first["heldout"][0]["claim_sha256"]
    text = report.read_text(encoding="utf-8")
    assert "|v1|0|complete|300032|—|2/3/1|1.25|`init-hash`|7.500|" in text
    assert "|matched|0|pending|—|—|—/—/—|—|`—`|—|" in text
