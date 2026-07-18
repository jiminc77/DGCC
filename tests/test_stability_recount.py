from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "sprint_stability_recount", Path(__file__).parents[1] / "scripts" / "sprint_stability_recount.py"
)
recount = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(recount)


def test_recount_sums_maxima_across_two_rebuild_resets(tmp_path: Path) -> None:
    log = tmp_path / "p1_sprint_train_fixture.log"
    log.write_text("\n".join([
        "p1_train start run_tag=fixture",
        "round transitions=1 nan_env=2 mag=3 rebuilds=0",
        "round transitions=2 nan_env=5 mag=7 rebuilds=0",
        "round_recovery rebuild=1 action=full_scene_rebuild",
        "round transitions=3 nan_env=1 mag=4 rebuilds=1",
        "round transitions=4 nan_env=6 mag=8 rebuilds=1",
        "round_recovery rebuild=2 action=full_scene_rebuild",
        "round transitions=5 nan_env=2 mag=1 rebuilds=2",
        "run complete transitions=6 nan_env=4 mag=5 rebuilds=2",
    ]))

    result = recount.recount_log(log)

    assert result["run_tag"] == "fixture"
    assert result["reported"] == {"nan": 4, "mag": 5}
    assert result["recounted_lower_bound"] == {"nan": 15, "mag": 20}
    assert result["rebuilds"] == 2
    assert [boundary["reason"] for boundary in result["reset_boundaries"]] == ["rebuild", "rebuild"]


def test_recount_without_reset_uses_one_segment(tmp_path: Path) -> None:
    log = tmp_path / "plain.log"
    log.write_text("round transitions=1 nan_env=2 mag=3 rebuilds=0\nrun complete nan_env=4 mag=5 rebuilds=0\n")

    result = recount.recount_log(log)

    assert result["reported"] == {"nan": 4, "mag": 5}
    assert result["recounted_lower_bound"] == {"nan": 4, "mag": 5}
    assert result["reset_boundaries"] == []


def test_counter_decrease_is_a_reset_without_rebuild_line(tmp_path: Path) -> None:
    log = tmp_path / "decrease.log"
    log.write_text("round transitions=1 nan_env=8 mag=9 rebuilds=0\nrun complete nan_env=2 mag=3 rebuilds=0\n")

    result = recount.recount_log(log)

    assert result["recounted_lower_bound"] == {"nan": 10, "mag": 12}
    assert result["reset_boundaries"][0]["reason"] == "counter_decrease"


def test_empty_log_and_json_cli(tmp_path: Path, capsys) -> None:
    log = tmp_path / "empty.log"
    log.write_text("")

    assert recount.recount_log(log) == {
        "run_tag": "empty",
        "reported": None,
        "recounted_lower_bound": {"nan": 0, "mag": 0},
        "rebuilds": 0,
        "reset_boundaries": [],
    }
    assert recount.main(["--log", str(log), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["run_tag"] == "empty"
