"""End-to-end smoke test for the command line entry point.

Runs on the reduced Tennessee slice from `conftest`, not the full dataset, so
the whole thing finishes in seconds.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from causal_bench.cli import main


def test_cli_runs_end_to_end_and_writes_a_result(
    tmp_path: Path, te_slice_csv: Path, te_ground_truth_path: Path, capsys
) -> None:
    """Exit code 0, a timestamped run directory, a report and a log inside it."""
    out_dir = tmp_path / "results"

    exit_code = main(
        [
            "--dataset",
            str(te_slice_csv),
            "--ground-truth",
            str(te_ground_truth_path),
            "--output-dir",
            str(out_dir),
            "--no-sensitivity",
        ]
    )

    assert exit_code == 0

    run_dirs = sorted(out_dir.iterdir())
    assert len(run_dirs) == 1, f"expected one run directory, got {run_dirs}"
    run_dir = run_dirs[0]
    assert run_dir.name.startswith("run-")

    report = run_dir / "results.txt"
    assert report.exists()
    text = report.read_text()
    assert "PC Algorithm - Results" in text
    assert "Evaluation against ground truth" in text
    assert "Skeleton_Precision" in text

    assert (run_dir / "run.log").exists()

    captured = capsys.readouterr().out
    assert "Run summary" in captured
    assert str(report) in captured


def test_cli_rejects_a_dataset_without_a_ground_truth(
    tmp_path: Path, te_slice_csv: Path
) -> None:
    """--dataset and --ground-truth are only meaningful together."""
    exit_code = main(
        ["--dataset", str(te_slice_csv), "--output-dir", str(tmp_path / "results")]
    )
    assert exit_code == 2


def test_cli_dry_run_writes_nothing(
    tmp_path: Path, te_slice_csv: Path, te_ground_truth_path: Path, capsys
) -> None:
    out_dir = tmp_path / "results"
    exit_code = main(
        [
            "--dataset",
            str(te_slice_csv),
            "--ground-truth",
            str(te_ground_truth_path),
            "--output-dir",
            str(out_dir),
            "--no-sensitivity",
            "--dry-run",
        ]
    )
    assert exit_code == 0
    assert not out_dir.exists()
    assert "PC Algorithm - Results" in capsys.readouterr().out


def test_cli_reports_a_missing_config(tmp_path: Path) -> None:
    exit_code = main(
        ["--config", str(tmp_path / "nope.yaml"), "--output-dir", str(tmp_path / "r")]
    )
    assert exit_code == 2


@pytest.mark.parametrize("flag", ["--help"])
def test_cli_help_exits_zero(flag: str) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([flag])
    assert excinfo.value.code == 0
