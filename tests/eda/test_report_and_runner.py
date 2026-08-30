"""Report rendering, the reference comparison, and the end-to-end CLI run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from causal_bench.cli import main
from causal_bench.config import EdaConfig
from causal_bench.eda import report, runner

# ---------------------------------------------------------------------------
# JSON safety
# ---------------------------------------------------------------------------


def test_json_is_valid_with_no_nan_or_numpy(tmp_path):
    """NaN and Infinity are not valid JSON, and numpy scalars are not
    serialisable at all. Both must survive the write as null / plain types."""
    findings = {
        "nan": float("nan"),
        "inf": float("inf"),
        "np_int": np.int64(7),
        "np_float": np.float64(1.5),
        "np_bool": np.bool_(True),
        "array": np.array([1.0, 2.0]),
        "path": Path("/tmp/x"),
        "nested": {"list": [float("nan"), 1]},
    }
    path = report.write_json(findings, tmp_path / "out.json")
    text = path.read_text()
    assert "NaN" not in text and "Infinity" not in text

    loaded = json.loads(text)  # strict parse: would raise on NaN
    assert loaded["nan"] is None
    assert loaded["inf"] is None
    assert loaded["np_int"] == 7
    assert loaded["np_bool"] is True
    assert loaded["array"] == [1.0, 2.0]
    assert loaded["nested"]["list"] == [None, 1]


# ---------------------------------------------------------------------------
# Reference comparison
# ---------------------------------------------------------------------------


def test_table_cells_escape_their_pipes():
    """Evidence strings contain `|skew|` and `|r|`; an unescaped pipe would
    silently split the row into extra columns."""
    text = report._table(["A", "B"], [["max |skew| = 0.2", "ok"]])
    body = text.splitlines()[2]
    assert body.count("|") - body.count("\\|") == 3  # the three cell delimiters
    assert "\\|skew\\|" in body


def test_resolve_path_handles_indexing():
    findings = {"a": {"b": [{"c": 5}, {"c": 6}]}}
    assert report.resolve_path(findings, "a.b[1].c") == 6


def test_reference_mismatch_is_reported_not_adopted():
    findings = {"x": {"y": 10.0}, "names": ["A", "B"]}
    reference = {
        "source": "test",
        "checks": [
            {"path": "x.y", "expected": 10.05, "tolerance": 0.1},
            {"path": "x.y", "expected": 99.0, "tolerance": 0.1},
            {"path": "names", "expected": ["A", "B"]},
            {"path": "does.not.exist", "expected": 1},
        ],
    }
    result = report.compare_to_reference(findings, reference)

    assert result["n_checks"] == 4
    assert result["n_match"] == 2
    assert result["n_mismatch"] == 2
    assert result["all_match"] is False
    # The computed value is preserved; the expected value is not written over it.
    assert result["checks"][1]["actual"] == 10.0
    assert result["checks"][1]["expected"] == 99.0
    assert "path not found" in result["checks"][3]["error"]


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


@pytest.fixture
def small_dataset(tmp_path) -> Path:
    """A small synthetic dataset in the on-disk shape the loader expects."""
    generator = np.random.default_rng(0)
    n = 300
    x1 = generator.normal(size=n)
    x2 = 0.8 * x1 + generator.normal(scale=0.4, size=n)
    x3 = generator.normal(size=n)
    x4 = 0.5 * x3 + generator.normal(scale=0.5, size=n)
    path = tmp_path / "small.csv"
    pd.DataFrame({"X1": x1, "X2": x2, "X3": x3, "X4": x4}).to_csv(path, index=False)
    return path


@pytest.fixture
def small_ground_truth_file(tmp_path) -> Path:
    """A 5-node ground truth against 4 data columns, so X5 is a latent node."""
    matrix = np.zeros((5, 5), dtype=int)
    matrix[0, 1] = 1  # X1 -> X2
    matrix[2, 3] = 1  # X3 -> X4
    matrix[4, 0] = 1  # X5 -> X1, unrecoverable
    path = tmp_path / "gt.txt"
    path.write_text("\n".join("\t".join(str(v) for v in row) for row in matrix) + "\n")
    return path


def test_runner_produces_both_outputs_and_figures(
    tmp_path, small_dataset, small_ground_truth_file
):
    out_dir = tmp_path / "run"
    findings = runner.run(
        dataset_path=small_dataset,
        cfg=EdaConfig(),
        out_dir=out_dir,
        ground_truth_path=small_ground_truth_file,
    )

    json_path = out_dir / runner.JSON_FILENAME
    markdown_path = out_dir / runner.MARKDOWN_FILENAME
    assert json_path.exists() and markdown_path.exists()
    assert (out_dir / runner.FIGURES_DIRNAME).is_dir()
    assert list((out_dir / runner.FIGURES_DIRNAME).glob("*.png"))

    # Every stage present in the mapping and in the file.
    for stage in (
        "structure",
        "distribution",
        "linearity",
        "temporal",
        "conditioning",
        "suitability",
        "meta",
    ):
        assert stage in findings
    assert json.loads(json_path.read_text()).keys() == findings.keys()

    text = markdown_path.read_text()
    assert text.startswith("# EDA report:")
    assert "## Findings" in text
    assert "## 7. Algorithm suitability" in text
    for algorithm in ("PC", "PCMCI+", "VAR-LiNGAM", "LSTE"):
        assert algorithm in text

    # The latent node must reach the report.
    assert findings["structure"]["ground_truth_reconciliation"]["absent_from_data"] == [
        "X5"
    ]


def test_runner_is_reproducible(tmp_path, small_dataset):
    """Same input and seed, same numbers. `meta` carries timings and a
    timestamp, so it is excluded from the comparison."""
    first = runner.run(small_dataset, EdaConfig(), tmp_path / "a")
    second = runner.run(small_dataset, EdaConfig(), tmp_path / "b")
    for stage in ("structure", "distribution", "linearity", "temporal", "conditioning"):
        assert first[stage] == second[stage], f"{stage} is not reproducible"


def test_runner_works_without_a_ground_truth(tmp_path, small_dataset):
    findings = runner.run(small_dataset, EdaConfig(), tmp_path / "run")
    reconciliation = findings["structure"]["ground_truth_reconciliation"]
    assert reconciliation["ground_truth_available"] is False
    assert (tmp_path / "run" / runner.MARKDOWN_FILENAME).exists()


def test_figures_can_be_switched_off(tmp_path, small_dataset):
    from dataclasses import replace

    cfg = EdaConfig()
    cfg = replace(cfg, figures=replace(cfg.figures, enabled=False))
    runner.run(small_dataset, cfg, tmp_path / "run")
    assert not list((tmp_path / "run" / runner.FIGURES_DIRNAME).glob("*.png"))


def test_cli_eda_end_to_end(tmp_path, small_dataset, small_ground_truth_file, capsys):
    reference = tmp_path / "reference.json"
    reference.write_text(
        json.dumps(
            {
                "source": "test",
                "checks": [
                    {"path": "structure.shape.n_rows", "expected": 300, "tolerance": 0},
                    {"path": "structure.shape.n_columns", "expected": 4, "tolerance": 0},
                ],
            }
        )
    )
    copy_to = tmp_path / "docs" / "EDA_REPORT.md"

    exit_code = main(
        [
            "eda",
            "--dataset",
            str(small_dataset),
            "--ground-truth",
            str(small_ground_truth_file),
            "--out",
            str(tmp_path / "results"),
            "--reference",
            str(reference),
            "--no-figures",
            "--copy-to",
            str(copy_to),
        ]
    )
    assert exit_code == 0

    run_dirs = sorted((tmp_path / "results").iterdir())
    assert len(run_dirs) == 1 and run_dirs[0].name.startswith("eda-")
    assert (run_dirs[0] / runner.JSON_FILENAME).exists()
    assert copy_to.exists()
    assert copy_to.read_text() == (run_dirs[0] / runner.MARKDOWN_FILENAME).read_text()

    out = capsys.readouterr().out
    assert "EDA summary" in out
    assert "2/2 checks match" in out


def test_copy_to_brings_the_figures_along(tmp_path, small_dataset, small_ground_truth_file):
    """The report links figures by a relative path, so a copy without them
    renders with broken images."""
    copy_to = tmp_path / "docs" / "EDA_REPORT.md"
    exit_code = main(
        [
            "eda",
            "--dataset",
            str(small_dataset),
            "--ground-truth",
            str(small_ground_truth_file),
            "--out",
            str(tmp_path / "results"),
            "--copy-to",
            str(copy_to),
        ]
    )
    assert exit_code == 0
    copied = tmp_path / "docs" / runner.FIGURES_DIRNAME
    assert copied.is_dir()
    linked = {
        line.split("(figures/")[1].split(")")[0]
        for line in copy_to.read_text().splitlines()
        if "(figures/" in line
    }
    assert linked
    for name in linked:
        assert (copied / name).exists(), f"{name} is linked but was not copied"


def test_cli_eda_requires_a_dataset():
    with pytest.raises(SystemExit) as excinfo:
        main(["eda"])
    assert excinfo.value.code == 2


def test_bare_invocation_still_means_run():
    """The `run` subcommand was added after the fact; a bare argument list must
    keep meaning what it meant before."""
    from causal_bench.cli import _normalise_argv

    assert _normalise_argv(["--dataset", "a.csv"])[0] == "run"
    assert _normalise_argv(["run", "--dry-run"])[0] == "run"
    assert _normalise_argv(["eda", "--dataset", "a.csv"])[0] == "eda"
    assert _normalise_argv(["--help"]) == ["--help"]
