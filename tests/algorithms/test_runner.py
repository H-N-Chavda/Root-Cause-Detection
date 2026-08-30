"""The orchestration layer, end to end on a synthetic dataset.

Covers the runner, the scoring targets, the report writers, the graph
serialisation and the CLI subcommand in one pass, because they only ever run
together and a unit test of each would not catch a mismatch between them.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from causal_bench.cli import main
from causal_bench.config import DiscoveryConfig
from causal_bench.discovery import runner
from causal_bench.graph.io import load_graph, save_graph
from causal_bench.graph.representation import CausalGraph

pytestmark = pytest.mark.slow


@pytest.fixture
def dataset_file(tmp_path, synthetic_data, synthetic_names) -> Path:
    path = tmp_path / "synthetic.csv"
    pd.DataFrame(synthetic_data, columns=synthetic_names).to_csv(path, index=False)
    return path


@pytest.fixture
def ground_truth_file(tmp_path, synthetic_truth) -> Path:
    """Written in the ground-truth file format: tab separated, no header."""
    path = tmp_path / "truth.txt"
    path.write_text(
        "\n".join("\t".join(str(v) for v in row) for row in synthetic_truth) + "\n"
    )
    return path


@pytest.fixture
def config() -> DiscoveryConfig:
    return DiscoveryConfig(
        seed=0,
        algorithms=("pc", "pcmci_plus", "var_lingam"),
        # Explicit tau_max: this fixture has no EDA report to take one from, and
        # the runner refuses to invent a lag order.
        tau_max=1,
        standardize=False,
        tau_sweep=False,
        prior_knowledge=None,
        algorithm_params={
            "pc": {"pc_alpha": 0.01},
            "pcmci_plus": {"pc_alpha": 0.01},
            "var_lingam": {"run_bootstrap": False},
        },
    )


# ---------------------------------------------------------------------------
# graph serialisation
# ---------------------------------------------------------------------------


def test_graph_round_trips_through_json(tmp_path):
    graph = CausalGraph.empty(["A", "B"], tau_max=1)
    graph.links[0, 1, 1] = "-->"
    graph.p_matrix = np.array([[[0.1, 0.2], [0.3, float("nan")]]] * 2).reshape(2, 2, 2)
    graph.meta = {"algorithm": "test", "value": np.int64(3)}

    path = save_graph(graph, tmp_path / "g.json")
    text = path.read_text()
    assert "NaN" not in text
    json.loads(text)  # strict parse

    back = load_graph(path)
    assert back.var_names == ["A", "B"]
    assert back.directed_edges() == [("A", "B", 1)]
    assert back.meta["value"] == 3


# ---------------------------------------------------------------------------
# the runner
# ---------------------------------------------------------------------------


def test_runner_writes_every_output(tmp_path, dataset_file, ground_truth_file, config):
    out = tmp_path / "run"
    results = runner.run(
        dataset_path=dataset_file,
        ground_truth_path=ground_truth_file,
        cfg=config,
        out_dir=out,
        eda_report_path=None,
    )

    assert (out / runner.SCORES_FILENAME).exists()
    assert (out / runner.REPORT_FILENAME).exists()
    assert (out / runner.RESULTS_FILENAME).exists()
    assert list((out / runner.GRAPHS_DIRNAME).glob("*.json"))
    assert list((out / runner.FIGURES_DIRNAME).glob("*.png"))

    assert len(results["runs"]) == 3
    for record in results["runs"]:
        assert record["completed"], f"{record['label']}: {record['error']}"
        assert record["scores"] is not None
        assert "gt_projected" in record["scores"]

    # The JSON is strictly parseable, which NaN-bearing numpy output would not be.
    json.loads((out / runner.RESULTS_FILENAME).read_text())


def test_scores_csv_is_long_format(tmp_path, dataset_file, ground_truth_file, config):
    out = tmp_path / "run"
    runner.run(dataset_file, ground_truth_file, config, out, eda_report_path=None)

    with (out / runner.SCORES_FILENAME).open() as handle:
        rows = list(csv.DictReader(handle))

    # One row per (run, target): 3 runs x 4 targets.
    assert len(rows) == 12
    assert {r["target"] for r in rows} == {
        "gt_projected",
        "gt_projected_cpdag",
        "gt_induced",
        "gt_induced_cpdag",
    }
    for row in rows:
        assert row["shd"] != ""
        assert float(row["shd"]) >= 0


def test_report_names_the_primary_target_and_the_cpdag_floor(
    tmp_path, dataset_file, ground_truth_file, config
):
    out = tmp_path / "run"
    runner.run(dataset_file, ground_truth_file, config, out, eda_report_path=None)

    text = (out / runner.REPORT_FILENAME).read_text()
    assert "gt_projected" in text
    assert "CPDAG floor" in text
    assert "Scoring targets" in text
    assert "Assumptions and failures" in text
    # The undirected-weight choice has to be visible, not buried.
    assert "0.5" in text


def test_runner_refuses_to_invent_a_lag_order(tmp_path, dataset_file, ground_truth_file):
    """`tau_max` comes from the EDA. With neither a config value nor a report,
    the run must stop rather than pick a number."""
    config = DiscoveryConfig(tau_max=None, algorithms=("pcmci_plus",))

    with pytest.raises(ValueError, match="will not choose a lag order"):
        runner.run(
            dataset_file,
            ground_truth_file,
            config,
            tmp_path / "run",
            eda_report_path=Path("/nonexistent/eda_report.json"),
        )


def test_tau_values_sweep_the_eda_range():
    """The criteria disagree, so the range is swept rather than one picked."""
    findings = {
        "temporal": {
            "lag_order": {
                "selected": {"aic": 3, "bic": 1, "hqic": 2, "fpe": 3},
                "min_order": 1,
                "max_order": 3,
            }
        }
    }
    values, reason = runner.tau_values_from_eda(
        findings, DiscoveryConfig(tau_max=None, tau_sweep=True)
    )
    assert values == [1, 2, 3]
    assert "criteria disagree" in reason

    single, reason = runner.tau_values_from_eda(
        findings, DiscoveryConfig(tau_max=None, tau_sweep=False)
    )
    assert single == [3]


def test_config_tau_max_overrides_the_eda():
    values, reason = runner.tau_values_from_eda({}, DiscoveryConfig(tau_max=2))
    assert values == [2]
    assert "config" in reason


def test_standardize_decision_comes_from_the_eda():
    findings = {
        "suitability": {
            "preprocessing": {
                "standardization": {"required": True, "reason": "spread is 2525x"}
            }
        }
    }
    value, reason = runner.standardize_from_eda(findings, DiscoveryConfig())
    assert value is True
    assert "2525x" in reason


def test_a_failing_algorithm_does_not_stop_the_run(
    tmp_path, dataset_file, ground_truth_file
):
    """pc_manual raises on prior knowledge. The run must record that and carry
    on: a missing result is a finding, not a crash."""
    config = DiscoveryConfig(
        seed=0,
        algorithms=("pc_manual", "pc"),
        tau_max=1,
        standardize=False,
        tau_sweep=False,
        prior_knowledge=None,
        algorithm_params={"pc_manual": {"max_cond_set_size": 1}},
    )
    results = runner.run(
        dataset_file, ground_truth_file, config, tmp_path / "run", eda_report_path=None
    )
    assert len(results["runs"]) == 2
    assert all(r["completed"] for r in results["runs"])


# ---------------------------------------------------------------------------
# the CLI
# ---------------------------------------------------------------------------


def test_cli_run_end_to_end(tmp_path, dataset_file, ground_truth_file, capsys):
    exit_code = main(
        [
            "run",
            "--algorithms",
            "pc,var_lingam",
            "--dataset",
            str(dataset_file),
            "--ground-truth",
            str(ground_truth_file),
            "--out",
            str(tmp_path / "results"),
            "--no-prior-knowledge",
            "--no-figures",
        ]
    )
    assert exit_code == 0

    run_dirs = sorted((tmp_path / "results").iterdir())
    assert len(run_dirs) == 1 and run_dirs[0].name.startswith("run-")
    assert (run_dirs[0] / runner.SCORES_FILENAME).exists()

    out = capsys.readouterr().out
    assert "Run summary" in out
    assert "gt_projected" in out


def test_cli_rejects_an_unknown_algorithm(tmp_path, dataset_file, ground_truth_file):
    exit_code = main(
        [
            "run",
            "--algorithms",
            "not_an_algorithm",
            "--dataset",
            str(dataset_file),
            "--ground-truth",
            str(ground_truth_file),
            "--out",
            str(tmp_path / "results"),
        ]
    )
    assert exit_code == 2
