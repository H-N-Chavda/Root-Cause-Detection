"""Structure and integrity, on data whose defects are known by construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from causal_bench.eda import structure


def test_shape_and_index(gaussian_frame):
    shape = structure.describe_shape(gaussian_frame)
    assert shape["n_rows"] == len(gaussian_frame)
    assert shape["n_columns"] == 4
    assert shape["columns"] == ["X1", "X2", "X3", "X4"]
    assert shape["index"]["is_monotonic_increasing"] is True
    assert shape["index"]["is_unique"] is True


def test_datetime_index_sampling_interval():
    """A regular 60-second index must be reported as regular, step 60."""
    index = pd.date_range("2020-01-01", periods=100, freq="60s")
    frame = pd.DataFrame({"X1": np.arange(100.0)}, index=index)
    interval = structure.describe_shape(frame)["sampling_interval"]
    assert interval["regular"] is True
    assert interval["step"] == 60.0
    assert interval["unit"] == "seconds"


def test_integrity_finds_every_planted_defect(dirty_frame, eda_config):
    integrity = structure.check_integrity(dirty_frame, eda_config.structure)

    assert integrity["constant_columns"] == ["X3"]
    assert ["X1", "X2"] in integrity["duplicate_column_groups"]
    assert integrity["missing_per_column"]["X5"] == 1
    assert integrity["missing_total"] == 1
    assert integrity["duplicate_rows"] == 1

    binary = [c["column"] for c in integrity["discrete_columns"] if c["binary"]]
    assert "X4" in binary


def test_clean_frame_reports_no_defects(gaussian_frame, eda_config):
    integrity = structure.check_integrity(gaussian_frame, eda_config.structure)
    assert integrity["constant_columns"] == []
    assert integrity["near_constant_columns"] == []
    assert integrity["duplicate_column_groups"] == []
    assert integrity["duplicate_rows"] == 0
    assert integrity["missing_total"] == 0
    assert integrity["discrete_columns"] == []


def test_ground_truth_reconciliation_reports_absent_nodes():
    """A 4-node ground truth against 3 data columns: X3 is absent, and both
    edges touching it become unrecoverable."""
    matrix = np.zeros((4, 4), dtype=int)
    matrix[0, 1] = 1  # X1 -> X2, kept
    matrix[1, 2] = 1  # X2 -> X3, lost
    matrix[2, 3] = 1  # X3 -> X4, lost

    result = structure.reconcile_with_ground_truth(["X1", "X2", "X4"], matrix)

    assert result["alignment"] == "by_name"
    assert result["node_prefix"] == "X"
    assert result["absent_from_data"] == ["X3"]
    assert result["absent_from_ground_truth"] == []
    assert result["ground_truth_edges"] == 3
    assert result["n_unrecoverable_edges"] == 2
    assert result["usable_edges"] == 1
    assert result["ground_truth_acyclic"] is True
    assert result["ground_truth_self_loops"] == 0


def test_reconciliation_detects_a_cycle():
    cyclic = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    result = structure.reconcile_with_ground_truth(["X1", "X2", "X3"], cyclic)
    assert result["ground_truth_acyclic"] is False


def test_reconciliation_detects_self_loops_and_reciprocal_pairs():
    matrix = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]])
    result = structure.reconcile_with_ground_truth(["X1", "X2", "X3"], matrix)
    assert result["ground_truth_self_loops"] == 1
    assert result["ground_truth_reciprocal_pairs"] == 1
    # The self-loop is excluded from the edge count.
    assert result["ground_truth_edges"] == 2


def test_reconciliation_degrades_when_names_are_not_indexed():
    """Names that are not <prefix><integer> cannot be lined up by name, and the
    function must say so rather than guessing positionally."""
    matrix = np.array([[0, 1], [0, 0]])
    result = structure.reconcile_with_ground_truth(["flow", "temp"], matrix)
    assert result["alignment"] == "unavailable"
    assert result["dimension_match"] is True


def test_reconciliation_without_ground_truth(gaussian_frame):
    result = structure.reconcile_with_ground_truth(list(gaussian_frame.columns), None)
    assert result["ground_truth_available"] is False
    assert result["n_data_columns"] == 4
