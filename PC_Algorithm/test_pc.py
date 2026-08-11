"""Tests for the PC implementation.

The original single test asserted `adjacency[0] == {1, 2}` for a X -> Y -> Z
chain, which is the *wrong* answer (X is independent of Z given Y). It therefore
passed only because of the premature-termination bug in the skeleton search, and
hid it. See insight-report/ finding F16.
"""

import math

import numpy as np
import pytest

from utils import (
    _partial_correlation,
    _partial_correlation_matrix,
    _p_value_for_partial_correlation,
    compute_metrics,
    load_dataset,
    load_ground_truth,
    pc_algorithm,
)


def _chain_data(n=500, seed=0):
    """X -> Y -> Z."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = x + rng.normal(scale=0.3, size=n)
    z = y + rng.normal(scale=0.3, size=n)
    return np.column_stack([x, y, z])


def _collider_data(n=500, seed=1):
    """X -> Z <- Y, with X and Y independent."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = rng.normal(size=n)
    z = x + y + rng.normal(scale=0.3, size=n)
    return np.column_stack([x, y, z])


# ---------------------------------------------------------------------------
# Skeleton discovery
# ---------------------------------------------------------------------------

def test_chain_removes_the_non_adjacent_pair():
    """F4: X and Z are separated by Y, so the X-Z edge must go.

    Level 0 deletes nothing here (all three are marginally correlated), which is
    exactly the situation in which the old `if not changed: break` aborted the
    search before level 1 ever ran.
    """
    result = pc_algorithm(_chain_data(), alpha=0.05, max_cond_set_size=1)
    assert result.adjacency[0] == {1}
    assert result.adjacency[2] == {1}
    assert result.sepset[(0, 2)] == frozenset({1})


def test_chain_is_not_oriented():
    """A 3-node chain has no unshielded collider, so nothing is oriented."""
    result = pc_algorithm(_chain_data(), alpha=0.05)
    assert result.directed_edges == set()
    assert result.undirected_edges == {(0, 1), (1, 2)}


def test_collider_is_oriented_correctly():
    """F6: X -> Z <- Y must produce both arrowheads into Z."""
    result = pc_algorithm(_collider_data(), alpha=0.05)
    assert result.adjacency[0] == {2}
    assert result.adjacency[1] == {2}
    assert result.directed_edges == {(0, 2), (1, 2)}
    assert result.conflicting_edges == set()
    assert not result.has_cycle


def test_independent_variables_give_an_empty_skeleton():
    rng = np.random.default_rng(2)
    data = rng.normal(size=(600, 4))
    result = pc_algorithm(data, alpha=0.01)
    assert result.skeleton_edges() == set()


def test_meek_rule_1_propagates_orientation():
    """X -> Z <- Y with Z - W (W adjacent only to Z) gives Z -> W by R1."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=800)
    y = rng.normal(size=800)
    z = x + y + rng.normal(scale=0.2, size=800)
    w = z + rng.normal(scale=0.2, size=800)
    result = pc_algorithm(np.column_stack([x, y, z, w]), alpha=0.01)
    assert (2, 3) in result.directed_edges
    assert (3, 2) not in result.directed_edges


def test_skeleton_search_is_order_independent():
    """F9: PC-stable, so permuting columns permutes the skeleton, nothing more."""
    data = _collider_data(n=800, seed=4)
    base = pc_algorithm(data, alpha=0.01).skeleton_edges()
    perm = [2, 0, 1]
    permuted = pc_algorithm(data[:, perm], alpha=0.01).skeleton_edges()
    back = {
        (min(perm[i], perm[j]), max(perm[i], perm[j])) for i, j in permuted
    }
    assert back == base


# ---------------------------------------------------------------------------
# Conditional independence test
# ---------------------------------------------------------------------------

def test_partial_correlation_implementations_agree():
    """The fast matrix path must match the recursive reference formula."""
    rng = np.random.default_rng(5)
    data = rng.normal(size=(400, 5))
    data[:, 2] += data[:, 0]
    data[:, 3] += data[:, 1] + data[:, 2]
    corr = np.corrcoef(data, rowvar=False)
    for cond in [(), (2,), (2, 3), (1, 2, 3)]:
        recursive = _partial_correlation(
            data[:, 0], data[:, 4], [data[:, c] for c in cond]
        )
        matrix = _partial_correlation_matrix(corr, 0, 4, cond)
        assert matrix == pytest.approx(recursive, abs=1e-9)


def test_chain_partial_correlation_is_near_zero():
    data = _chain_data()
    p = _p_value_for_partial_correlation(data[:, 0], data[:, 2], [data[:, 1]], len(data))
    assert p > 0.05


# ---------------------------------------------------------------------------
# Ground-truth loading
# ---------------------------------------------------------------------------

def test_matrix_ground_truth_is_not_parsed_as_an_edge_list(tmp_path):
    """F1: the exact failure that zeroed every Tennessee metric.

    A square matrix whose rows start '0 0' used to be read as an edge list,
    yielding a single diagonal entry - an empty target - reported as 0.0 scores.
    """
    path = tmp_path / "gt.txt"
    path.write_text("0 1 0\n0 0 1\n0 0 0\n")
    matrix, fmt, info = load_ground_truth(path, ["X1", "X2", "X3"])
    assert fmt == "matrix"
    assert info["kept_edges"] == 2
    assert matrix[0, 1] == 1 and matrix[1, 2] == 1


def test_empty_ground_truth_is_rejected(tmp_path):
    """F1/F8: refuse to score against a target with no edges."""
    path = tmp_path / "gt.txt"
    path.write_text("0 0 0\n0 0 0\n0 0 0\n")
    with pytest.raises(ValueError, match="no edges"):
        load_ground_truth(path, ["X1", "X2", "X3"])


def test_ground_truth_is_aligned_by_variable_name(tmp_path):
    """F3: a 4x4 truth against a dataset missing X3 keeps only X3-free edges."""
    path = tmp_path / "gt.txt"
    path.write_text(
        "0 1 0 0\n"   # X1 -> X2   (kept)
        "0 0 1 0\n"   # X2 -> X3   (dropped, X3 absent)
        "0 0 0 1\n"   # X3 -> X4   (dropped)
        "0 0 0 0\n"
    )
    matrix, fmt, info = load_ground_truth(path, ["X1", "X2", "X4"])
    assert fmt == "matrix"
    assert info["alignment"] == "by_name"
    assert info["total_edges"] == 3
    assert info["kept_edges"] == 1
    assert sorted(info["dropped_edges"]) == [("X2", "X3"), ("X3", "X4")]
    assert matrix.shape == (3, 3)
    assert matrix[0, 1] == 1


def test_tennessee_ground_truth_alignment():
    """The real Tennessee case: 33x33 truth, 31 variables, 4 edges unrecoverable."""
    names, _, dropped = load_dataset("datasetTE.csv")
    assert dropped["index_like"] == ["Unnamed: 0"]  # F2
    assert len(names) == 31
    assert "Unnamed: 0" not in names

    _, fmt, info = load_ground_truth("TEGroundTruth.txt", names)
    assert fmt == "matrix"
    assert info["alignment"] == "by_name"
    assert info["total_edges"] == 32
    assert info["kept_edges"] == 28
    assert len(info["dropped_edges"]) == 4


def test_edge_list_ground_truth_still_works(tmp_path):
    path = tmp_path / "gt.txt"
    path.write_text("X1,X2\nX2,X3\n")
    matrix, fmt, info = load_ground_truth(path, ["X1", "X2", "X3"])
    assert fmt == "edge_list"
    assert info["kept_edges"] == 2
    assert matrix[0, 1] == 1 and matrix[1, 2] == 1


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_metrics_on_a_perfect_recovery():
    data = _collider_data(n=800, seed=6)
    result = pc_algorithm(data, alpha=0.01)
    truth = np.zeros((3, 3), dtype=int)
    truth[0, 2] = truth[1, 2] = 1
    m = compute_metrics(result, truth, 3)
    assert m["Skeleton_Precision"] == 1.0
    assert m["Skeleton_Recall_TPR"] == 1.0
    assert m["Arrowhead_correct"] == 2
    assert m["Arrowhead_reversed"] == 0
    assert m["SHD"] == 0


def test_recall_is_undefined_not_zero_for_an_empty_target():
    """F8: 0/0 must not be reported as a recall of 0.0."""
    data = _collider_data(n=400, seed=7)
    result = pc_algorithm(data, alpha=0.01)
    m = compute_metrics(result, np.zeros((3, 3), dtype=int), 3)
    assert math.isnan(m["Skeleton_Recall_TPR"])
    assert m["Skeleton_TP"] == 0 and m["Skeleton_FN"] == 0


def test_shd_charges_one_for_a_reversed_edge():
    data = _collider_data(n=800, seed=8)
    result = pc_algorithm(data, alpha=0.01)
    truth = np.zeros((3, 3), dtype=int)
    truth[2, 0] = 1   # reversed relative to the discovered X -> Z
    truth[1, 2] = 1
    m = compute_metrics(result, truth, 3)
    assert m["Arrowhead_reversed"] == 1
    assert m["SHD"] == 1
