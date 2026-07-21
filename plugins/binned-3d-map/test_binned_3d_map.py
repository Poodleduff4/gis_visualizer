"""Plain-assert tests for `bin_points` — no pytest dependency, run directly
with `python3 test_binned_3d_map.py` (matches this repo's plugins, which
don't carry a test framework dependency).
"""

import numpy as np

from main import bin_points


def test_bin_points_counts_total_matches_input():
    xs = np.random.uniform(0, 10, 200)
    ys = np.random.uniform(0, 10, 200)
    counts, xedges, yedges = bin_points(xs, ys, bins=5)
    assert counts.sum() == 200
    assert len(xedges) == 6
    assert len(yedges) == 6


def test_bin_points_places_single_cluster_in_one_cell():
    xs = np.full(10, 1.0)
    ys = np.full(10, 1.0)
    counts, _, _ = bin_points(xs, ys, bins=4)
    assert counts.max() == 10
    assert counts.sum() == 10


def test_bin_points_rejects_empty_input():
    try:
        bin_points(np.array([]), np.array([]), bins=4)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for empty input")


if __name__ == "__main__":
    test_bin_points_counts_total_matches_input()
    test_bin_points_places_single_cluster_in_one_cell()
    test_bin_points_rejects_empty_input()
    print("all tests passed")
