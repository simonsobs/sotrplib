"""Tests for the peak grouping in extract_sources."""

import numpy as np
import pytest

from sotrplib.sources.finding import select_isolated_peaks


def _reference(x, y, radius):
    """The original O(N^2) loop."""
    kept = []
    for j in range(len(x)):
        if kept:
            d = np.hypot(x[kept] - x[j], y[kept] - y[j])
            if (d <= radius[j]).any():
                continue
        kept.append(j)
    keep = np.zeros(len(x), dtype=bool)
    keep[kept] = True
    return keep


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_reference_loop(seed):
    rng = np.random.default_rng(seed)
    # Random peaks plus tight clusters, with mixed radii.
    n = 2000
    x = rng.uniform(0, 500, n)
    y = rng.uniform(0, 200, n)
    centres = rng.integers(0, n, 100)
    x[n - 300 :] = x[centres].repeat(3) + rng.normal(0, 2, 300)
    y[n - 300 :] = y[centres].repeat(3) + rng.normal(0, 2, 300)
    radius = rng.choice([1.0, 3.0, 6.0], n)
    assert np.array_equal(select_isolated_peaks(x, y, radius), _reference(x, y, radius))


def test_distance_equal_to_radius_is_excluded():
    # Integer grid: neighbours at exactly the radius are grouped (<=).
    x = np.array([0.0, 2.0, 4.0, 10.0])
    y = np.zeros(4)
    radius = np.full(4, 2.0)
    keep = select_isolated_peaks(x, y, radius)
    assert keep.tolist() == [True, False, True, True]
    assert np.array_equal(keep, _reference(x, y, radius))


def test_order_sets_priority():
    x = np.array([0.0, 1.0])
    y = np.zeros(2)
    assert select_isolated_peaks(x, y, np.full(2, 1.5)).tolist() == [True, False]
    assert select_isolated_peaks(x[::-1], y, np.full(2, 1.5)).tolist() == [
        True,
        False,
    ]


def test_single_peak():
    assert select_isolated_peaks(
        np.array([3.0]), np.array([4.0]), np.array([1.0])
    ).tolist() == [True]
