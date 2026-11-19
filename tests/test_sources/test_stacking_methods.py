"""
Unit tests for the stacking methods in sotrplib.sources.stacking.

These tests build stamps in memory and do not need the map database.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from pixell import enmap
from structlog.testing import capture_logs

from sotrplib.sources.stacking import (
    InverseVarianceStacking,
    MeanStacking,
    StackedStamp,
    Stacker,
    Stamps,
    clean_stamps,
)


@pytest.fixture
def geometry():
    return enmap.geometry(pos=[[-0.01, -0.01], [0.01, 0.01]], res=0.001)


def make_stamps(geometry, fluxes, ivars, times=None) -> Stamps:
    """Stamps with a constant value per stamp."""
    shape, wcs = geometry
    if times is None:
        times = [float(i) for i in range(len(fluxes))]
    return Stamps(
        flux_stamps=[enmap.full(shape, wcs, f) for f in fluxes],
        ivar_stamps=[enmap.full(shape, wcs, w) for w in ivars],
        times=list(times),
    )


def events(logs, name):
    return [e for e in logs if e["event"] == name]


# clean_stamps


def test_clean_stamps_keeps_good_stamps(geometry):
    stamps = make_stamps(geometry, fluxes=[1.0, 2.0, 3.0], ivars=[1.0, 1.0, 1.0])

    cleaned = clean_stamps(stamps)

    assert len(cleaned.flux_stamps) == 3
    assert len(cleaned.ivar_stamps) == 3
    assert cleaned.times == [0.0, 1.0, 2.0]


@pytest.mark.parametrize(
    "array, value",
    [
        ("flux", np.nan),
        ("flux", np.inf),
        ("ivar", np.nan),
        ("ivar", 0.0),
        ("ivar", -1.0),
    ],
)
def test_clean_stamps_drops_stamp_with_one_bad_pixel(geometry, array, value):
    stamps = make_stamps(geometry, fluxes=[1.0, 2.0, 3.0], ivars=[1.0, 1.0, 1.0])
    target = stamps.flux_stamps if array == "flux" else stamps.ivar_stamps
    target[1][3, 4] = value

    with capture_logs() as logs:
        cleaned = clean_stamps(stamps)

    assert len(cleaned.flux_stamps) == 2
    assert cleaned.times == [0.0, 2.0]
    (summary,) = events(logs, "clean_stamps.summary")
    assert summary["n_input"] == 3
    assert summary["n_kept"] == 2
    assert summary["n_bad_pixels"] == 1
    assert summary["n_bad_time"] == 0


def test_clean_stamps_drops_non_finite_time(geometry):
    stamps = make_stamps(
        geometry, fluxes=[1.0, 2.0], ivars=[1.0, 1.0], times=[np.nan, 5.0]
    )

    with capture_logs() as logs:
        cleaned = clean_stamps(stamps)

    assert cleaned.times == [5.0]
    (summary,) = events(logs, "clean_stamps.summary")
    assert summary["n_bad_time"] == 1
    assert summary["n_bad_pixels"] == 0


def test_clean_stamps_does_not_modify_input(geometry):
    stamps = make_stamps(geometry, fluxes=[1.0, 2.0], ivars=[1.0, 0.0])

    clean_stamps(stamps)

    assert len(stamps.flux_stamps) == 2
    assert len(stamps.ivar_stamps) == 2


def test_clean_stamps_warns_when_nothing_is_left(geometry):
    stamps = make_stamps(geometry, fluxes=[np.nan, np.nan], ivars=[1.0, 1.0])

    with capture_logs() as logs:
        cleaned = clean_stamps(stamps)

    assert cleaned.flux_stamps == []
    (warning,) = events(logs, "clean_stamps.no_usable_stamps")
    assert warning["log_level"] == "warning"
    assert warning["n_input"] == 2


def test_clean_stamps_empty_input_does_not_warn(geometry):
    with capture_logs() as logs:
        cleaned = clean_stamps(Stamps([], [], []))

    assert cleaned.flux_stamps == []
    assert events(logs, "clean_stamps.no_usable_stamps") == []


# Behaviour every stacking method must have

ALL_METHODS = [MeanStacking, InverseVarianceStacking]
all_methods = pytest.mark.parametrize("method_cls", ALL_METHODS)


@all_methods
def test_stack_constant_stamps(geometry, method_cls):
    n, f, w = 5, 2.0, 4.0
    stamps = make_stamps(geometry, fluxes=[f] * n, ivars=[w] * n)

    result = method_cls().stack(stamps)

    assert len(result) == 1
    (stacked,) = result
    assert isinstance(stacked, StackedStamp)
    assert stacked.flux.shape == geometry[0]
    np.testing.assert_allclose(stacked.flux, f)
    # Equal noise: stacking N stamps gives N times the ivar of one stamp.
    np.testing.assert_allclose(stacked.ivar, n * w)


@all_methods
def test_stack_keeps_wcs(geometry, method_cls):
    shape, wcs = geometry
    stamps = make_stamps(geometry, fluxes=[1.0, 1.0], ivars=[1.0, 1.0])

    (stacked,) = method_cls().stack(stamps)

    assert isinstance(stacked.flux, enmap.ndmap)
    assert isinstance(stacked.ivar, enmap.ndmap)
    assert stacked.flux.shape == shape
    assert stacked.flux.wcs.wcs.compare(wcs.wcs)
    assert stacked.ivar.wcs.wcs.compare(wcs.wcs)


@all_methods
def test_stack_skips_bad_stamp(geometry, method_cls):
    n, f, w = 5, 2.0, 4.0
    stamps = make_stamps(geometry, fluxes=[f] * n, ivars=[w] * n)
    stamps.ivar_stamps[2][3, 3] = np.nan
    stamps.flux_stamps[2][:] = 1000.0  # Would change the result if it were used.

    with capture_logs() as logs:
        (stacked,) = method_cls().stack(stamps)

    np.testing.assert_allclose(stacked.flux, f)
    np.testing.assert_allclose(stacked.ivar, (n - 1) * w)
    (summary,) = events(logs, "StackingMethod.stack.stacked")
    assert summary["n_stamps"] == n - 1
    assert summary["method"] == method_cls.__name__


@all_methods
def test_stack_all_bad_returns_empty(geometry, method_cls):
    stamps = make_stamps(geometry, fluxes=[1.0, 1.0], ivars=[0.0, 0.0])

    with capture_logs() as logs:
        result = method_cls().stack(stamps)

    assert result == []
    assert len(events(logs, "clean_stamps.no_usable_stamps")) == 1


# MeanStacking


def test_mean_stack_unequal_noise(geometry):
    fluxes = [1.0, 2.0, 6.0]
    ivars = [1.0, 4.0, 0.25]
    stamps = make_stamps(geometry, fluxes=fluxes, ivars=ivars)

    (stacked,) = MeanStacking().stack(stamps)

    # Unweighted mean: Var(mean) = sum(1/ivar) / N**2.
    n = len(fluxes)
    expected_ivar = n**2 / sum(1 / w for w in ivars)
    np.testing.assert_allclose(stacked.flux, np.mean(fluxes))
    np.testing.assert_allclose(stacked.ivar, expected_ivar)


# InverseVarianceStacking


def test_ivar_stack_unequal_noise(geometry):
    fluxes = [1.0, 2.0, 6.0]
    ivars = [1.0, 4.0, 0.25]
    stamps = make_stamps(geometry, fluxes=fluxes, ivars=ivars)

    (stacked,) = InverseVarianceStacking().stack(stamps)

    # sum(f*w) = 1 + 8 + 1.5 = 10.5, sum(w) = 5.25.
    np.testing.assert_allclose(stacked.flux, 2.0)
    np.testing.assert_allclose(stacked.ivar, 5.25)


def test_ivar_stack_weights_each_pixel_separately(geometry):
    shape, wcs = geometry
    rng = np.random.default_rng(0)
    n = 4
    flux = rng.normal(size=(n, *shape))
    ivar = rng.uniform(0.5, 5.0, size=(n, *shape))
    stamps = Stamps(
        flux_stamps=[enmap.enmap(f, wcs) for f in flux],
        ivar_stamps=[enmap.enmap(w, wcs) for w in ivar],
        times=[float(i) for i in range(n)],
    )

    (stacked,) = InverseVarianceStacking().stack(stamps)

    for y, x in [(0, 0), (3, 7), (shape[0] - 1, shape[1] - 1)]:
        f, w = flux[:, y, x], ivar[:, y, x]
        np.testing.assert_allclose(stacked.flux[y, x], np.sum(f * w) / np.sum(w))
        np.testing.assert_allclose(stacked.ivar[y, x], np.sum(w))


def test_ivar_stack_noisy_stamp_has_little_weight(geometry):
    stamps = make_stamps(geometry, fluxes=[1.0, 1.0, 100.0], ivars=[1e4, 1e4, 1e-4])

    (ivar_stacked,) = InverseVarianceStacking().stack(stamps)
    (mean_stacked,) = MeanStacking().stack(stamps)

    np.testing.assert_allclose(ivar_stacked.flux, 1.0, rtol=1e-6)
    assert np.all(np.abs(mean_stacked.flux - 1.0) > 30)


def test_ivar_stack_is_at_least_as_deep_as_mean(geometry):
    shape, wcs = geometry
    rng = np.random.default_rng(1)
    n = 6
    ivar = rng.uniform(0.1, 10.0, size=(n, *shape))
    stamps = Stamps(
        flux_stamps=[enmap.zeros(shape, wcs) for _ in range(n)],
        ivar_stamps=[enmap.enmap(w, wcs) for w in ivar],
        times=[float(i) for i in range(n)],
    )

    (ivar_stacked,) = InverseVarianceStacking().stack(stamps)
    (mean_stacked,) = MeanStacking().stack(stamps)

    # Inverse-variance weighting gives the lowest-variance linear combination,
    # so its ivar is never smaller than the unweighted mean's.
    assert np.all(ivar_stacked.ivar >= mean_stacked.ivar * (1 - 1e-12))


# Stacker.stack, with stamp() replaced so that no maps are read


class _NoMapsReader:
    def map_list(self):
        return []


def _stacker_with_fake_stamps(geometry, n_sources):
    stacker = Stacker(reader=_NoMapsReader())
    calls = []

    def fake_stamp(sources, thumb_width):
        calls.append((list(sources), thumb_width))
        stacker.stamps = [
            make_stamps(geometry, fluxes=[float(k + 1)] * 3, ivars=[1.0] * 3)
            for k in range(len(sources))
        ]
        stacker._stamped_sources = list(sources)
        stacker._stamped_thumb_width = thumb_width
        return stacker.stamps

    stacker.stamp = fake_stamp
    sources = [SimpleNamespace(source_id=f"src{k}") for k in range(n_sources)]
    return stacker, sources, calls


def test_stacker_stack_returns_one_list_per_source(geometry):
    stacker, sources, _ = _stacker_with_fake_stamps(geometry, n_sources=3)

    result = stacker.stack(sources, MeanStacking())

    assert len(result) == 3
    for k, per_source in enumerate(result):
        (stacked,) = per_source
        np.testing.assert_allclose(stacked.flux, k + 1)
        np.testing.assert_allclose(stacked.ivar, 3.0)


def test_stacker_stack_reuses_stamps(geometry):
    stacker, sources, calls = _stacker_with_fake_stamps(geometry, n_sources=2)

    with capture_logs() as logs:
        stacker.stack(sources, MeanStacking())
        stacker.stack(sources, MeanStacking())

    assert len(calls) == 1
    assert len(events(logs, "Stacker.stack.restamping")) == 1
    assert len(events(logs, "Stacker.stack.using_cached_stamps")) == 1


def test_stacker_stack_restamps_for_new_sources_or_width(geometry):
    stacker, sources, calls = _stacker_with_fake_stamps(geometry, n_sources=2)

    stacker.stack(sources, MeanStacking())
    stacker.stack(sources[:1], MeanStacking())
    stacker.stack(
        sources[:1], MeanStacking(), thumb_width=stacker._stamped_thumb_width * 2
    )

    assert len(calls) == 3


def test_stacker_stack_logs_source_id(geometry):
    stacker, sources, _ = _stacker_with_fake_stamps(geometry, n_sources=2)

    with capture_logs() as logs:
        stacker.stack(sources, MeanStacking())

    stacked_events = events(logs, "StackingMethod.stack.stacked")
    assert [e["source_id"] for e in stacked_events] == ["src0", "src1"]
