"""
Testing source object generation and usage.
"""

import astropy.units as u
import pytest

from sotrplib.sources.sources import (
    BaseSource,
    RegisteredSource,
)


@pytest.fixture
def dummy_source():
    return BaseSource(ra=10.0 * u.deg, dec=10.0 * u.deg, flux=1.0 * u.Jy)


def test_base_source(dummy_source):
    assert dummy_source.ra == 10.0 * u.deg
    assert dummy_source.dec == 10.0 * u.deg
    assert dummy_source.flux == 1.0 * u.Jy


def test_registered_source(dummy_source):
    source = RegisteredSource(
        ra=dummy_source.ra,
        dec=dummy_source.dec,
        flux=dummy_source.flux,
        sourceID="test_id",
        source_type="Unknown",
    )
    assert source.sourceID == "test_id"
    assert source.source_type == "Unknown"
    assert source.ra == dummy_source.ra
    assert source.dec == dummy_source.dec
    assert source.flux == dummy_source.flux


def test_update_crossmatch(dummy_source):
    source = RegisteredSource(
        ra=dummy_source.ra,
        dec=dummy_source.dec,
        flux=dummy_source.flux,
        sourceID="test_id",
        source_type="Unknown",
    )
    assert source.sourceID == "test_id"
    assert source.source_type == "Unknown"
    assert source.ra == dummy_source.ra
    assert source.dec == dummy_source.dec
    assert source.flux == dummy_source.flux

    source.add_crossmatches(
        new_names=["crossmatch_name_1", "crossmatch_name_2"],
        new_probabilities=None,
        new_fluxes=[1.0 * u.Jy, 2.0 * u.Jy],
        new_frequencies=["93.1GHz", "103.5GHz"],
    )

    assert source.crossmatch_fluxes == [1.0 * u.Jy, 2.0 * u.Jy]
    assert source.crossmatch_frequencies == ["93.1GHz", "103.5GHz"]
    assert source.crossmatch_names == ["crossmatch_name_1", "crossmatch_name_2"]

    source.update_crossmatches(
        ["crossmatch_name_3"],
        new_probabilities=None,
        new_fluxes=[3.0 * u.Jy],
        new_frequencies=["113.5GHz"],
    )

    assert source.crossmatch_fluxes == [1.0 * u.Jy, 2.0 * u.Jy, 3.0 * u.Jy]
    assert source.crossmatch_frequencies == ["93.1GHz", "103.5GHz", "113.5GHz"]
    assert source.crossmatch_names == [
        "crossmatch_name_1",
        "crossmatch_name_2",
        "crossmatch_name_3",
    ]
