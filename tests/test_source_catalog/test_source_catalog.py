"""
Testing source catalog generation and usage.
"""

import astropy.units as u
import pytest

from sotrplib.source_catalog.core import SourceCandidateCatalog
from sotrplib.sources.sources import (
    BaseSource,
    RegisteredSource,
    SourceCandidate,
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


def test_simple_source_catalog():
    outside_source = SourceCandidate(
        ra=10.0,
        dec=10.0,
        err_ra=0.0,
        err_dec=0.0,
        flux=10.0,
        err_flux=0.1,
        snr=10.0,
        freq="f090",
        ctime=12345.0,
        arr="pa5",
    )

    inside_source = SourceCandidate(
        ra=20.0,
        dec=20.0,
        err_ra=0.0,
        err_dec=0.0,
        flux=10.0,
        err_flux=0.1,
        snr=10.0,
        freq="f090",
        ctime=12345.0,
        arr="pa5",
    )

    catalog = SourceCandidateCatalog(sources=[outside_source, inside_source])

    match = catalog.crossmatch(
        ra=20.1 * u.deg, dec=20.1 * u.deg, radius=2.0 * u.deg, method="all"
    )

    assert len(match) == 1
    assert match[0] == inside_source
