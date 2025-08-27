"""
Testing source object generation and usage.
"""

import astropy.units as u
import pytest

from sotrplib.sources.sources import (
    BaseSource,
    CrossMatch,
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
        source_id="test_id",
        source_type="Unknown",
    )
    assert source.source_id == "test_id"
    assert source.source_type == "Unknown"
    assert source.ra == dummy_source.ra
    assert source.dec == dummy_source.dec
    assert source.flux == dummy_source.flux


def test_update_crossmatch(dummy_source):
    source = RegisteredSource(
        ra=dummy_source.ra,
        dec=dummy_source.dec,
        flux=dummy_source.flux,
        source_id="test_id",
        source_type="Unknown",
    )
    assert source.source_id == "test_id"
    assert source.source_type == "Unknown"
    assert source.ra == dummy_source.ra
    assert source.dec == dummy_source.dec
    assert source.flux == dummy_source.flux

    crossmatch_1 = CrossMatch(
        ra=dummy_source.ra,
        dec=dummy_source.dec,
        name="crossmatch_name_1",
        probability=None,
        distance=0.0 * u.arcsec,
        flux=1.0 * u.Jy,
        frequency=93.1 * u.GHz,
    )
    crossmatch_2 = CrossMatch(
        ra=dummy_source.ra + 1.0 * u.arcsec,
        dec=dummy_source.dec + 1.0 * u.arcsec,
        name="crossmatch_name_2",
        probability=None,
        distance=(2**-0.5) * u.arcsec,
        flux=2.0 * u.Jy,
        frequency=103.5 * u.GHz,
    )
    source.add_crossmatch(crossmatch_1)
    source.add_crossmatch(crossmatch_2)
    assert len(source.crossmatches) == 2
    assert source.crossmatches[0].flux == 1.0 * u.Jy
    assert source.crossmatches[1].flux == 2.0 * u.Jy
    assert source.crossmatches[0].frequency == 93.1 * u.GHz
    assert source.crossmatches[1].frequency == 103.5 * u.GHz
    assert source.crossmatches[0].name == "crossmatch_name_1"
    assert source.crossmatches[1].name == "crossmatch_name_2"
