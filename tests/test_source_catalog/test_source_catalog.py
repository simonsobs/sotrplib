"""
Testing source catalog generation and usage.
"""

import astropy.units as u
import pytest

from sotrplib.source_catalog.core import SourceCandidateCatalog
from sotrplib.source_catalog.database import MockDatabase
from sotrplib.sources.sources import (
    SourceCandidate,
)


@pytest.fixture
def dummy_socat_db():
    return MockDatabase()


def test_socat_mock_db(dummy_socat_db):
    db = dummy_socat_db
    assert db is not None
    assert isinstance(db, MockDatabase)

    assert (
        db.get_nearby_source(ra=10.0 * u.deg, dec=10.0 * u.deg, radius=0.1 * u.deg)
        == []
    )

    db.add_source(ra=10.0 * u.deg, dec=10.0 * u.deg, name="test_source_1")
    nearby = db.get_nearby_source(ra=10.0 * u.deg, dec=10.0 * u.deg)
    assert len(nearby) == 1
    near_source = nearby[0]
    assert near_source.crossmatches[0].name == "test_source_1"
    assert near_source.ra == 10.0 * u.deg
    assert near_source.dec == 10.0 * u.deg
    assert near_source.source_id == "0"

    db.add_source(ra=12.0 * u.deg, dec=10.0 * u.deg, name="test_source_2")
    nearby = db.get_nearby_source(ra=10.0 * u.deg, dec=10.0 * u.deg, radius=1.0 * u.deg)
    assert len(nearby) == 1
    near_source = nearby[0]
    assert near_source.crossmatches[0].name == "test_source_1"
    assert near_source.ra == 10.0 * u.deg
    assert near_source.dec == 10.0 * u.deg
    assert near_source.source_id == "0"

    nearby = db.get_nearby_source(ra=12.0 * u.deg, dec=10.0 * u.deg, radius=1.0 * u.deg)
    assert len(nearby) == 1
    near_source = nearby[0]
    assert near_source.crossmatches[0].name == "test_source_2"
    assert near_source.ra == 12.0 * u.deg
    assert near_source.dec == 10.0 * u.deg
    assert near_source.source_id == "1"


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
