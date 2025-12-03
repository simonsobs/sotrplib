"""
Testing source catalog generation and usage.
"""

import astropy.units as u
import pytest

from sotrplib.source_catalog.core import RegisteredSourceCatalog
from sotrplib.sources.sources import RegisteredSource


@pytest.fixture
def dummy_socat_db():
    return RegisteredSourceCatalog(sources=[])


def test_socat_mock_db(dummy_socat_db):
    db = dummy_socat_db
    assert db is not None
    assert isinstance(db, RegisteredSourceCatalog)

    assert (
        db.crossmatch(
            ra=10.0 * u.deg, dec=10.0 * u.deg, radius=0.1 * u.deg, method="all"
        )
        == []
    )
    source1 = RegisteredSource(
        source_id="test_source_1",
        ra=10.0 * u.deg,
        dec=10.0 * u.deg,
        err_ra=0.0 * u.deg,
        err_dec=0.0 * u.deg,
        flux=10.0 * u.Jy,
    )
    db.add_sources(sources=[source1])
    nearby = db.crossmatch(
        ra=source1.ra, dec=source1.dec, radius=0.1 * u.deg, method="all"
    )
    assert len(nearby) == 1
    near_source = nearby[0]
    assert near_source.source_id == "test_source_1"
    assert near_source.ra == source1.ra
    assert near_source.dec == source1.dec
    assert near_source.catalog_idx == 0

    source2 = RegisteredSource(
        source_id="test_source_2",
        ra=12.0 * u.deg,
        dec=10.0 * u.deg,
        err_ra=0.0 * u.deg,
        err_dec=0.0 * u.deg,
        flux=10.0 * u.Jy,
    )
    db.add_sources(sources=[source2])
    nearby = db.crossmatch(
        ra=source1.ra, dec=source1.dec, radius=0.1 * u.deg, method="all"
    )
    assert len(nearby) == 1
    near_source = nearby[0]
    assert near_source.source_id == "test_source_1"
    assert near_source.ra == source1.ra
    assert near_source.dec == source1.dec
    assert near_source.catalog_idx == 0

    nearby = db.crossmatch(
        ra=source2.ra, dec=source2.dec, radius=0.1 * u.deg, method="closest"
    )
    assert len(nearby) == 1
    near_source = nearby[0]
    assert near_source.source_id == "test_source_2"
    assert near_source.ra == source2.ra
    assert near_source.dec == source2.dec
    assert near_source.catalog_idx == 1

    nearby = db.crossmatch(
        ra=source2.ra, dec=source2.dec, radius=3 * u.deg, method="all"
    )
    assert len(nearby) == 2
    assert nearby[0].source_id == "test_source_1"
    assert nearby[0].ra == source1.ra
    assert nearby[0].dec == source1.dec
    assert nearby[0].catalog_idx == 0
    assert nearby[1].source_id == "test_source_2"
    assert nearby[1].ra == source2.ra
    assert nearby[1].dec == source2.dec
    assert nearby[1].catalog_idx == 1


def test_simple_source_catalog():
    outside_source = RegisteredSource(
        source_id="outside",
        ra=10.0 * u.deg,
        dec=10.0 * u.deg,
        err_ra=0.0 * u.deg,
        err_dec=0.0 * u.deg,
        flux=10.0 * u.Jy,
    )

    inside_source = RegisteredSource(
        source_id="inside",
        ra=20.0 * u.deg,
        dec=20.0 * u.deg,
        err_ra=0.0 * u.deg,
        err_dec=0.0 * u.deg,
        flux=10.0 * u.Jy,
    )

    catalog = RegisteredSourceCatalog(sources=[outside_source, inside_source])

    match = catalog.crossmatch(
        ra=20.1 * u.deg, dec=20.1 * u.deg, radius=2.0 * u.deg, method="all"
    )

    assert len(match) == 1
    assert match[0].source_id == inside_source.source_id
