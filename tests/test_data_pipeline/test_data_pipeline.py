"""
Tests for the full pipeline for data.
"""

import pytest
import structlog
from structlog.types import FilteringBoundLogger

from sotrplib.source_catalog.database import MockForcedPhotometryDatabase


@pytest.mark.skip(reason="this needs tiger path, not a runtime test")
def test_mock_forced_photometry_database(
    db_path="/scratch/gpfs/SIMONSOBS/users/amfoster/depth1_act_maps/inputs/PS_S19_f090_2pass_optimalCatalog.fits",
    log: FilteringBoundLogger | None = None,
):
    """
    Tests loading a mock forced photometry database.
    """
    log = log or structlog.get_logger()
    catalog = MockForcedPhotometryDatabase(db_path=db_path)
    assert len(catalog.catalog_list) > 0
    log.info(
        "mock_forced_photometry_catalog.catalog_list",
        n_sources=len(catalog.catalog_list),
    )
    for source in catalog.catalog_list:
        assert source.ra is not None
        assert source.dec is not None
