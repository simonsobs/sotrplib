"""
Tests for the full pipeline for data.
"""

from datetime import datetime, timezone

import pytest
import structlog
from astropy import units as u
from structlog.types import FilteringBoundLogger

from sotrplib.handlers.basic import PipelineRunner
from sotrplib.maps.core import RhoAndKappaMap
from sotrplib.maps.preprocessor import KappaRhoCleaner
from sotrplib.outputs.core import PickleSerializer
from sotrplib.sifter.core import DefaultSifter
from sotrplib.source_catalog.database import (
    MockACTDatabase,
    MockForcedPhotometryDatabase,
)
from sotrplib.sources.blind import BlindSearchParameters, SigmaClipBlindSearch
from sotrplib.sources.force import SimpleForcedPhotometry
from sotrplib.sources.subtractor import PhotutilsSourceSubtractor


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


@pytest.mark.skip(reason="this runs a full pipeline on real data, not a runtime test")
def test_basic_map_pipeline(
    tmp_path,
    rho_map_path="/scratch/gpfs/SIMONSOBS/so/maps/actpol/depth1/15386/depth1_1538613353_pa5_f090_rho.fits",
    kappa_map_path="/scratch/gpfs/SIMONSOBS/so/maps/actpol/depth1/15386/depth1_1538613353_pa5_f090_kappa.fits",
    time_map_path="/scratch/gpfs/SIMONSOBS/so/maps/actpol/depth1/15386/depth1_1538613353_pa5_f090_time.fits",
):
    """
    Tests a complete setup of the basic pipeline run.
    """
    start_time_unix = rho_map_path.split("_")[-4]
    end_time_unix = rho_map_path.split("_")[-4]
    new_map = RhoAndKappaMap(
        rho_filename=rho_map_path,
        kappa_filename=kappa_map_path,
        time_filename=time_map_path,
        start_time=datetime.fromtimestamp(int(start_time_unix), tz=timezone.utc),
        end_time=datetime.fromtimestamp(int(end_time_unix), tz=timezone.utc),
        flux_units=u.mJy,
        frequency="f090",
        array="pa5",
    )
    forced_photometry_sources = MockForcedPhotometryDatabase(
        db_path="/scratch/gpfs/SIMONSOBS/users/amfoster/depth1_act_maps/inputs/PS_S19_f090_2pass_optimalCatalog.fits",
        flux_lower_limit=1 * u.Jy,
    ).catalog_list
    catalog_sources = MockACTDatabase(
        db_path="/scratch/gpfs/SIMONSOBS/users/amfoster/depth1_act_maps/inputs/PS_S19_f090_2pass_optimalCatalog.fits"
    ).catalog_list

    blind_search_params = BlindSearchParameters()
    blind_search_params.sigma_threshold = 20.0

    runner = PipelineRunner(
        maps=[new_map],
        preprocessors=[KappaRhoCleaner()],
        postprocessors=None,
        source_simulators=None,
        forced_photometry=SimpleForcedPhotometry(
            mode="nn", sources=forced_photometry_sources
        ),  # Scipy2DGaussianFitter(sources=forced_photometry_sources),
        source_subtractor=PhotutilsSourceSubtractor(),
        blind_search=SigmaClipBlindSearch(parameters=blind_search_params),
        sifter=DefaultSifter(catalog_sources=catalog_sources),
        outputs=[PickleSerializer(directory=tmp_path)],
    )

    runner.run()
