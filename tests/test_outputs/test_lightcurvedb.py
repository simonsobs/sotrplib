"""
Tests for LightcurveDBOutput, in particular that a real UUID socat
crossmatch (CrossMatch.catalog_idx) round-trips correctly -- socat's real
source IDs are UUIDs, and lightcurvedb.models.source.Source.socat_id used
to be typed `int`, which made this path crash on any real crossmatch.

Verification reads the parquet backend's files directly rather than
through higher-level lightcurvedb read APIs (e.g. get_source_lightcurve),
since those return aggregated/binned lightcurve structures rather than
raw per-measurement rows -- reading the parquet files is simpler and more
direct for asserting exactly what got written.
"""

import asyncio
import uuid

import astropy.units as u
import numpy as np
import pandas as pd
import pytest
from astropy.time import Time
from lightcurvedb.config import Settings
from lightcurvedb.models.source import Source

from sotrplib.outputs.lightcurvedb import LightcurveDBOutput
from sotrplib.sifter.core import SifterResult
from sotrplib.sources.sources import CrossMatch, MeasuredSource


@pytest.fixture
def lightcurvedb_settings(tmp_path):
    return Settings(backend_type="parquet", parquet_base_path=str(tmp_path))


def _register_socat_source(settings: Settings, socat_id: uuid.UUID) -> uuid.UUID:
    async def _create():
        async with settings.backend as backend:
            return await backend.sources.create(
                Source(socat_id=socat_id, name="Test", ra=10.0, dec=5.0, variable=False)
            )

    return asyncio.run(_create())


def make_candidate(socat_id: uuid.UUID, flux_mjy: float, with_thumbnail: bool):
    return MeasuredSource(
        ra=10.0 * u.deg,
        dec=5.0 * u.deg,
        flux=flux_mjy * u.mJy,
        err_flux=1.0 * u.mJy,
        observation_mean_time=Time("2025-09-10T00:00:00"),
        crossmatches=[
            CrossMatch(
                source_id="TestSource", catalog_idx=socat_id, catalog_name="socat"
            )
        ],
        thumbnail=np.zeros((4, 4)) if with_thumbnail else None,
        thumbnail_unit=u.mJy if with_thumbnail else None,
    )


def test_output_uploads_flux_measurement_with_uuid_crossmatch(
    tmp_path, lightcurvedb_settings
):
    socat_id = uuid.uuid4()
    lc_source_id = _register_socat_source(lightcurvedb_settings, socat_id)

    candidate = make_candidate(socat_id, flux_mjy=42.0, with_thumbnail=False)
    output = LightcurveDBOutput(settings=lightcurvedb_settings, upsert_sources=False)

    # Must not raise -- previously crashed on int(catalog_idx) since
    # catalog_idx is a real uuid.UUID, not something int() can parse.
    output.output(
        forced_photometry_candidates=[candidate],
        sifter_result=SifterResult(
            source_candidates=[], transient_candidates=[], noise_candidates=[]
        ),
        map_id="test_map",
    )

    fluxes = pd.read_parquet(tmp_path / "fluxes" / f"{lc_source_id}.parquet")
    assert len(fluxes) == 1
    assert fluxes.iloc[0]["flux"] == pytest.approx(0.042)  # mJy -> Jy


def test_output_links_cutout_to_correct_measurement(tmp_path, lightcurvedb_settings):
    """The cutout must be linked to the measurement it actually belongs to,
    not any other measurement in the same batch -- this is what broke when
    create_batch()'s (nonexistent) return value was used for the linkage
    instead of each FluxMeasurement's own measurement_id."""
    socat_id = uuid.uuid4()
    lc_source_id = _register_socat_source(lightcurvedb_settings, socat_id)

    no_thumb = make_candidate(socat_id, flux_mjy=10.0, with_thumbnail=False)
    with_thumb = make_candidate(socat_id, flux_mjy=20.0, with_thumbnail=True)

    output = LightcurveDBOutput(settings=lightcurvedb_settings, upsert_sources=False)
    output.output(
        forced_photometry_candidates=[no_thumb, with_thumb],
        sifter_result=SifterResult(
            source_candidates=[], transient_candidates=[], noise_candidates=[]
        ),
        map_id="test_map",
    )

    fluxes = pd.read_parquet(tmp_path / "fluxes" / f"{lc_source_id}.parquet")
    assert len(fluxes) == 2

    twenty_mjy_id = fluxes.index[
        fluxes["flux"].apply(lambda f: f == pytest.approx(0.020))
    ][0]
    ten_mjy_id = fluxes.index[
        fluxes["flux"].apply(lambda f: f == pytest.approx(0.010))
    ][0]

    cutouts = pd.read_parquet(tmp_path / "cutouts" / f"{lc_source_id}.parquet")
    assert len(cutouts) == 1
    assert str(cutouts.index[0]) == str(twenty_mjy_id)
    assert str(cutouts.index[0]) != str(ten_mjy_id)
