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

import astropy.units as u
import pandas as pd
import pytest
import uuid7
from lightcurvedb.config import Settings
from lightcurvedb.models.source import Source

from sotrplib.outputs.lightcurvedb import LightcurveDBOutput, lightcurvedb_ra
from sotrplib.sifter.core import SifterResult

from .helpers import make_candidate


@pytest.fixture
def lightcurvedb_settings(tmp_path):
    return Settings(backend_type="parquet", parquet_base_path=str(tmp_path))


def _register_socat_source(settings: Settings, socat_id: uuid7.UUID) -> uuid7.UUID:
    async def _create():
        async with settings.backend as backend:
            return await backend.sources.create(
                Source(
                    source_id=socat_id,
                    socat_id=socat_id,
                    name="Test",
                    ra=10.0,
                    dec=5.0,
                    variable=False,
                )
            )

    return asyncio.run(_create())


def test_output_uploads_flux_measurement_with_uuid_crossmatch(
    tmp_path, lightcurvedb_settings
):
    socat_id = uuid7.create()
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
        map_name="test_map",
    )

    fluxes = pd.read_parquet(tmp_path / "fluxes" / f"{lc_source_id}.parquet")
    assert len(fluxes) == 1
    assert fluxes.iloc[0]["flux"] == pytest.approx(0.042)  # mJy -> Jy


def test_output_links_cutout_to_correct_measurement(tmp_path, lightcurvedb_settings):
    """The cutout must be linked to the measurement it actually belongs to,
    not any other measurement in the same batch -- this is what broke when
    create_batch()'s (nonexistent) return value was used for the linkage
    instead of each FluxMeasurement's own measurement_id."""
    socat_id = uuid7.create()
    lc_source_id = _register_socat_source(lightcurvedb_settings, socat_id)

    no_thumb = make_candidate(socat_id, flux_mjy=10.0, with_thumbnail=False)
    with_thumb = make_candidate(socat_id, flux_mjy=20.0, with_thumbnail=True)

    output = LightcurveDBOutput(settings=lightcurvedb_settings, upsert_sources=False)
    output.output(
        forced_photometry_candidates=[no_thumb, with_thumb],
        sifter_result=SifterResult(
            source_candidates=[], transient_candidates=[], noise_candidates=[]
        ),
        map_name="test_map",
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


def test_output_creates_unregistered_source_when_upsert_enabled(
    tmp_path, lightcurvedb_settings
):
    """A socat_id with no matching lightcurvedb source (get_by_socat_id
    misses) should be created on the fly when upsert_sources=True, rather
    than silently dropping the measurement."""
    socat_id = uuid7.create()
    candidate = make_candidate(socat_id, flux_mjy=42.0, with_thumbnail=False)
    output = LightcurveDBOutput(settings=lightcurvedb_settings, upsert_sources=True)

    output.output(
        forced_photometry_candidates=[candidate],
        sifter_result=SifterResult(
            source_candidates=[], transient_candidates=[], noise_candidates=[]
        ),
        map_name="test_map",
    )

    sources = pd.read_parquet(tmp_path / "sources.parquet")
    assert len(sources) == 1
    assert sources.iloc[0]["socat_id"] == str(socat_id)

    lc_source_id = sources.index[0]
    fluxes = pd.read_parquet(tmp_path / "fluxes" / f"{lc_source_id}.parquet")
    assert len(fluxes) == 1
    assert fluxes.iloc[0]["flux"] == pytest.approx(0.042)


def test_output_skips_unregistered_source_when_upsert_disabled(
    tmp_path, lightcurvedb_settings
):
    """A socat_id with no matching lightcurvedb source (get_by_socat_id
    misses) should be skipped, not raise, when upsert_sources=False."""
    socat_id = uuid7.create()
    candidate = make_candidate(socat_id, flux_mjy=42.0, with_thumbnail=False)
    output = LightcurveDBOutput(settings=lightcurvedb_settings, upsert_sources=False)

    # Must not raise, even though the source was never registered.
    output.output(
        forced_photometry_candidates=[candidate],
        sifter_result=SifterResult(
            source_candidates=[], transient_candidates=[], noise_candidates=[]
        ),
        map_name="test_map",
    )

    assert not (tmp_path / "sources.parquet").exists()
    assert not any((tmp_path / "fluxes").glob("*.parquet"))


def test_output_uses_measurement_frequency_and_array(tmp_path, lightcurvedb_settings):
    """Each flux row and cutout must carry the band and the tube of its own
    measurement. Previously all rows were written as 90 GHz, module i1."""
    socat_id = uuid7.create()
    lc_source_id = _register_socat_source(lightcurvedb_settings, socat_id)

    f150_i3 = make_candidate(socat_id, flux_mjy=10.0, with_thumbnail=True)
    f150_i3.frequency = 150.0 * u.GHz
    f150_i3.array = "i3"
    f280_c1 = make_candidate(socat_id, flux_mjy=20.0, with_thumbnail=True)
    f280_c1.frequency = 280.0 * u.GHz
    f280_c1.array = "c1"

    output = LightcurveDBOutput(settings=lightcurvedb_settings, upsert_sources=False)
    output.output(
        forced_photometry_candidates=[f150_i3, f280_c1],
        sifter_result=SifterResult(
            source_candidates=[], transient_candidates=[], noise_candidates=[]
        ),
        map_name="test_map",
    )

    fluxes = pd.read_parquet(tmp_path / "fluxes" / f"{lc_source_id}.parquet")
    by_flux = {round(row["flux"], 6): row for _, row in fluxes.iterrows()}
    assert by_flux[0.01]["frequency"] == 150
    assert by_flux[0.01]["module"] == "i3"
    assert by_flux[0.02]["frequency"] == 280
    assert by_flux[0.02]["module"] == "c1"

    cutouts = pd.read_parquet(tmp_path / "cutouts" / f"{lc_source_id}.parquet")
    assert sorted(zip(cutouts["frequency"], cutouts["module"])) == [
        (150, "i3"),
        (280, "c1"),
    ]


def test_output_skips_measurement_without_array(tmp_path, lightcurvedb_settings):
    """A measurement with no array (or no frequency) is skipped. The output
    must not write a guessed band or tube."""
    socat_id = uuid7.create()
    lc_source_id = _register_socat_source(lightcurvedb_settings, socat_id)

    good = make_candidate(socat_id, flux_mjy=10.0, with_thumbnail=False)
    no_array = make_candidate(socat_id, flux_mjy=20.0, with_thumbnail=False)
    no_array.array = None

    output = LightcurveDBOutput(settings=lightcurvedb_settings, upsert_sources=False)
    output.output(
        forced_photometry_candidates=[good, no_array],
        sifter_result=SifterResult(
            source_candidates=[], transient_candidates=[], noise_candidates=[]
        ),
        map_name="test_map",
    )

    fluxes = pd.read_parquet(tmp_path / "fluxes" / f"{lc_source_id}.parquet")
    assert len(fluxes) == 1
    assert fluxes.iloc[0]["flux"] == pytest.approx(0.010)


def test_output_wraps_ra_to_lightcurvedb_range(tmp_path, lightcurvedb_settings):
    """lightcurvedb keeps RA in [-180, 180). A source and a measurement at
    RA 351.9 deg must be written at -8.1 deg."""
    socat_id = uuid7.create()
    candidate = make_candidate(socat_id, flux_mjy=42.0, with_thumbnail=False)
    candidate.ra = 351.9 * u.deg
    output = LightcurveDBOutput(settings=lightcurvedb_settings, upsert_sources=True)

    output.output(
        forced_photometry_candidates=[candidate],
        sifter_result=SifterResult(
            source_candidates=[], transient_candidates=[], noise_candidates=[]
        ),
        map_name="test_map",
    )

    sources = pd.read_parquet(tmp_path / "sources.parquet")
    assert sources.iloc[0]["ra"] == pytest.approx(-8.1)
    fluxes = pd.read_parquet(tmp_path / "fluxes" / f"{sources.index[0]}.parquet")
    assert fluxes.iloc[0]["ra"] == pytest.approx(-8.1)


@pytest.mark.parametrize(
    "ra, expected", [(0.0, 0.0), (179.9, 179.9), (180.0, -180.0), (359.0, -1.0)]
)
def test_lightcurvedb_ra(ra, expected):
    assert lightcurvedb_ra(ra) == pytest.approx(expected)


def _instruments(settings: Settings) -> set[tuple[int, str]]:
    async def _get():
        async with settings.backend as backend:
            return await backend.instruments.get_all()

    return {(i.frequency, i.module) for i in asyncio.run(_get())}


def test_output_creates_missing_instruments(tmp_path, lightcurvedb_settings):
    """Each flux row refers to an instrument (frequency, module). The output
    creates the instruments that are not registered, one time each."""
    socat_id = uuid7.create()
    _register_socat_source(lightcurvedb_settings, socat_id)
    candidates = []
    for flux, band, array in [
        (10.0, 150.0, "i6"),
        (20.0, 220.0, "i5"),
        (30.0, 150.0, "i6"),
    ]:
        candidate = make_candidate(socat_id, flux_mjy=flux, with_thumbnail=False)
        candidate.frequency = band * u.GHz
        candidate.array = array
        candidates.append(candidate)

    output = LightcurveDBOutput(settings=lightcurvedb_settings, upsert_sources=False)
    for _ in range(2):
        output.output(
            forced_photometry_candidates=candidates,
            sifter_result=SifterResult(
                source_candidates=[], transient_candidates=[], noise_candidates=[]
            ),
            map_name="test_map",
        )

    assert _instruments(lightcurvedb_settings) == {(150, "i6"), (220, "i5")}


def test_output_does_not_create_instruments_when_disabled(
    tmp_path, lightcurvedb_settings
):
    socat_id = uuid7.create()
    _register_socat_source(lightcurvedb_settings, socat_id)
    candidate = make_candidate(socat_id, flux_mjy=10.0, with_thumbnail=False)
    output = LightcurveDBOutput(
        settings=lightcurvedb_settings, upsert_sources=False, create_instruments=False
    )
    output.output(
        forced_photometry_candidates=[candidate],
        sifter_result=SifterResult(
            source_candidates=[], transient_candidates=[], noise_candidates=[]
        ),
        map_name="test_map",
    )

    assert _instruments(lightcurvedb_settings) == set()
