"""
Tests the parquet output backend.
"""

import uuid

import pandas as pd
import pytest
import uuid7
from astropy.time import TimezoneInfo

from sotrplib.outputs.parquet import ParquetOutput
from sotrplib.sifter.core import SifterResult

from .helpers import make_candidate

UTC = TimezoneInfo(tzname="utc")


@pytest.fixture
def candidate():
    return make_candidate(uuid.uuid4(), flux_mjy=42.0, with_thumbnail=True)


@pytest.fixture
def output(tmp_path):
    return ParquetOutput(directory=tmp_path)


def test_create_sources_deduplicates_crossmatches(output, candidate):
    sources = output.create_sources([candidate, candidate.model_copy()])

    assert len(sources) == 1
    assert sources.iloc[0]["source_id"] == "TestSource"
    assert sources.iloc[0]["ra"] == pytest.approx(10.1)
    assert sources.iloc[0]["dec"] == pytest.approx(5.1)


@pytest.mark.parametrize("with_optional_fields", [True, False])
def test_create_lightcurve(output, candidate, with_optional_fields):
    if not with_optional_fields:
        candidate.err_ra = candidate.err_dec = candidate.err_flux = None
        candidate.observation_mean_time = None
    map_id = uuid7.create()

    lightcurve = output.create_lightcurve([candidate], map_id=map_id)

    assert len(lightcurve) == 1
    row = lightcurve.iloc[0]
    assert row["measurement_id"] == candidate.measurement_id
    assert row["map_id"] == map_id
    assert row["flux"] == pytest.approx(42.0)
    if with_optional_fields:
        assert row["ra_uncertainty"] == pytest.approx(0.001)
        assert row["dec_uncertainty"] == pytest.approx(0.002)
        assert row["flux_err"] == pytest.approx(1.0)
        assert row["time"] == candidate.observation_mean_time.to_datetime(timezone=UTC)
    else:
        assert (
            row[["ra_uncertainty", "dec_uncertainty", "flux_err", "time"]].isna().all()
        )


@pytest.mark.parametrize("include_noise", [False, True])
def test_create_sifter_lightcurve_filters_noise(output, candidate, include_noise):
    transient = candidate.model_copy(update={"measurement_id": uuid7.create()})
    noise = candidate.model_copy(update={"measurement_id": uuid7.create()})
    sifter_result = SifterResult([candidate], [transient], [noise])
    output.output_noise_candidates = include_noise

    lightcurve = output.create_sifter_lightcurve(sifter_result, map_id=None)

    expected_ids = [candidate.measurement_id, transient.measurement_id]
    if include_noise:
        expected_ids.append(noise.measurement_id)
    assert lightcurve["measurement_id"].tolist() == expected_ids


def test_output_writes_parquet_files(tmp_path, output, candidate):
    map_id = uuid7.create()
    output.output(
        forced_photometry_candidates=[candidate],
        sifter_result=SifterResult([], [], []),
        map_name="test_map",
        mapcat_id=map_id,
    )

    sources = pd.read_parquet(tmp_path / f"{map_id}_sources.parquet")
    lightcurve = pd.read_parquet(tmp_path / f"{map_id}_lightcurve.parquet")
    cutouts = pd.read_parquet(tmp_path / f"{map_id}_cutouts.parquet")
    assert len(sources) == len(lightcurve) == len(cutouts) == 1
    assert sources.iloc[0]["source_id"] == candidate.source_id
    assert lightcurve.iloc[0]["flux"] == pytest.approx(42.0)
    assert cutouts.iloc[0]["measurement_id"] == lightcurve.iloc[0]["measurement_id"]
    assert cutouts.iloc[0]["units"] == "mJy"
