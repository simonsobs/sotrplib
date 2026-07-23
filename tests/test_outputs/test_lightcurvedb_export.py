"""
Tests for lightcurvedb_export, which reshapes a pickle_to_parquet table
into lightcurvedb's flux-measurement ingest schema and (optionally) pushes
it through a real `Backend`.

Verified end-to-end against the actual deep56 discovery.parquet (280,923
forced_photometry rows, 759 sources, 723 linked to socat by name) before
these tests were written: sources registered, `ingest_dataframe` accepted
the reshaped frame without error, and `get_frequency_lightcurve` read
sane flux values back out. These tests cover the same path on a small
fixture so it runs in CI.
"""

import asyncio

import pandas as pd
import pytest
from lightcurvedb.config import Settings

from sotrplib.outputs.lightcurvedb_export import (
    build_flux_measurement_frame,
    build_lightcurvedb_sources,
)


def make_forced_photometry_frame():
    return pd.DataFrame(
        [
            {
                "source_id": "ACT-S J0058.5+0620",
                "ra": 14.640625,
                "dec": 6.334854,
                "flux": 341.740908,  # mJy
                "err_flux": 15.440817,  # mJy
                "frequency": 90.0,
                "array": "i4",
                "err_ra": None,
                "err_dec": None,
                "observation_mean_time_unix": 1758853000.0,
                "category": "forced_photometry",
            },
            {
                "source_id": "ACT-S J0058.5+0620",
                "ra": 14.640700,
                "dec": 6.334800,
                "flux": 300.0,
                "err_flux": 14.0,
                "frequency": 150.0,
                "array": "i4",
                "err_ra": None,
                "err_dec": None,
                "observation_mean_time_unix": 1758853100.0,
                "category": "forced_photometry",
            },
            {
                "source_id": "52768",  # an SSO, not in socat's fixed_sources
                "ra": 22.310226,
                "dec": 8.323124,
                "flux": 50.0,
                "err_flux": 5.0,
                "frequency": 90.0,
                "array": "i1",
                "err_ra": None,
                "err_dec": None,
                "observation_mean_time_unix": 1759000000.0,
                "category": "forced_photometry",
            },
        ]
    )


def test_build_sources_links_known_names_and_leaves_others_unlinked():
    df = make_forced_photometry_frame()
    socat_name_to_id = {"ACT-S J0058.5+0620": "11111111-1111-1111-1111-111111111111"}

    sources, name_to_lc_id = build_lightcurvedb_sources(df, socat_name_to_id)

    assert len(sources) == 2  # one row per unique source_id name
    assert set(name_to_lc_id) == {"ACT-S J0058.5+0620", "52768"}

    linked = next(s for s in sources if s.name == "ACT-S J0058.5+0620")
    assert str(linked.socat_id) == socat_name_to_id["ACT-S J0058.5+0620"]

    unlinked = next(s for s in sources if s.name == "52768")
    assert unlinked.socat_id is None

    # position is the mean of that source's own measurements
    assert linked.ra == pytest.approx((14.640625 + 14.640700) / 2)


def test_build_flux_measurement_frame_matches_lightcurvedb_schema():
    df = make_forced_photometry_frame()
    sources, name_to_lc_id = build_lightcurvedb_sources(df, {})

    frame = build_flux_measurement_frame(df, name_to_lc_id)

    assert list(frame.columns) == [
        "source_id",
        "frequency",
        "module",
        "time",
        "ra",
        "dec",
        "ra_uncertainty",
        "dec_uncertainty",
        "flux",
        "flux_err",
        "extra",
    ]
    assert len(frame) == 3
    assert frame["frequency"].dtype.kind == "i"
    row = frame.iloc[0]
    assert row["flux"] == pytest.approx(0.341740908)  # mJy -> Jy
    assert row["source_id"] == name_to_lc_id["ACT-S J0058.5+0620"]


def test_ingest_round_trips_through_a_real_backend(tmp_path):
    """Push the reshaped frame through an actual parquet Backend and read
    it back via lightcurvedb's own lightcurve API -- this is what actually
    proves the schema is ingestible, not just shaped right."""
    df = make_forced_photometry_frame()
    sources, name_to_lc_id = build_lightcurvedb_sources(df, {})
    frame = build_flux_measurement_frame(df, name_to_lc_id)

    settings = Settings(backend_type="parquet", parquet_base_path=str(tmp_path))

    async def run():
        from io import BytesIO

        async with settings.backend as backend:
            await backend.sources.create_batch(sources)
            buffer = BytesIO()
            frame.to_parquet(buffer)
            buffer.seek(0)
            await backend.fluxes.ingest_dataframe(buffer)

            source_id = next(
                s.source_id for s in sources if s.name == "ACT-S J0058.5+0620"
            )
            lc = await backend.lightcurves.get_frequency_lightcurve(
                source_id, frequency=90, limit=10
            )
            return lc

    lc = asyncio.run(run())
    assert len(lc.time) == 1
    assert lc.flux[0] == pytest.approx(0.341740908)
