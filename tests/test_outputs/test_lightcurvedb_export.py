"""
Tests for lightcurvedb_export, which reshapes a pickle_to_parquet table
into lightcurvedb's flux-measurement ingest schema and (optionally) pushes
it through a real `Backend`.

Verified end-to-end against the actual deep56 discovery.parquet (283,586
forced_photometry + transient_candidates rows, 2471 sources, 1558 linked
to socat) before these tests were written: sources registered,
`ingest_dataframe` accepted the reshaped frame without error, and
`get_frequency_lightcurve` read sane flux values back out, including for
an uncrossmatched transient with two same-name repeat detections across
bands. These tests cover the same paths on a small fixture so they run
in CI.
"""

import asyncio
import json

import pandas as pd
import pytest
from lightcurvedb.config import Settings

from sotrplib.outputs.lightcurvedb_export import (
    build_flux_measurement_frame,
    build_lightcurvedb_sources,
    resolve_source_identity,
)


def _row(
    source_id=None,
    ra=0.0,
    dec=0.0,
    flux=100.0,
    err_flux=10.0,
    frequency=90.0,
    array="i4",
    time_unix=1758853000.0,
    category="forced_photometry",
    crossmatch_source_id=None,
    crossmatch_catalog_idx=None,
):
    crossmatches = None
    n_crossmatches = 0
    if crossmatch_source_id is not None:
        crossmatches = json.dumps(
            [{"source_id": crossmatch_source_id, "catalog_idx": crossmatch_catalog_idx}]
        )
        n_crossmatches = 1
    return {
        "source_id": source_id,
        "ra": ra,
        "dec": dec,
        "flux": flux,
        "err_flux": err_flux,
        "frequency": frequency,
        "array": array,
        "err_ra": None,
        "err_dec": None,
        "observation_mean_time_unix": time_unix,
        "category": category,
        "n_crossmatches": n_crossmatches,
        "crossmatches": crossmatches,
    }


def make_forced_photometry_frame():
    return pd.DataFrame(
        [
            _row(
                source_id="ACT-S J0058.5+0620",
                ra=14.640625,
                dec=6.334854,
                flux=341.740908,
                err_flux=15.440817,
                frequency=90.0,
            ),
            _row(
                source_id="ACT-S J0058.5+0620",
                ra=14.640700,
                dec=6.334800,
                flux=300.0,
                err_flux=14.0,
                frequency=150.0,
                time_unix=1758853100.0,
            ),
            _row(
                source_id="52768",  # an SSO, not in socat's fixed_sources
                ra=22.310226,
                dec=8.323124,
                flux=50.0,
                err_flux=5.0,
                frequency=90.0,
                array="i1",
                time_unix=1759000000.0,
            ),
        ]
    )


def test_build_sources_links_known_names_and_leaves_others_unlinked():
    df = resolve_source_identity(make_forced_photometry_frame())
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
    df = resolve_source_identity(make_forced_photometry_frame())
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
    df = resolve_source_identity(make_forced_photometry_frame())
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


def test_transient_candidate_crossmatched_to_known_source_shares_its_source():
    """A blind detection that the sifter crossmatched to a known,
    already-forced-photometered source must collapse onto that same
    Source, not create a second one for the same physical object."""
    known_socat_id = "22222222-2222-2222-2222-222222222222"
    df = pd.DataFrame(
        [
            _row(source_id="ACT-S J0058.5+0620", ra=14.6406, dec=6.3349, flux=340.0),
            _row(
                source_id=None,
                ra=14.6410,
                dec=6.3350,
                flux=900.0,  # a flare caught via blind search, not forced phot
                category="transient_candidates",
                crossmatch_source_id="ACT-S J0058.5+0620",
                crossmatch_catalog_idx=known_socat_id,
            ),
        ]
    )
    df = resolve_source_identity(df)

    sources, name_to_lc_id = build_lightcurvedb_sources(df, {})

    assert len(sources) == 1
    assert sources[0].name == "ACT-S J0058.5+0620"
    assert str(sources[0].socat_id) == known_socat_id

    frame = build_flux_measurement_frame(df, name_to_lc_id)
    assert frame["source_id"].nunique() == 1
    assert sorted(frame["flux"]) == pytest.approx([0.34, 0.9])


def test_uncrossmatched_transient_gets_its_own_source():
    """A genuinely new transient (no crossmatch) keys off the pipeline's
    own auto-generated name and is never dropped."""
    df = pd.DataFrame(
        [
            _row(
                source_id=None,
                ra=38.6857,
                dec=7.9660,
                flux=200.0,
                category="transient_candidates",
            ),
        ]
    )
    df.loc[0, "source_id"] = "SO-T J023444+0757.9"  # pipeline-assigned name
    df = resolve_source_identity(df)

    sources, name_to_lc_id = build_lightcurvedb_sources(df, {})

    assert len(sources) == 1
    assert sources[0].name == "SO-T J023444+0757.9"
    assert sources[0].socat_id is None


def test_unresolvable_rows_are_dropped_not_crashed():
    """No source_id and no crossmatch (shouldn't normally happen for the
    eligible categories, but must not blow up if it does)."""
    df = pd.DataFrame(
        [
            _row(source_id=None, ra=1.0, dec=1.0, category="transient_candidates"),
            _row(source_id="ACT-S J0058.5+0620", ra=14.64, dec=6.33),
        ]
    )
    resolved = resolve_source_identity(df)
    assert len(resolved) == 1
    assert resolved.iloc[0]["_identity_name"] == "ACT-S J0058.5+0620"
