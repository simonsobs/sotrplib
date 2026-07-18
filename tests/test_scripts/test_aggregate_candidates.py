"""
Tests for scripts/historical_lightcurve_extractor/aggregate_candidates.py.
"""

import pickle
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.time import Time

from scripts.historical_lightcurve_extractor.aggregate_candidates import (
    CandidateRecord,
    apply_qa_cuts,
    apply_time_window,
    band_tag_from_ghz,
    cluster_candidates,
    exclude_known_catalog_matches,
    find_pickle_files,
    load_candidates,
    main,
    parse_map_id,
    summarize_cluster,
)
from sotrplib.sifter.core import SifterResult
from sotrplib.sources.sources import CrossMatch, MeasuredSource


def make_source(
    ra_deg=10.0,
    dec_deg=5.0,
    flux_mjy=50.0,
    snr=10.0,
    frequency_ghz=90.0,
    array="i1",
    fwhm_arcmin=2.2,
    thumbnail=None,
    crossmatches=None,
    observation_mean_time=None,
) -> MeasuredSource:
    return MeasuredSource(
        ra=ra_deg * u.deg,
        dec=dec_deg * u.deg,
        flux=flux_mjy * u.mJy,
        err_flux=1.0 * u.mJy,
        snr=snr,
        frequency=frequency_ghz * u.GHz if frequency_ghz is not None else None,
        array=array,
        fwhm_ra=fwhm_arcmin * u.arcmin,
        fwhm_dec=fwhm_arcmin * u.arcmin,
        thumbnail=thumbnail,
        crossmatches=crossmatches,
        observation_mean_time=observation_mean_time or Time("2025-09-10T00:00:00"),
    )


def make_record(
    source: MeasuredSource | None = None,
    map_id="f090_i1_1700000000",
    band="f090",
    tube="i1",
    epoch_unix=1700000000,
    bucket="transient",
) -> CandidateRecord:
    return CandidateRecord(
        source=source or make_source(),
        map_id=map_id,
        band=band,
        tube=tube,
        epoch_unix=epoch_unix,
        bucket=bucket,
        pickle_path=Path("fake.pickle"),
    )


# --- parse_map_id ---


def test_parse_map_id_depth1():
    assert parse_map_id("f090_i1_1725840000") == ("f090", "i1", 1725840000)


def test_parse_map_id_coadd_uuid():
    assert parse_map_id("e2abe992-e8e5-4df5-bd72-f393911607e1") == (None, None, None)


def test_parse_map_id_garbage():
    assert parse_map_id("not_a_map_id") == (None, None, None)


# --- band_tag_from_ghz ---


def test_band_tag_from_ghz_exact():
    assert band_tag_from_ghz(90.0) == "f090"
    assert band_tag_from_ghz(220.0) == "f220"


def test_band_tag_from_ghz_near():
    assert band_tag_from_ghz(95.0) == "f090"


def test_band_tag_from_ghz_out_of_tolerance():
    assert band_tag_from_ghz(500.0) is None


def test_band_tag_from_ghz_none():
    assert band_tag_from_ghz(None) is None


# --- exclude_known_catalog_matches ---


def test_exclude_known_catalog_matches_drops_crossmatched_by_default():
    unmatched = make_record(source=make_source(crossmatches=None))
    matched = make_record(
        source=make_source(
            crossmatches=[CrossMatch(source_id="ACT-S J0000", catalog_name="socat")]
        )
    )
    kept, excluded = exclude_known_catalog_matches(
        [unmatched, matched], include_known_source_flares=False
    )
    assert kept == [unmatched]
    assert excluded == [matched]


def test_exclude_known_catalog_matches_keeps_when_flag_set():
    matched = make_record(
        source=make_source(
            crossmatches=[CrossMatch(source_id="ACT-S J0000", catalog_name="socat")]
        )
    )
    kept, excluded = exclude_known_catalog_matches(
        [matched], include_known_source_flares=True
    )
    assert kept == [matched]
    assert excluded == [matched]


# --- apply_qa_cuts (cut-by-cut isolation) ---


def test_apply_qa_cuts_min_snr():
    low_snr = make_record(source=make_source(snr=2.0))
    high_snr = make_record(source=make_source(snr=20.0))
    kept, tally = apply_qa_cuts(
        [low_snr, high_snr],
        min_snr=8.0,
        max_fwhm_ratio=1000.0,
        max_flux_jy=1000.0,
        require_thumbnail=False,
    )
    assert kept == [high_snr]
    assert tally.get("min_snr") == 1


def test_apply_qa_cuts_max_flux_jy():
    faint = make_record(source=make_source(flux_mjy=10.0))
    bright = make_record(source=make_source(flux_mjy=200_000.0))  # 200 Jy
    kept, tally = apply_qa_cuts(
        [faint, bright],
        min_snr=0.0,
        max_fwhm_ratio=1000.0,
        max_flux_jy=100.0,
        require_thumbnail=False,
    )
    assert kept == [faint]
    assert tally.get("max_flux_jy") == 1


def test_apply_qa_cuts_require_thumbnail():
    no_thumb = make_record(source=make_source(thumbnail=None))
    with_thumb = make_record(source=make_source(thumbnail=np.zeros((4, 4))))
    kept, tally = apply_qa_cuts(
        [no_thumb, with_thumb],
        min_snr=0.0,
        max_fwhm_ratio=1000.0,
        max_flux_jy=1000.0,
        require_thumbnail=True,
    )
    assert kept == [with_thumb]
    assert tally.get("no_thumbnail") == 1


def test_apply_qa_cuts_max_fwhm_ratio():
    # nominal f090 fwhm is 2.2 arcmin; 10x that comfortably exceeds a 3x cut.
    sharp = make_record(source=make_source(fwhm_arcmin=2.2), band="f090", tube="i1")
    smeared = make_record(source=make_source(fwhm_arcmin=22.0), band="f090", tube="i1")
    kept, tally = apply_qa_cuts(
        [sharp, smeared],
        min_snr=0.0,
        max_fwhm_ratio=3.0,
        max_flux_jy=1000.0,
        require_thumbnail=False,
    )
    assert kept == [sharp]
    assert tally.get("max_fwhm_ratio") == 1


def test_apply_qa_cuts_max_beam_residual():
    beamlike = make_record(source=make_source())
    beamlike.source.beam_model_residual = 0.05
    artifact = make_record(source=make_source())
    artifact.source.beam_model_residual = 0.9
    unmeasured = make_record(source=make_source())  # no residual, no thumbnail
    kept, tally = apply_qa_cuts(
        [beamlike, artifact, unmeasured],
        min_snr=0.0,
        max_fwhm_ratio=1000.0,
        max_flux_jy=1000.0,
        require_thumbnail=False,
        max_beam_residual=0.3,
    )
    assert kept == [beamlike, unmeasured]
    assert tally.get("max_beam_residual") == 1


def test_apply_qa_cuts_max_beam_residual_computed_from_thumbnail():
    # old pickle: no stored residual, but a thumbnail is present. A flat
    # top-hat disk is nothing like the beam, so it must be cut and the
    # computed residual cached on the source.
    x = np.arange(41) - 20
    X, Y = np.meshgrid(x, x)
    disk = np.where(np.hypot(X, Y) <= 6.0, 1.0, 0.0)
    artifact = make_record(source=make_source(thumbnail=disk))
    artifact.source.thumbnail_res = u.Quantity([0.5, 0.5], "arcmin")
    kept, tally = apply_qa_cuts(
        [artifact],
        min_snr=0.0,
        max_fwhm_ratio=1000.0,
        max_flux_jy=1000.0,
        require_thumbnail=False,
        max_beam_residual=0.3,
    )
    assert kept == []
    assert tally.get("max_beam_residual") == 1
    assert artifact.source.beam_model_residual > 0.3


# --- apply_time_window ---


def test_apply_time_window_filters_depth1_keeps_coadd():
    in_window = make_record(epoch_unix=1_700_000_500)
    out_of_window = make_record(epoch_unix=1_700_999_999)
    coadd_record = make_record(
        map_id="some-uuid", band=None, tube=None, epoch_unix=None
    )

    kept = apply_time_window(
        [in_window, out_of_window, coadd_record],
        start_unix=1_700_000_000,
        stop_unix=1_700_000_600,
    )
    assert kept == [in_window, coadd_record]


# --- clustering ---


def test_cluster_candidates_merges_nearby_different_bands():
    close_a = make_record(
        source=make_source(ra_deg=10.0, dec_deg=5.0),
        map_id="f090_i1_1700000000",
        band="f090",
        tube="i1",
        epoch_unix=1700000000,
    )
    close_b = make_record(
        source=make_source(ra_deg=10.0 + 0.5 / 60.0, dec_deg=5.0),  # 0.5 arcmin away
        map_id="f150_i3_1700001000",
        band="f150",
        tube="i3",
        epoch_unix=1700001000,  # 1000s later, well within a 6h bucket
    )
    clusters = cluster_candidates([close_a, close_b], eps_arcmin=5.0, eps_hours=6.0)
    assert len(clusters) == 1
    assert len(clusters[0]) == 2


def test_cluster_candidates_splits_distant():
    here = make_record(source=make_source(ra_deg=10.0, dec_deg=5.0))
    far = make_record(source=make_source(ra_deg=10.5, dec_deg=5.0))  # ~30 arcmin away
    clusters = cluster_candidates([here, far], eps_arcmin=5.0, eps_hours=6.0)
    assert len(clusters) == 2
    assert all(len(c) == 1 for c in clusters)


def test_cluster_candidates_time_buckets_depth1_records():
    """Two detections at the same position but far apart in time (depth-1
    map_ids) should NOT merge -- different observation epochs."""
    early = make_record(
        source=make_source(ra_deg=10.0, dec_deg=5.0),
        epoch_unix=1_700_000_000,
    )
    late = make_record(
        source=make_source(ra_deg=10.0, dec_deg=5.0),
        epoch_unix=1_700_000_000 + int(30 * 3600),  # 30 hours later
    )
    clusters = cluster_candidates([early, late], eps_arcmin=5.0, eps_hours=6.0)
    assert len(clusters) == 2


def test_cluster_candidates_coadd_records_are_position_only():
    a = make_record(
        source=make_source(ra_deg=10.0, dec_deg=5.0),
        map_id="uuid-a",
        band=None,
        tube=None,
        epoch_unix=None,
    )
    b = make_record(
        source=make_source(ra_deg=10.0 + 0.5 / 60.0, dec_deg=5.0),
        map_id="uuid-b",
        band=None,
        tube=None,
        epoch_unix=None,
    )
    clusters = cluster_candidates([a, b], eps_arcmin=5.0, eps_hours=6.0)
    assert len(clusters) == 1
    assert len(clusters[0]) == 2


# --- summarize_cluster ---


def test_summarize_cluster_repeat_confirmed():
    from collections import Counter

    a = make_record(
        source=make_source(ra_deg=10.0, dec_deg=5.0, snr=10.0, flux_mjy=40.0),
        band="f090",
        tube="i1",
        epoch_unix=1_700_000_000,
    )
    b = make_record(
        source=make_source(ra_deg=10.001, dec_deg=5.0, snr=20.0, flux_mjy=60.0),
        band="f150",
        tube="i3",
        epoch_unix=1_700_001_000,
    )
    row = summarize_cluster([a, b], Counter())
    assert row["n_detections"] == 2
    assert row["n_bands"] == 2
    assert row["n_tubes"] == 2
    assert row["max_snr"] == 20.0
    assert row["flux_mjy"] == 60.0  # from the max-snr member
    assert row["flux_jy"] == pytest.approx(0.06)
    assert "singleton" not in row["qa_flags"]
    assert row["crossmatched"] is False


def test_summarize_cluster_singleton_flagged():
    from collections import Counter

    a = make_record(source=make_source())
    row = summarize_cluster([a], Counter())
    assert row["n_detections"] == 1
    assert "singleton" in row["qa_flags"].split(";")


def test_summarize_cluster_unique_cluster_ids_on_collision():
    from collections import Counter

    used_names = Counter()
    a = make_record(source=make_source(ra_deg=10.0, dec_deg=5.0))
    b = make_record(source=make_source(ra_deg=10.0, dec_deg=5.0))
    row_a = summarize_cluster([a], used_names)
    row_b = summarize_cluster([b], used_names)
    assert row_a["cluster_id"] != row_b["cluster_id"]


# --- end-to-end smoke test against real pickle files ---


def _write_pickle(path: Path, map_id: str, transient_candidates: list[MeasuredSource]):
    data = {
        "map_id": map_id,
        "forced_photometry": [],
        "sifted_blind_search": SifterResult(
            source_candidates=[],
            transient_candidates=transient_candidates,
            noise_candidates=[],
        ),
        "pointing_sources": [],
        "injected_sources": [],
    }
    with open(path, "wb") as fh:
        pickle.dump(data, fh)


def test_main_end_to_end_smoke(tmp_path):
    pickle_dir = tmp_path / "pickles"
    pickle_dir.mkdir()

    # Repeat-confirmed candidate: same position, two different bands/pickles.
    _write_pickle(
        pickle_dir / "week1_f090.pickle",
        "f090_i1_1700000000",
        [make_source(ra_deg=10.0, dec_deg=5.0, snr=15.0)],
    )
    _write_pickle(
        pickle_dir / "week1_f150.pickle",
        "f150_i3_1700001000",
        [make_source(ra_deg=10.001, dec_deg=5.0, snr=25.0)],
    )
    # Singleton: distinct position, only ever seen once.
    _write_pickle(
        pickle_dir / "week2_f090.pickle",
        "f090_i1_1700100000",
        [make_source(ra_deg=50.0, dec_deg=-10.0, snr=12.0)],
    )

    pickle_paths = find_pickle_files(pickle_dir, "*.pickle", recursive=True)
    assert len(pickle_paths) == 3

    records = load_candidates(pickle_paths, include_noise=False)
    assert len(records) == 3

    records, _ = exclude_known_catalog_matches(
        records, include_known_source_flares=False
    )
    records, _ = apply_qa_cuts(
        records,
        min_snr=8.0,
        max_fwhm_ratio=3.0,
        max_flux_jy=100.0,
        require_thumbnail=False,
    )
    assert len(records) == 3

    clusters = cluster_candidates(records, eps_arcmin=5.0, eps_hours=6.0)
    assert len(clusters) == 2

    sizes = sorted(len(c) for c in clusters)
    assert sizes == [1, 2]


def test_main_writes_two_csvs(tmp_path, capsys):
    pickle_dir = tmp_path / "pickles"
    pickle_dir.mkdir()
    _write_pickle(
        pickle_dir / "a.pickle",
        "f090_i1_1700000000",
        [make_source(ra_deg=10.0, dec_deg=5.0, snr=15.0)],
    )
    _write_pickle(
        pickle_dir / "b.pickle",
        "f150_i3_1700001000",
        [make_source(ra_deg=10.001, dec_deg=5.0, snr=25.0)],
    )
    _write_pickle(
        pickle_dir / "c.pickle",
        "f090_i1_1700100000",
        [make_source(ra_deg=50.0, dec_deg=-10.0, snr=12.0)],
    )

    out_dir = tmp_path / "out"
    import sys

    argv = [
        "aggregate-candidates",
        "--pickle-dir",
        str(pickle_dir),
        "--out-dir",
        str(out_dir),
        "--no-save-thumbnail-pngs",
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        main()
    finally:
        sys.argv = old_argv

    import csv

    with open(out_dir / "candidates.csv") as fh:
        repeat_rows = list(csv.DictReader(fh))
    with open(out_dir / "candidates_singletons.csv") as fh:
        singleton_rows = list(csv.DictReader(fh))

    assert len(repeat_rows) == 1
    assert int(repeat_rows[0]["n_detections"]) == 2
    assert len(singleton_rows) == 1
    assert int(singleton_rows[0]["n_detections"]) == 1
    assert singleton_rows[0]["flux_jy"] != ""
    assert singleton_rows[0]["ra_deg"] != ""


def test_main_is_idempotent(tmp_path):
    pickle_dir = tmp_path / "pickles"
    pickle_dir.mkdir()
    _write_pickle(
        pickle_dir / "a.pickle",
        "f090_i1_1700000000",
        [make_source(ra_deg=10.0, dec_deg=5.0, snr=15.0)],
    )
    _write_pickle(
        pickle_dir / "b.pickle",
        "f150_i3_1700001000",
        [make_source(ra_deg=10.001, dec_deg=5.0, snr=25.0)],
    )

    import sys

    def run(out_dir):
        argv = [
            "aggregate-candidates",
            "--pickle-dir",
            str(pickle_dir),
            "--out-dir",
            str(out_dir),
            "--no-save-thumbnail-pngs",
        ]
        old_argv = sys.argv
        sys.argv = argv
        try:
            main()
        finally:
            sys.argv = old_argv

    out_dir_1, out_dir_2 = tmp_path / "out1", tmp_path / "out2"
    run(out_dir_1)
    run(out_dir_2)

    assert (out_dir_1 / "candidates.csv").read_text() == (
        out_dir_2 / "candidates.csv"
    ).read_text()
