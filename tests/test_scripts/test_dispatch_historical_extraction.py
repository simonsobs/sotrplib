"""
Tests for scripts/historical_lightcurve_extractor/dispatch_historical_extraction.py.
"""

import csv
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.historical_lightcurve_extractor.dispatch_historical_extraction import (
    build_wrapper_command,
    chunk,
    ensure_trailing_slash,
    read_candidate_csv,
    register_socat_sources,
    resolve_slurm_script_dir,
    sync_lightcurvedb,
    validate_rows,
)

_COLUMNS = [
    "cluster_id",
    "ra_deg",
    "dec_deg",
    "position_scatter_arcmin",
    "flux_mjy",
    "flux_jy",
    "max_snr",
    "n_detections",
    "n_bands",
    "n_tubes",
    "bands",
    "tubes",
    "map_ids",
    "first_obs_unix",
    "last_obs_unix",
    "fwhm_ra_arcmin",
    "fwhm_dec_arcmin",
    "crossmatched",
    "known_source_id",
    "thumbnail_png",
    "qa_flags",
]


def _write_csv(path: Path, rows: list[dict]):
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_COLUMNS)
        writer.writeheader()
        for row in rows:
            full_row = {col: row.get(col, "") for col in _COLUMNS}
            writer.writerow(full_row)


def make_row(**overrides) -> dict:
    row = {
        "cluster_id": "SO-SV_TEST_001",
        "ra_deg": "20.0",
        "dec_deg": "3.0",
        "flux_mjy": "50.0",
        "flux_jy": "0.05",
        "max_snr": "10.0",
        "n_detections": "2",
    }
    row.update(overrides)
    return row


# --- read_candidate_csv / validate_rows ---


def test_read_candidate_csv_round_trip(tmp_path):
    csv_path = tmp_path / "candidates.csv"
    _write_csv(csv_path, [make_row()])
    rows = read_candidate_csv(csv_path)
    assert len(rows) == 1
    assert rows[0]["cluster_id"] == "SO-SV_TEST_001"


def test_validate_rows_passes_good_data():
    rows = [make_row(cluster_id="a"), make_row(cluster_id="b")]
    assert validate_rows(rows) == rows


def test_validate_rows_rejects_missing_flux():
    rows = [make_row(flux_jy="")]
    with pytest.raises(ValueError, match="flux_jy"):
        validate_rows(rows)


def test_validate_rows_rejects_missing_ra():
    rows = [make_row(ra_deg="")]
    with pytest.raises(ValueError, match="ra_deg"):
        validate_rows(rows)


def test_validate_rows_rejects_duplicate_cluster_id():
    rows = [make_row(cluster_id="dup"), make_row(cluster_id="dup")]
    with pytest.raises(ValueError, match="Duplicate cluster_id"):
        validate_rows(rows)


# --- chunk ---


def test_chunk_zero_size_yields_one_batch():
    assert list(chunk([1, 2, 3], 0)) == [[1, 2, 3]]


def test_chunk_splits_into_batches():
    assert list(chunk([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]


# --- ensure_trailing_slash / resolve_slurm_script_dir ---


def test_ensure_trailing_slash_adds_when_missing():
    assert ensure_trailing_slash("/a/b") == "/a/b/"


def test_ensure_trailing_slash_idempotent():
    assert ensure_trailing_slash("/a/b/") == "/a/b/"


def test_resolve_slurm_script_dir_matches_wrapper_concat_convention():
    # Must exactly replicate the wrapper's own naive string concatenation
    # (args.out_dir + args.slurm_script_dir, not os.path.join) so Script B
    # looks in the same place the wrapper actually wrote to. Passing a
    # Path (rather than a pre-slashed str) here would silently strip the
    # trailing slash and produce the wrong, run-together path -- this is
    # exactly the bug this function's docstring warns about.
    assert resolve_slurm_script_dir("/a/b/") == Path("/a/b/slurm_job_scripts/")


# --- build_wrapper_command ---


def test_build_wrapper_command_index_aligned():
    import argparse

    rows = [
        make_row(cluster_id="c1", ra_deg="10.0", dec_deg="1.0", flux_jy="0.01"),
        make_row(cluster_id="c2", ra_deg="20.0", dec_deg="2.0", flux_jy="0.02"),
    ]
    args = argparse.Namespace(
        socat_db_path=Path("/fake/socat.db"),
        bands=["f090"],
        optics_tubes=None,
        flux_threshold=None,
        ncores=None,
        nserial=None,
        save_thumbnails=None,
        thumbnail_radius=None,
        box_half_width=None,
        group_name=None,
        script_dir=None,
        scratch_dir=None,
    )
    cmd = build_wrapper_command(Path("/fake/wrapper.py"), rows, "/fake/out/", args)

    ra_idx = cmd.index("--ra")
    dec_idx = cmd.index("--dec")
    flux_idx = cmd.index("--flux")
    source_id_idx = cmd.index("--source-id")

    assert cmd[ra_idx + 1 : ra_idx + 3] == ["10.0", "20.0"]
    assert cmd[dec_idx + 1 : dec_idx + 3] == ["1.0", "2.0"]
    assert cmd[flux_idx + 1 : flux_idx + 3] == ["0.01", "0.02"]
    assert cmd[source_id_idx + 1 : source_id_idx + 3] == ["c1", "c2"]

    assert "--also-output-lightcurvedb" in cmd
    assert "--bands" in cmd
    assert cmd[cmd.index("--bands") + 1] == "f090"
    # Unset passthrough flags must not appear at all.
    assert "--optics-tubes" not in cmd
    assert "--ncores" not in cmd


def test_build_wrapper_command_boolean_passthrough():
    import argparse

    args = argparse.Namespace(
        socat_db_path=Path("/fake/socat.db"),
        bands=None,
        optics_tubes=None,
        flux_threshold=None,
        ncores=None,
        nserial=None,
        save_thumbnails=True,
        thumbnail_radius=None,
        box_half_width=None,
        group_name=None,
        script_dir=None,
        scratch_dir=None,
    )
    cmd = build_wrapper_command(
        Path("/fake/wrapper.py"), [make_row()], "/fake/out/", args
    )
    assert "--save-thumbnails" in cmd


# --- register_socat_sources (mocked, no real DB writes) ---


def test_register_socat_sources_dry_run_makes_no_client_calls(capsys):
    with patch("socat.client.db.Client") as mock_client_cls:
        register_socat_sources([make_row()], Path("/fake/socat.db"), dry_run=True)
        mock_client_cls.assert_not_called()
    captured = capsys.readouterr()
    assert "would register socat source" in captured.out


def test_register_socat_sources_calls_create_source_with_monitored_flag():
    rows = [make_row(cluster_id="c1", ra_deg="10.0", dec_deg="1.0", flux_jy="0.01")]

    mock_client = MagicMock()
    mock_source = MagicMock()
    mock_source.source_id = "fake-uuid"
    mock_client.create_source.return_value = mock_source

    with patch("socat.client.db.Client", return_value=mock_client) as mock_client_cls:
        register_socat_sources(rows, Path("/fake/socat.db"), dry_run=False)

    mock_client_cls.assert_called_once()
    call_kwargs = mock_client.create_source.call_args.kwargs
    assert call_kwargs["name"] == "c1"
    assert call_kwargs["flags"] == {"monitored": True}
    assert call_kwargs["position"].ra.to_value("deg") == pytest.approx(10.0)
    assert call_kwargs["position"].dec.to_value("deg") == pytest.approx(1.0)
    assert call_kwargs["flux"].to_value("Jy") == pytest.approx(0.01)


# --- sync_lightcurvedb graceful degradation ---


def test_sync_lightcurvedb_does_not_raise_on_subprocess_failure():
    import subprocess

    with patch(
        "subprocess.run",
        side_effect=subprocess.CalledProcessError(1, ["lightcurvedb-socat"]),
    ):
        # Must not raise -- socat registration/dispatch must not be
        # blocked by lightcurvedb being unreachable.
        sync_lightcurvedb(Path("/fake/socat.db"), dry_run=False)


def test_sync_lightcurvedb_dry_run_makes_no_subprocess_call():
    with patch("subprocess.run") as mock_run:
        sync_lightcurvedb(Path("/fake/socat.db"), dry_run=True)
        mock_run.assert_not_called()
