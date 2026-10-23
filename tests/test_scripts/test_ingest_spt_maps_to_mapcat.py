"""
Tests scripts/mock_spt_run/ingest_spt_maps_to_mapcat.py: building
DepthOneMapTable rows from list_grid_maps.py's per-observation cache.

Only the pure logic is tested here (field mapping, obsid selection); reading
a real .g3 file needs spt3g_software and is exercised indirectly through
SPTFluxAndSNRMap.build().
"""

import importlib.util
import json
import sys
from datetime import timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from astropy.time import Time

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "mock_spt_run"
    / "ingest_spt_maps_to_mapcat.py"
)


@pytest.fixture(scope="module")
def ingest_module():
    spec = importlib.util.spec_from_file_location(
        "ingest_spt_maps_to_mapcat", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ─── build_row ──────────────────────────────────────────────────────────────


def test_build_row_maps_fields(ingest_module, tmp_path):
    start = Time("2026-01-01T00:00:00")
    stop = Time("2026-01-01T00:05:00")
    with patch.object(
        ingest_module,
        "observation_frame_times",
        return_value=(start, stop, "ra0hdec-52.25"),
    ) as fetch:
        row, subfield = ingest_module.build_row(
            obsid="123",
            band="90GHz",
            filename="123_90GHz_tonly.g3.gz",
            grid_root=None,
            scratch_dir=tmp_path,
        )
    fetch.assert_called_once_with("123_90GHz_tonly.g3.gz", tmp_path, keep=True)
    assert subfield == "ra0hdec-52.25"
    assert row.map_name == "spt_123_90GHz"
    assert row.map_path == "123_90GHz_tonly.g3.gz"
    assert row.tube_slot == "all"
    assert row.frequency == "90GHz"
    assert row.start_time == start.to_datetime(timezone=timezone.utc)
    assert row.stop_time == stop.to_datetime(timezone=timezone.utc)
    # ctime is the midpoint of start/stop.
    assert row.ctime == Time("2026-01-01T00:02:30").to_datetime(timezone=timezone.utc)


def test_build_row_uses_grid_url_when_grid_root_set(ingest_module, tmp_path):
    start = stop = Time("2026-01-01T00:00:00")
    grid_root = "davs://host:2880/sptgrid/data/onlinemaps/ra0hdec-52.25"
    with patch.object(
        ingest_module, "observation_frame_times", return_value=(start, stop, "x")
    ) as fetch:
        ingest_module.build_row(
            obsid="123",
            band="90GHz",
            filename="123_90GHz_tonly.g3.gz",
            grid_root=grid_root,
            scratch_dir=tmp_path,
        )
    fetch.assert_called_once_with(
        f"{grid_root}/123_90GHz_tonly.g3.gz", tmp_path, keep=True
    )


def test_build_row_respects_keep_flag(ingest_module, tmp_path):
    start = stop = Time("2026-01-01T00:00:00")
    with patch.object(
        ingest_module, "observation_frame_times", return_value=(start, stop, "x")
    ) as fetch:
        ingest_module.build_row(
            obsid="123",
            band="90GHz",
            filename="f.g3.gz",
            grid_root=None,
            scratch_dir=tmp_path,
            keep=False,
        )
    fetch.assert_called_once_with("f.g3.gz", tmp_path, keep=False)


# ─── main: obsid selection ──────────────────────────────────────────────────


@pytest.fixture
def maps_cache(tmp_path):
    cache = {
        "100": {"90GHz": "100_90GHz_tonly.g3.gz", "150GHz": "100_150GHz_tonly.g3.gz"},
        "200": {"90GHz": "200_90GHz_tonly.g3.gz"},
    }
    path = tmp_path / "maps.json"
    path.write_text(json.dumps(cache))
    return path


def test_main_ingests_all_obsids_by_default(ingest_module, maps_cache, tmp_path):
    session = MagicMock()
    session.__enter__.return_value = session
    settings = MagicMock()
    settings.session.return_value = session
    with (
        patch.object(ingest_module, "build_row") as build_row,
        patch("mapcat.helper.settings", settings),
    ):
        build_row.return_value = (MagicMock(start_time="t0", stop_time="t1"), "field")
        ingest_module.main(
            [
                "--maps-cache",
                str(maps_cache),
                "--scratch-dir",
                str(tmp_path),
            ]
        )
    # 2 files for obsid 100, 1 for obsid 200.
    assert build_row.call_count == 3
    session.add.assert_called()
    session.commit.assert_called_once()


def test_main_restricts_to_given_obsids(ingest_module, maps_cache, tmp_path):
    session = MagicMock()
    session.__enter__.return_value = session
    settings = MagicMock()
    settings.session.return_value = session
    with (
        patch.object(ingest_module, "build_row") as build_row,
        patch("mapcat.helper.settings", settings),
    ):
        build_row.return_value = (MagicMock(start_time="t0", stop_time="t1"), "field")
        ingest_module.main(
            [
                "--maps-cache",
                str(maps_cache),
                "--scratch-dir",
                str(tmp_path),
                "--obsids",
                "200",
            ]
        )
    assert build_row.call_count == 1
    build_row.assert_called_once_with(
        "200", "90GHz", "200_90GHz_tonly.g3.gz", None, tmp_path, keep=True
    )


def test_main_skips_unknown_obsid(ingest_module, maps_cache, tmp_path, capsys):
    session = MagicMock()
    session.__enter__.return_value = session
    settings = MagicMock()
    settings.session.return_value = session
    with (
        patch.object(ingest_module, "build_row") as build_row,
        patch("mapcat.helper.settings", settings),
    ):
        ingest_module.main(
            [
                "--maps-cache",
                str(maps_cache),
                "--scratch-dir",
                str(tmp_path),
                "--obsids",
                "999",
            ]
        )
    build_row.assert_not_called()
    assert "skip 999" in capsys.readouterr().out


def test_main_no_keep_scratch_flag_propagates(ingest_module, maps_cache, tmp_path):
    session = MagicMock()
    session.__enter__.return_value = session
    settings = MagicMock()
    settings.session.return_value = session
    with (
        patch.object(ingest_module, "build_row") as build_row,
        patch("mapcat.helper.settings", settings),
    ):
        build_row.return_value = (MagicMock(start_time="t0", stop_time="t1"), "field")
        ingest_module.main(
            [
                "--maps-cache",
                str(maps_cache),
                "--scratch-dir",
                str(tmp_path),
                "--obsids",
                "200",
                "--no-keep-scratch",
            ]
        )
    build_row.assert_called_once_with(
        "200", "90GHz", "200_90GHz_tonly.g3.gz", None, tmp_path, keep=False
    )
