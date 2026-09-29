"""
Tests for the status handling of sotrp-coadd. By default, only the
registered coadd gets a status row. With maps.track_processing, each map
ends as "completed" or "failed", not "processing".
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from sotrplib.coadd_cli import main


def _mock_config(map_ids, track_processing=False):
    config = MagicMock()
    config.log_level = logging.INFO
    config.mapcat_registration.enabled = False  # keep register_coadd() out of scope
    reader = MagicMock()
    reader.map_ids = map_ids
    reader.track_processing = track_processing
    config.to_dependencies.return_value = {
        "maps": reader,
        "preprocessors": [],
        "coadder": MagicMock(),
        "map_outputs": [],
    }
    return config, reader


def _run(config, **stream_coadd_kwargs):
    with (
        patch("sotrplib.coadd_cli.parse_args"),
        patch("sotrplib.coadd_cli.CoaddSettings.from_file", return_value=config),
        patch("sotrplib.coadd_cli._check_registration_paths"),
        patch("sotrplib.coadd_cli.stream_coadd", **stream_coadd_kwargs),
        patch("sotrplib.coadd_cli.set_processing_end") as mock_end,
    ):
        main()
    return mock_end


# ─── default: no per-map status ──────────────────────────────────────────────


def test_main_leaves_map_status_alone_on_success():
    config, _ = _mock_config(map_ids=[1, 2, 3])
    mock_end = _run(config, return_value=(MagicMock(), [1, 2, 3], []))
    mock_end.assert_not_called()


def test_main_leaves_map_status_alone_on_exception_and_reraises():
    config, _ = _mock_config(map_ids=[10, 20])
    with pytest.raises(RuntimeError, match="boom"):
        _run(config, side_effect=RuntimeError("boom"))


def test_main_leaves_map_status_alone_when_maps_excluded():
    config, _ = _mock_config(map_ids=[1, 2, 3])
    mock_end = _run(config, return_value=(MagicMock(), [1, 3], [2]))
    mock_end.assert_not_called()


def test_main_raises_when_every_map_failed():
    config, _ = _mock_config(map_ids=[1, 2])
    with pytest.raises(RuntimeError, match="All 2 input maps failed"):
        _run(config, return_value=(None, [], [1, 2]))


def test_main_no_maps_found_marks_nothing():
    config, _ = _mock_config(map_ids=[])
    mock_end = _run(config, return_value=(None, [], []))
    mock_end.assert_not_called()


# ─── track_processing opted in ───────────────────────────────────────────────


def test_tracked_main_marks_merged_completed_and_excluded_failed():
    config, _ = _mock_config(map_ids=[1, 2, 3], track_processing=True)
    mock_end = _run(config, return_value=(MagicMock(), [1, 3], [2]))

    statuses = {c.args[0]: c.kwargs["status"] for c in mock_end.call_args_list}
    assert statuses == {1: "completed", 2: "failed", 3: "completed"}


def test_tracked_main_marks_all_read_maps_failed_on_exception():
    """
    If the run stops with an error, no map is in a registered coadd. Thus,
    each map that the reader read is marked failed.
    """
    config, _ = _mock_config(map_ids=[1, 2, 3, 4, 5], track_processing=True)

    with (
        patch("sotrplib.coadd_cli.parse_args"),
        patch("sotrplib.coadd_cli.CoaddSettings.from_file", return_value=config),
        patch("sotrplib.coadd_cli._check_registration_paths"),
        patch("sotrplib.coadd_cli.stream_coadd", side_effect=RuntimeError("crashed")),
        patch("sotrplib.coadd_cli.set_processing_end") as mock_end,
        pytest.raises(RuntimeError),
    ):
        main()

    assert {c.args[0] for c in mock_end.call_args_list} == {1, 2, 3, 4, 5}
    for c in mock_end.call_args_list:
        assert c.kwargs["status"] == "failed"


# ─── registration paths ──────────────────────────────────────────────────────


def _registration_config(directory):
    config = MagicMock()
    config.mapcat_registration.enabled = True
    output = MagicMock()
    output.directory = directory
    config.map_outputs = [output]
    return config


def test_check_registration_paths_uses_coadd_parent(tmp_path):
    from sotrplib.coadd_cli import _check_registration_paths

    depth_one_parent = tmp_path / "depth1"
    coadd_parent = tmp_path / "my_coadds"
    config = _registration_config(coadd_parent / "weekly")

    with patch("sotrplib.coadd_cli.mapcat_settings") as settings:
        settings.depth_one_parent = depth_one_parent
        settings.depth_one_coadd_parent = coadd_parent
        # Outside depth_one_parent but under depth_one_coadd_parent: allowed.
        _check_registration_paths(config)


def test_check_registration_paths_rejects_outside_coadd_parent(tmp_path):
    from sotrplib.coadd_cli import _check_registration_paths

    config = _registration_config(tmp_path / "elsewhere")

    with patch("sotrplib.coadd_cli.mapcat_settings") as settings:
        # Under depth_one_parent doesn't count -- only the coadd parent does.
        settings.depth_one_parent = tmp_path
        settings.depth_one_coadd_parent = tmp_path / "my_coadds"
        with pytest.raises(ValueError, match="MAPCAT_DEPTH_ONE_COADD_PARENT"):
            _check_registration_paths(config)


# ─── coadd status ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("track_processing", [False, True])
def test_main_registers_coadd_and_records_its_status(track_processing):
    """
    With registration, the coadd gets a status row before
    set_processing_end() marks it completed. The input maps get a status
    only with track_processing.
    """
    config, _ = _mock_config(map_ids=[1, 2, 3], track_processing=track_processing)
    config.mapcat_registration.enabled = True
    rows: dict = {}

    def fake_start(mapcat_id, *, map_type="depth1_map", **_):
        rows[(map_type, mapcat_id)] = "processing"

    def fake_end(mapcat_id, *, map_type="depth1_map", status="completed", **_):
        if map_type == "coadd" and (map_type, mapcat_id) not in rows:
            raise ValueError("No processing_start status found")
        rows[(map_type, mapcat_id)] = status

    with (
        patch("sotrplib.coadd_cli.parse_args"),
        patch("sotrplib.coadd_cli.CoaddSettings.from_file", return_value=config),
        patch("sotrplib.coadd_cli._check_registration_paths"),
        patch(
            "sotrplib.coadd_cli.stream_coadd",
            return_value=(MagicMock(), [1, 2, 3], []),
        ),
        patch("sotrplib.coadd_cli.register_coadd", return_value="coadd-id"),
        patch("sotrplib.coadd_cli.set_processing_start", side_effect=fake_start),
        patch("sotrplib.coadd_cli.set_processing_end", side_effect=fake_end),
    ):
        main()

    assert rows[("coadd", "coadd-id")] == "completed"
    depth1_rows = {k: v for k, v in rows.items() if k[0] == "depth1_map"}
    if track_processing:
        assert depth1_rows == {("depth1_map", i): "completed" for i in (1, 2, 3)}
    else:
        assert depth1_rows == {}
