"""
Tests for sotrp-coadd's exit-status handling: maps a run touches should end
up "completed" on success or "failed" on any exception, never left dangling
as "processing" (which would make the reader silently skip them next time).
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from sotrplib.coadd_cli import main


def _mock_config(map_ids):
    config = MagicMock()
    config.log_level = logging.INFO
    config.mapcat_registration.enabled = False  # keep register_coadd() out of scope
    reader = MagicMock()
    reader.map_ids = map_ids
    config.to_dependencies.return_value = {
        "maps": reader,
        "preprocessors": [],
        "coadder": MagicMock(),
        "map_outputs": [],
    }
    return config, reader


def test_main_marks_maps_completed_on_success():
    config, reader = _mock_config(map_ids=[1, 2, 3])
    coadd = MagicMock()

    with (
        patch("sotrplib.coadd_cli.parse_args"),
        patch("sotrplib.coadd_cli.CoaddSettings.from_file", return_value=config),
        patch("sotrplib.coadd_cli._check_registration_paths"),
        patch("sotrplib.coadd_cli.stream_coadd", return_value=(coadd, [1, 2, 3])),
        patch("sotrplib.coadd_cli.set_processing_end") as mock_end,
    ):
        main()

    assert {c.args[0] for c in mock_end.call_args_list} == {1, 2, 3}
    for c in mock_end.call_args_list:
        assert c.kwargs["status"] == "completed"


def test_main_marks_maps_failed_on_exception_and_reraises():
    config, reader = _mock_config(map_ids=[10, 20])

    with (
        patch("sotrplib.coadd_cli.parse_args"),
        patch("sotrplib.coadd_cli.CoaddSettings.from_file", return_value=config),
        patch("sotrplib.coadd_cli._check_registration_paths"),
        patch("sotrplib.coadd_cli.stream_coadd", side_effect=RuntimeError("boom")),
        patch("sotrplib.coadd_cli.set_processing_end") as mock_end,
    ):
        with pytest.raises(RuntimeError, match="boom"):
            main()

    assert {c.args[0] for c in mock_end.call_args_list} == {10, 20}
    for c in mock_end.call_args_list:
        assert c.kwargs["status"] == "failed"


def test_main_partial_failure_marks_all_read_maps_failed():
    """
    If stream_coadd merges some maps before crashing on a later one, none of
    them ended up in a registered coadd -- so all maps the reader read for
    this run (reader.map_ids, populated eagerly up front) should be marked
    failed, not just the ones that got merged before the crash.
    """
    config, reader = _mock_config(map_ids=[1, 2, 3, 4, 5])

    with (
        patch("sotrplib.coadd_cli.parse_args"),
        patch("sotrplib.coadd_cli.CoaddSettings.from_file", return_value=config),
        patch("sotrplib.coadd_cli._check_registration_paths"),
        patch(
            "sotrplib.coadd_cli.stream_coadd",
            side_effect=RuntimeError("crashed on map 3"),
        ),
        patch("sotrplib.coadd_cli.set_processing_end") as mock_end,
    ):
        with pytest.raises(RuntimeError):
            main()

    assert {c.args[0] for c in mock_end.call_args_list} == {1, 2, 3, 4, 5}


def test_main_no_maps_found_marks_nothing():
    config, reader = _mock_config(map_ids=[])

    with (
        patch("sotrplib.coadd_cli.parse_args"),
        patch("sotrplib.coadd_cli.CoaddSettings.from_file", return_value=config),
        patch("sotrplib.coadd_cli._check_registration_paths"),
        patch("sotrplib.coadd_cli.stream_coadd", return_value=(None, [])),
        patch("sotrplib.coadd_cli.set_processing_end") as mock_end,
    ):
        main()

    mock_end.assert_not_called()


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


def test_main_registers_coadd_and_records_its_status():
    """
    With registration enabled, the new coadd needs a status row before
    set_processing_end() can mark it completed (the real function raises if
    none exists). Previously this raised, which marked every input map
    "failed" even though the coadd had been written and registered.
    """
    config, reader = _mock_config(map_ids=[1, 2, 3])
    config.mapcat_registration.enabled = True
    coadd = MagicMock()
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
        patch("sotrplib.coadd_cli.stream_coadd", return_value=(coadd, [1, 2, 3])),
        patch("sotrplib.coadd_cli.register_coadd", return_value="coadd-id"),
        patch("sotrplib.coadd_cli.set_processing_start", side_effect=fake_start),
        patch("sotrplib.coadd_cli.set_processing_end", side_effect=fake_end),
    ):
        main()

    assert rows[("coadd", "coadd-id")] == "completed"
    assert all(rows[("depth1_map", i)] == "completed" for i in (1, 2, 3))
