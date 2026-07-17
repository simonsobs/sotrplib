"""
Tests for sotrplib.handlers.base's is_coadd-aware processing-status marking.
"""

from unittest.mock import MagicMock, patch

from sotrplib.handlers.base import _mark_processing_end


def test_mark_processing_end_uses_map_id_for_maps():
    input_map = MagicMock()
    input_map.is_coadd = False
    input_map.map_id = "some-map-id"

    with patch("sotrplib.handlers.base.set_processing_end") as mock_set:
        _mark_processing_end(input_map, status="completed")

    mock_set.assert_called_once_with(map_id="some-map-id", status="completed")


def test_mark_processing_end_uses_coadd_id_for_coadds():
    input_map = MagicMock()
    input_map.is_coadd = True
    input_map.map_id = "some-coadd-id"

    with patch("sotrplib.handlers.base.set_processing_end") as mock_set:
        _mark_processing_end(input_map, status="failed")

    mock_set.assert_called_once_with(coadd_id="some-coadd-id", status="failed")
