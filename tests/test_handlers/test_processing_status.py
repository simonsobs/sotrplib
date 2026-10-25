"""
The runner sets the processing end status of a map only if the reader set
a processing start status (track_processing). A map read from mapcat with
track_processing false (for example by the historical extractor) has no
status row, and set_processing_end would raise.
"""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from sotrplib.handlers.basic import PipelineRunner
from sotrplib.sifter.core import SifterResult
from sotrplib.sifter.map_matching import MapResult


def _runner(source_outputs=None) -> PipelineRunner:
    return PipelineRunner(
        map_coadder=None,
        source_simulators=None,
        source_injector=None,
        source_catalogs=None,
        preprocessors=None,
        pointing_provider=None,
        pointing_residual_model=None,
        postprocessors=None,
        forced_photometry=None,
        source_subtractor=None,
        blind_search=None,
        sifter=None,
        source_outputs=source_outputs,
        map_outputs=None,
    )


def _empty_map(track_processing: bool):
    return SimpleNamespace(
        build=lambda: None,
        hits=np.zeros((4, 4)),
        mapcat_id="map-1",
        map_type="depth1_map",
        _parent_database="mapcat.sqlite",
        _track_processing=track_processing,
    )


def _result(track_processing: bool) -> MapResult:
    return MapResult(
        map_name="map",
        mapcat_id="map-1",
        array="i1",
        frequency="f090",
        observation_start=None,
        forced_photometry_candidates=[],
        sifter_result=SifterResult(
            source_candidates=[], transient_candidates=[], noise_candidates=[]
        ),
        track_processing=track_processing,
    )


class _FailingOutput:
    def output(self, **kwargs):
        raise RuntimeError("output failed")


@pytest.mark.parametrize("track_processing", [True, False])
def test_build_map_sets_end_status_of_empty_map_only_if_tracked(track_processing):
    with patch("sotrplib.handlers.base.set_processing_end") as mock_end:
        assert _runner().build_map(_empty_map(track_processing)) is None
    assert mock_end.called == track_processing


@pytest.mark.parametrize("track_processing", [True, False])
def test_output_map_result_sets_end_status_only_if_tracked(track_processing):
    with patch("sotrplib.handlers.base.set_processing_end") as mock_end:
        _runner().output_map_result(_result(track_processing))
    assert mock_end.called == track_processing


@pytest.mark.parametrize("track_processing", [True, False])
def test_failed_output_sets_failed_status_only_if_tracked(track_processing):
    with patch("sotrplib.handlers.base.set_processing_end") as mock_end:
        with pytest.raises(RuntimeError):
            _runner([_FailingOutput()]).output_map_result(_result(track_processing))
    if track_processing:
        assert mock_end.call_args.kwargs["status"] == "failed"
    else:
        assert not mock_end.called
