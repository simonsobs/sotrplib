"""
Tests for sotrp's --mode analyze/match/run split: analyze builds maps and
pickles the resulting MapResult list without matching; match loads pickled
MapResult lists (no map building) and matches + writes outputs; run is the
unmodified, single-shot default. This is the split a per-map grid job
("analyze") and a per-observation matching job ("match") rely on.
"""

import pickle
from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

from sotrplib.cli import main, parse_args


def _mock_config(maps):
    config = MagicMock()
    config.log_level = 20
    config.seed = None
    config.maps = maps
    return config


def _args(mode, config_path, result_out=None, result_in=None):
    return Namespace(
        config=config_path,
        mode=mode,
        result_out=result_out,
        result_in=result_in,
    )


def test_run_mode_calls_pipeline_run(tmp_path):
    config = _mock_config(maps=[])
    pipeline = config.to_runner.return_value

    with (
        patch("sotrplib.cli.parse_args", return_value=_args("run", tmp_path / "c.json")),
        patch("sotrplib.cli.Settings.from_file", return_value=config),
    ):
        main()

    pipeline.run.assert_called_once()
    pipeline.analyze.assert_not_called()
    pipeline.match_and_output.assert_not_called()


def test_analyze_mode_pickles_results(tmp_path):
    config = _mock_config(maps=[])
    pipeline = config.to_runner.return_value
    pipeline.analyze.return_value = ["one_map_result", "another"]
    result_out = tmp_path / "out" / "result.pickle"

    with (
        patch(
            "sotrplib.cli.parse_args",
            return_value=_args("analyze", tmp_path / "c.json", result_out=result_out),
        ),
        patch("sotrplib.cli.Settings.from_file", return_value=config),
    ):
        main()

    pipeline.analyze.assert_called_once()
    pipeline.run.assert_not_called()
    assert pickle.loads(result_out.read_bytes()) == ["one_map_result", "another"]


def test_match_mode_loads_and_concatenates_pickles(tmp_path):
    config = _mock_config(maps=[])
    pipeline = config.to_runner.return_value

    p1 = tmp_path / "band1.pickle"
    p2 = tmp_path / "band2.pickle"
    p1.write_bytes(pickle.dumps(["r1"]))
    p2.write_bytes(pickle.dumps(["r2", "r3"]))

    with (
        patch(
            "sotrplib.cli.parse_args",
            return_value=_args("match", tmp_path / "c.json", result_in=[p1, p2]),
        ),
        patch("sotrplib.cli.Settings.from_file", return_value=config),
    ):
        main()

    pipeline.match_and_output.assert_called_once_with(["r1", "r2", "r3"])
    pipeline.run.assert_not_called()
    pipeline.analyze.assert_not_called()


def test_parse_args_requires_result_out_for_analyze(tmp_path, monkeypatch):
    import sys

    monkeypatch.setattr(
        sys, "argv", ["sotrp", "-c", str(tmp_path / "c.json"), "--mode", "analyze"]
    )
    with pytest.raises(SystemExit):
        parse_args()


def test_parse_args_requires_result_in_for_match(tmp_path, monkeypatch):
    import sys

    monkeypatch.setattr(
        sys, "argv", ["sotrp", "-c", str(tmp_path / "c.json"), "--mode", "match"]
    )
    with pytest.raises(SystemExit):
        parse_args()
