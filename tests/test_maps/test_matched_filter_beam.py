"""
Tests for choosing the matched filter's 1D beam profile and simple-model lknee per band.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from sotrplib.config.preprocessors import MatchedFilterConfig


def _map(frequency: str):
    return SimpleNamespace(frequency=frequency, map_name=f"{frequency}_i1_123")


def test_single_beam1d_used_for_every_band():
    matched_filter = MatchedFilterConfig(beam1d="profile_f090.txt").to_preprocessor()

    assert matched_filter._beam1d_for(_map("f090")) == Path("profile_f090.txt")
    assert matched_filter._beam1d_for(_map("f150")) == Path("profile_f090.txt")


def test_beam1d_per_band():
    matched_filter = MatchedFilterConfig.model_validate(
        {
            "preprocessor_type": "matched_filter",
            "beam1d": {"f090": "profile_f090.txt", "f150": "profile_f150.txt"},
        }
    ).to_preprocessor()

    assert matched_filter._beam1d_for(_map("f090")) == Path("profile_f090.txt")
    assert matched_filter._beam1d_for(_map("f150")) == Path("profile_f150.txt")


def test_beam1d_per_band_missing_band_raises():
    matched_filter = MatchedFilterConfig(
        beam1d={"f090": "profile_f090.txt"}
    ).to_preprocessor()

    with pytest.raises(ValueError, match="f220"):
        matched_filter._beam1d_for(_map("f220"))


def test_no_beam1d():
    assert MatchedFilterConfig().to_preprocessor()._beam1d_for(_map("f090")) is None


def test_single_lknee_used_for_every_band():
    matched_filter = MatchedFilterConfig(simple_lknee=2000).to_preprocessor()

    assert matched_filter._lknee_for(_map("f090")) == 2000
    assert matched_filter._lknee_for(_map("f280")) == 2000


def test_lknee_per_band():
    matched_filter = MatchedFilterConfig.model_validate(
        {
            "preprocessor_type": "matched_filter",
            "simple_lknee": {"f090": 2000, "f150": 3000},
        }
    ).to_preprocessor()

    assert matched_filter._lknee_for(_map("f090")) == 2000
    assert matched_filter._lknee_for(_map("f150")) == 3000
    with pytest.raises(ValueError, match="f220"):
        matched_filter._lknee_for(_map("f220"))
