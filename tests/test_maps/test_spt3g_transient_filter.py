"""
Tests for SPT3GTransientFilter / SPT3GTransientFilterConfig. The real
preprocess() path requires spt3g_software and a real .g3 file, so it's
validated by hand against real SPT3G data (see scripts/mock_spt_run);
these tests only cover config wiring and the input-type guard clause.
"""

from types import SimpleNamespace

import pytest

from sotrplib.config.preprocessors import (
    AllPreprocessorConfigTypes,
    SPT3GTransientFilterConfig,
)
from sotrplib.maps.preprocessor import SPT3GTransientFilter


def test_default_config_round_trip():
    preprocessor = SPT3GTransientFilterConfig().to_preprocessor()

    assert isinstance(preprocessor, SPT3GTransientFilter)
    assert preprocessor.mask is False
    assert preprocessor.field is None
    assert preprocessor.high_pass is True
    assert preprocessor.match_sources is True
    # Mirrors transients/scripts/filter_maps.py: normalize_weights and
    # compute_noisechisq are only disabled by --filter-coadd mode, not by
    # the absence of a coadd (which is our only mode, since no per-field
    # coadd map is staged).
    assert preprocessor.normalize_weights is True
    assert preprocessor.compute_noisechisq is True
    assert preprocessor.compute_snr_annulus is False


def test_config_params_pass_through():
    preprocessor = SPT3GTransientFilterConfig(
        mask=True,
        field="ra0hdec-52.25",
        high_pass=False,
        match_sources=False,
        normalize_weights=False,
        compute_noisechisq=False,
        compute_snr_annulus=True,
    ).to_preprocessor()

    assert preprocessor.mask is True
    assert preprocessor.field == "ra0hdec-52.25"
    assert preprocessor.high_pass is False
    assert preprocessor.match_sources is False
    assert preprocessor.normalize_weights is False
    assert preprocessor.compute_noisechisq is False
    assert preprocessor.compute_snr_annulus is True


def test_mask_without_field_raises():
    with pytest.raises(ValueError, match="field"):
        SPT3GTransientFilterConfig(mask=True).to_preprocessor()


def test_preprocessor_type_literal_is_spt3g_transient_filter():
    cfg = SPT3GTransientFilterConfig.model_validate(
        {"preprocessor_type": "spt3g_transient_filter"}
    )
    assert cfg.preprocessor_type == "spt3g_transient_filter"


def test_included_in_all_preprocessor_config_types():
    cfg = AllPreprocessorConfigTypes.__args__ if hasattr(
        AllPreprocessorConfigTypes, "__args__"
    ) else []
    assert SPT3GTransientFilterConfig in cfg


def test_preprocess_requires_map_filename_attribute():
    """
    Without map_filename (i.e. not a map built from a raw .g3 file),
    preprocess() should fail fast and clearly rather than reach the spt3g
    import.
    """
    preprocessor = SPT3GTransientFilterConfig().to_preprocessor()
    bad_map = SimpleNamespace(frequency="90GHz")

    with pytest.raises(TypeError, match="map_filename"):
        preprocessor.preprocess(bad_map)
