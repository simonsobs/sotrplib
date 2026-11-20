"""
Tests for the DefaultSifter cuts in the sifter config.
"""

import json

import numpy as np

from sotrplib.config.sifter import DefaultSifterConfig
from sotrplib.sifter.core import DEFAULT_SIFTER_CUTS


def test_default_cuts_unchanged():
    sifter = DefaultSifterConfig().to_sifter()
    assert sifter.cuts == DEFAULT_SIFTER_CUTS
    assert sifter.cuts is not DEFAULT_SIFTER_CUTS


def test_cuts_override_only_given_keys():
    config = DefaultSifterConfig.model_validate(
        json.loads('{"sifter_type": "default", "cuts": {"snr": [3.0, "inf"]}}')
    )
    sifter = config.to_sifter()
    assert sifter.cuts["snr"] == [3.0, np.inf]
    assert sifter.cuts["fwhm"] == DEFAULT_SIFTER_CUTS["fwhm"]
    assert DEFAULT_SIFTER_CUTS["snr"] == [5.0, np.inf]
