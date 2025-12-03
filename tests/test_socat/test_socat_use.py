"""
Tests for the SOCat source catalog integration
"""

import datetime
import os

from astropy import units as u


def test_socat_read(socat_pickle):
    """
    Test that the socat pickle fixture works.
    """

    os.environ["socat_client_client_type"] = "pickle"
    os.environ["socat_client_pickle_path"] = str(socat_pickle)

    from sotrplib.source_catalog.socat import SOCat

    cat = SOCat(flux_lower_limit=0.0 * u.Jy)

    sources = cat.get_all_sources()

    assert len(sources) == 128


def test_socat_source_generator(socat_pickle):
    """
    Test that the socat source generator works.
    """

    os.environ["socat_client_client_type"] = "pickle"
    os.environ["socat_client_pickle_path"] = str(socat_pickle)

    from sotrplib.sims.sim_source_generators import SOCatSourceGenerator

    flare_start = datetime.datetime.now()
    flare_end = flare_start + datetime.timedelta(days=1)
    flare_width_shortest = datetime.timedelta(hours=1)
    flare_width_longest = flare_width_shortest * 2.0

    source_generator = SOCatSourceGenerator(
        fraction_fixed=0.5,
        fraction_gaussian=0.25,
        flare_earliest_time=flare_start,
        flare_latest_time=flare_end,
        flare_width_shortest=flare_width_shortest,
        flare_width_longest=flare_width_longest,
        peak_amplitude_maximum_factor=10.0,
        peak_amplitude_minimum_factor=1.0,
    )

    sources, _ = source_generator.generate()

    assert len(sources) == int(0.75 * 128)
