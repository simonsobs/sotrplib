import datetime

import pytest
from astropy import units as u

from sotrplib.maps.core import RhoAndKappaMap
from sotrplib.maps.map_coadding import RhoKappaMapCoadder


def test_rhokappa_coadder(map_set_1, map_set_2):
    """Test the RhoKappaMapCoadderConfig coadder."""
    map_path_1 = map_set_1
    map_path_2 = map_set_2

    start_time = datetime.datetime(2025, 10, 10, 0, 0, 0)
    input_maps = [
        RhoAndKappaMap(
            rho_filename=map_path_1["rho"],
            kappa_filename=map_path_1["kappa"],
            time_filename=map_path_1["time"],
            frequency="f090",
            start_time=start_time,
            end_time=start_time + datetime.timedelta(hours=1),
        ),
        RhoAndKappaMap(
            rho_filename=map_path_2["rho"],
            kappa_filename=map_path_2["kappa"],
            time_filename=map_path_2["time"],
            frequency="f090",
            start_time=start_time + datetime.timedelta(hours=1),
            end_time=start_time + datetime.timedelta(hours=2),
        ),
    ]

    coadder = RhoKappaMapCoadder(frequencies=["f090"])
    coadded_maps = coadder.coadd(input_maps=input_maps)

    # Check that the coadded map is a CoaddedRhoKappaMap
    assert isinstance(coadded_maps, list)
    assert coadded_maps[0].rho is not None
    assert coadded_maps[0].kappa is not None
    assert coadded_maps[0].time_mean is not None
    corners = coadded_maps[0].bbox
    assert corners[0].ra.to_value(u.deg) == pytest.approx(
        min(input_maps[0].bbox[0].ra, input_maps[1].bbox[0].ra).to_value(u.deg)
    )
    assert corners[0].dec.to_value(u.deg) == pytest.approx(
        min(input_maps[0].bbox[0].dec, input_maps[1].bbox[0].dec).to_value(u.deg)
    )

    assert corners[1].ra.to_value(u.deg) == pytest.approx(
        max(input_maps[0].bbox[1].ra, input_maps[1].bbox[1].ra).to_value(u.deg)
    )
    assert corners[1].dec.to_value(u.deg) == pytest.approx(
        max(input_maps[0].bbox[1].dec, input_maps[1].bbox[1].dec).to_value(u.deg)
    )
