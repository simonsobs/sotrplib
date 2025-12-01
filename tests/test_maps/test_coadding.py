import datetime

import pytest
from astropy import units as u

from sotrplib.config.map_coadding import RhoKappaMapCoadderConfig
from sotrplib.maps.core import CoaddedRhoKappaMap, RhoAndKappaMap


def test_rhokappa_coadder(map_params, tmp_path, map_set_1, map_set_2):
    """Test the RhoKappaMapCoadderConfig coadder."""
    map_path_1 = map_set_1
    map_path_2 = map_set_2

    input_maps = [
        RhoAndKappaMap(
            rho_filename=map_path_1["rho"],
            kappa_filename=map_path_1["kappa"],
            time_filename=map_path_1["time"],
            start_time=datetime.datetime.now(),
            end_time=datetime.datetime.now() + datetime.timedelta(hours=1),
        ),
        RhoAndKappaMap(
            rho_filename=map_path_2["rho"],
            kappa_filename=map_path_2["kappa"],
            time_filename=map_path_2["time"],
            start_time=datetime.datetime.now() + datetime.timedelta(hours=1),
            end_time=datetime.datetime.now() + datetime.timedelta(hours=2),
        ),
    ]

    coadder_config = RhoKappaMapCoadderConfig()
    coadded_map = coadder_config.coadd(input_maps=input_maps)

    # Check that the coadded map is a CoaddedRhoKappaMap
    assert isinstance(coadded_map, CoaddedRhoKappaMap)
    assert coadded_map.rho is not None
    assert coadded_map.kappa is not None
    assert coadded_map.time_mean is not None
    corners = coadded_map.bbox

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
    # Check that the coadded map has the correct number of input maps
    assert len(coadded_map.input_maps) == len(input_maps)
