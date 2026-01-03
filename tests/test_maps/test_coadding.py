import datetime

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord

from sotrplib.maps.core import RhoAndKappaMap
from sotrplib.maps.map_coadding import RhoKappaMapCoadder
from sotrplib.sims import (
    source_injector,
)
from sotrplib.sims.sim_sources import FixedSimulatedSource
from sotrplib.source_catalog.core import RegisteredSource


def test_rhokappa_coadder_separate(separate_map_set_1, separate_map_set_2):
    """Test the RhoKappaMapCoadderConfig coadder."""
    map_path_1 = separate_map_set_1
    map_path_2 = separate_map_set_2

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

    for i in range(len(input_maps)):
        input_maps[i].build()

    assert np.all(input_maps[0].rho != input_maps[1].rho)

    coadder = RhoKappaMapCoadder(frequencies=["f090"])
    coadded_maps = coadder.coadd(input_maps=input_maps)
    assert isinstance(coadded_maps, list)

    coadd = coadded_maps[0]
    coadd.finalize()
    # Check that the coadded map is a CoaddedRhoKappaMap

    assert coadd.flux is not None
    assert coadd.snr is not None
    assert coadd.time_mean is not None
    corners = coadd.bbox
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
    assert np.all(coadd.hits <= 1)


def test_rhokappa_coadder_overlapping(overlapping_map_set_1, overlapping_map_set_2):
    """Test the RhoKappaMapCoadderConfig coadder."""
    map_path_1 = overlapping_map_set_1
    map_path_2 = overlapping_map_set_2

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
    [input_maps[i].build() for i in range(len(input_maps))]
    assert np.all(input_maps[0].rho != input_maps[1].rho)
    coadder = RhoKappaMapCoadder(frequencies=["f090"])
    coadded_maps = coadder.coadd(input_maps=input_maps)
    coadd = coadded_maps[0]
    coadd.finalize()
    # Check that the coadded map is a CoaddedRhoKappaMap
    assert isinstance(coadded_maps, list)
    assert coadd.flux is not None
    assert coadd.snr is not None
    assert coadd.time_mean is not None
    corners = coadd.bbox
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
    assert ~np.all(coadd.hits <= 1)


def test_coadding_with_sources_in_separate_maps(
    separate_map_set_1, separate_map_set_2, separate_source_params
):
    map_path_1 = separate_map_set_1
    map_path_2 = separate_map_set_2

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

    sources = [
        RegisteredSource(
            ra=separate_source_params[s]["ra"],
            dec=separate_source_params[s]["dec"],
            frequency=90.0 * u.GHz,
            flux=separate_source_params[s]["flux"],
            source_id=f"sim-{s}",
            source_type="simulated",
            err_ra=0.0 * u.deg,
            err_dec=0.0 * u.deg,
            err_flux=0.0 * u.Jy,
        )
        for s in separate_source_params
    ]

    [input_maps[i].build() for i in range(len(input_maps))]
    assert np.all(input_maps[0].rho != input_maps[1].rho)

    simulated_sources = [
        FixedSimulatedSource(position=SkyCoord(ra=x.ra, dec=x.dec), flux=x.flux)
        for x in sources
    ]

    coadder = RhoKappaMapCoadder(frequencies=["f090"])
    coadded_maps = coadder.coadd(input_maps=input_maps)
    coadd = coadded_maps[0]
    coadd.finalize()
    injector = source_injector.PhotutilsSourceInjector()
    new_map = injector.inject(input_map=coadd, simulated_sources=simulated_sources)
    new_map.finalize()

    assert new_map != coadd
    assert new_map.original_map == coadd

    source_pos_1 = coadd.flux.sky2pix(
        [
            separate_source_params["source1"]["dec"].to_value(u.rad),
            separate_source_params["source1"]["ra"].to_value(u.rad),
        ]
    )
    source_pos_2 = coadd.flux.sky2pix(
        [
            separate_source_params["source2"]["dec"].to_value(u.rad),
            separate_source_params["source2"]["ra"].to_value(u.rad),
        ]
    )
    assert new_map.flux[int(source_pos_1[0]), int(source_pos_1[1])] == pytest.approx(
        separate_source_params["source1"]["flux"].to_value(u.Jy), abs=0.5
    )
    assert new_map.flux[int(source_pos_2[0]), int(source_pos_2[1])] == pytest.approx(
        separate_source_params["source2"]["flux"].to_value(u.Jy), abs=0.5
    )
    assert coadd.hits[int(source_pos_1[0]), int(source_pos_1[1])] == 1
    assert coadd.hits[int(source_pos_2[0]), int(source_pos_2[1])] == 1


def test_coadding_with_sources_in_overlapping_maps(
    overlapping_map_set_1, overlapping_map_set_2, overlapping_source_params
):
    map_path_1 = overlapping_map_set_1
    map_path_2 = overlapping_map_set_2

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

    sources = [
        RegisteredSource(
            ra=overlapping_source_params[s]["ra"],
            dec=overlapping_source_params[s]["dec"],
            frequency=90.0 * u.GHz,
            flux=overlapping_source_params[s]["flux"],
            source_id=f"sim-{s}",
            source_type="simulated",
            err_ra=0.0 * u.deg,
            err_dec=0.0 * u.deg,
            err_flux=0.0 * u.Jy,
        )
        for s in overlapping_source_params
    ]

    [input_maps[i].build() for i in range(len(input_maps))]
    assert np.all(input_maps[0].rho != input_maps[1].rho)

    simulated_sources = [
        FixedSimulatedSource(position=SkyCoord(ra=x.ra, dec=x.dec), flux=x.flux)
        for x in sources
    ]

    coadder = RhoKappaMapCoadder(frequencies=["f090"])
    coadded_maps = coadder.coadd(input_maps=input_maps)
    coadd = coadded_maps[0]

    coadd.finalize()
    injector = source_injector.PhotutilsSourceInjector()
    new_map = injector.inject(input_map=coadd, simulated_sources=simulated_sources)
    new_map.finalize()

    assert new_map != coadd
    assert new_map.original_map == coadd

    source_pos_1 = coadd.flux.sky2pix(
        [
            overlapping_source_params["source1"]["dec"].to_value(u.rad),
            overlapping_source_params["source1"]["ra"].to_value(u.rad),
        ]
    )
    source_pos_2 = coadd.flux.sky2pix(
        [
            overlapping_source_params["source2"]["dec"].to_value(u.rad),
            overlapping_source_params["source2"]["ra"].to_value(u.rad),
        ]
    )
    assert new_map.flux[int(source_pos_1[0]), int(source_pos_1[1])] == pytest.approx(
        overlapping_source_params["source1"]["flux"].to_value(u.Jy), abs=0.5
    )
    assert new_map.flux[int(source_pos_2[0]), int(source_pos_2[1])] == pytest.approx(
        overlapping_source_params["source2"]["flux"].to_value(u.Jy), abs=0.5
    )
    assert coadd.hits[int(source_pos_1[0]), int(source_pos_1[1])] == 2
    assert coadd.hits[int(source_pos_2[0]), int(source_pos_2[1])] == 1
