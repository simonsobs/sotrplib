import pytest
from astropy import units as u
from structlog import get_logger

log = get_logger()


def test_ra_lims_valid():
    from sotrplib.sims.sim_utils import ra_lims_valid

    assert ra_lims_valid((0, 360) * u.deg)
    assert ra_lims_valid((0, 180) * u.deg)
    assert not ra_lims_valid((-10, 10) * u.deg)
    assert not ra_lims_valid((370, 380) * u.deg)
    assert not ra_lims_valid((0, 370) * u.deg)
    assert not ra_lims_valid()
    assert not ra_lims_valid([0 * u.deg])


def test_dec_lims_valid():
    from sotrplib.sims.sim_utils import dec_lims_valid

    assert dec_lims_valid((-90, 90) * u.deg)
    assert dec_lims_valid((-45, 45) * u.deg)
    assert not dec_lims_valid((-100, 100) * u.deg)
    assert not dec_lims_valid((0, 100) * u.deg)
    assert not dec_lims_valid()
    assert not dec_lims_valid([0 * u.deg])


def test_random_positions(
    sim_map_params,
    log=log,
):
    from sotrplib.sims.sim_utils import generate_random_positions

    with pytest.raises(ValueError):
        generate_random_positions(0, log=log)

    ra_lims = (
        sim_map_params["maps"]["center_ra"] - sim_map_params["maps"]["width_ra"] / 2,
        sim_map_params["maps"]["center_ra"] + sim_map_params["maps"]["width_ra"] / 2,
    )
    dec_lims = (
        sim_map_params["maps"]["center_dec"] - sim_map_params["maps"]["width_dec"] / 2,
        sim_map_params["maps"]["center_dec"] + sim_map_params["maps"]["width_dec"] / 2,
    )
    positions = generate_random_positions(
        10, ra_lims=ra_lims, dec_lims=dec_lims, log=log
    )
    assert len(positions) == 10
    for dec, ra in positions:
        if ra_lims[0] < 0 * u.deg:
            if ra > 180 * u.deg:
                ra -= 360 * u.deg
        assert ra_lims[0] <= ra <= ra_lims[1]
        assert dec_lims[0] <= dec <= dec_lims[1]


def test_random_positions_with_map(sim_map_params, log=log):
    from sotrplib.sims import sim_maps
    from sotrplib.sims.sim_utils import generate_random_positions

    m = sim_maps.make_enmap(**sim_map_params["maps"], log=log)
    positions = generate_random_positions(10, imap=m, log=log)
    assert len(positions) == 10
    ra_lims = (
        sim_map_params["maps"]["center_ra"] - sim_map_params["maps"]["width_ra"],
        sim_map_params["maps"]["center_ra"] + sim_map_params["maps"]["width_ra"],
    )
    dec_lims = (
        sim_map_params["maps"]["center_dec"] - sim_map_params["maps"]["width_dec"],
        sim_map_params["maps"]["center_dec"] + sim_map_params["maps"]["width_dec"],
    )
    for dec, ra in positions:
        if ra_lims[0] < 0 * u.deg:
            if ra > 180 * u.deg:
                ra -= 360 * u.deg
        tol = 1e-2 * u.deg
        assert (ra_lims[0] - tol) <= ra <= (ra_lims[1] + tol), (
            f"RA {ra} out of range {ra_lims}"
        )
        assert (dec_lims[0] - tol) <= dec <= (dec_lims[1] + tol), (
            f"Dec {dec} out of range {dec_lims}"
        )


def test_random_flare_times(log=log):
    from sotrplib.sims.sim_utils import generate_random_flare_times

    flare_times = generate_random_flare_times(
        10, start_time=1.4e9, end_time=1.7e9, log=log
    )
    assert len(flare_times) == 10
    for t in flare_times:
        assert 1.4e9 <= t <= 1.7e9
