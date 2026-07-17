from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
from astropy.time import Time, TimeDelta

from sotrplib.maps.core import CoaddedRhoKappaMap, RhoAndKappaMap
from sotrplib.maps.database import register_coadd
from sotrplib.maps.map_coadding import RhoKappaMapCoadder
from sotrplib.maps.streaming_coadd import stream_coadd


def _rho_kappa_map(paths, map_id, start_time):
    m = RhoAndKappaMap(
        rho_filename=paths["rho"],
        kappa_filename=paths["kappa"],
        time_filename=paths["time"],
        frequency="f090",
        start_time=start_time,
        end_time=start_time + TimeDelta(3600, format="sec"),
    )
    m.map_id = map_id
    return m


def test_stream_coadd_matches_batch_coadd(overlapping_map_set_1, overlapping_map_set_2):
    """stream_coadd() (one map at a time) must match RhoKappaMapCoadder.coadd()
    (all maps at once) exactly, for the same inputs and no preprocessing."""
    start_time = Time("2025-10-10", format="iso")
    paths = [overlapping_map_set_1, overlapping_map_set_2]

    batch_maps = [_rho_kappa_map(p, i, start_time) for i, p in enumerate(paths)]
    [m.build() for m in batch_maps]
    batch = RhoKappaMapCoadder(frequencies=["f090"]).coadd(batch_maps)[0]

    stream_maps = [_rho_kappa_map(p, i, start_time) for i, p in enumerate(paths)]
    streamed, map_ids = stream_coadd(
        maps=stream_maps,
        preprocessors=[],
        coadder=RhoKappaMapCoadder(frequencies=["f090"]),
    )

    assert map_ids == [0, 1]
    np.testing.assert_allclose(streamed.rho, batch.rho)
    np.testing.assert_allclose(streamed.kappa, batch.kappa)
    np.testing.assert_allclose(streamed.hits, batch.hits)
    np.testing.assert_allclose(streamed.time_mean, batch.time_mean)


def test_stream_coadd_tracks_all_map_ids(separate_map_set_1):
    """
    Regression test: RhoKappaMapCoadder.coadd_maps() seeds map_ids from
    base_map.map_id (singular), which is wrong once base_map is itself a
    running coadd from a previous streaming iteration -- it would silently
    drop everything merged before it. stream_coadd() must track map_ids
    itself instead, so all N maps show up regardless of coadder internals.
    """
    start_time = Time("2025-10-10", format="iso")
    maps = [
        _rho_kappa_map(separate_map_set_1, map_id, start_time)
        for map_id in [10, 20, 30, 40, 50]
    ]

    _, map_ids = stream_coadd(
        maps=maps,
        preprocessors=[],
        coadder=RhoKappaMapCoadder(frequencies=["f090"]),
    )

    assert map_ids == [10, 20, 30, 40, 50]


def test_stream_coadd_empty_input_returns_none():
    coadd, map_ids = stream_coadd(
        maps=[], preprocessors=[], coadder=RhoKappaMapCoadder(frequencies=["f090"])
    )
    assert coadd is None
    assert map_ids == []


def test_register_coadd_writes_row_and_links(separate_map_set_1):
    start_time = Time("2025-10-10", format="iso")
    coadd = CoaddedRhoKappaMap(
        rho=None,
        kappa=None,
        observation_start=start_time,
        observation_end=start_time + TimeDelta(3600, format="sec"),
        observation_length=TimeDelta(3600, format="sec"),
        frequency="f090",
    )

    linked_map = MagicMock()
    linked_map.map_id = 7
    session = MagicMock()
    session.execute.return_value.scalars.return_value.all.return_value = [linked_map]

    depth_one_parent = Path("/data/depth1")
    output_paths = {
        "flux": depth_one_parent / "coadds" / "f090_flux.fits",
        "rho": depth_one_parent / "coadds" / "f090_rho.fits",
        "kappa": depth_one_parent / "coadds" / "f090_kappa.fits",
        "time_mean": depth_one_parent / "coadds" / "f090_time_mean.fits",
    }

    from unittest.mock import patch

    with patch("sotrplib.maps.database.mapcat_settings") as settings:
        settings.depth_one_parent = depth_one_parent
        register_coadd(
            coadd=coadd,
            map_ids=["11111111-1111-1111-1111-111111111111"],
            coadd_name="f090_test_coadd",
            coadd_type="depth1_streaming_coadd",
            output_paths=output_paths,
            session=session,
        )

    assert session.add.called
    row = session.add.call_args.args[0]
    assert row.coadd_name == "f090_test_coadd"
    assert row.coadd_type == "depth1_streaming_coadd"
    assert row.frequency == "f090"
    assert row.map_path == "coadds/f090_flux.fits"
    assert row.rho_path == "coadds/f090_rho.fits"
    assert row.kappa_path == "coadds/f090_kappa.fits"
    assert row.mean_time_path == "coadds/f090_time_mean.fits"
    assert row.start_time_path is None
    assert row.end_time_path is None
    assert row.maps == [linked_map]
    assert session.commit.called


def test_coadded_rho_kappa_map_does_not_share_mutable_defaults():
    """
    Regression test: map_ids/input_map_times used to default to a single
    shared [] object, so every CoaddedRhoKappaMap built without passing
    them explicitly (which coadd_maps() never does for input_map_times)
    silently accumulated state across unrelated coadds in the same process.
    """
    start_time = Time("2025-10-10", format="iso")
    kwargs = dict(
        rho=None,
        kappa=None,
        observation_start=start_time,
        observation_end=start_time + TimeDelta(3600, format="sec"),
        observation_length=TimeDelta(3600, format="sec"),
        frequency="f090",
    )
    a = CoaddedRhoKappaMap(**kwargs)
    b = CoaddedRhoKappaMap(**kwargs)

    a.input_map_times.append("only-in-a")
    a.map_ids.append("only-in-a")

    assert b.input_map_times == []
    assert b.map_ids == []
