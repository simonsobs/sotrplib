"""
Tests for a coadd of registered coadds (e.g. a month of weekly coadds) with
sotrp-coadd: the config, the hits of an input coadd, the depth-1 links, and
the status rows of the input coadds.
"""

import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from astropy.time import Time
from mapcat.database import DepthOneCoaddTable, DepthOneMapTable
from pixell import enmap
from pydantic import ValidationError
from sqlmodel import Session, SQLModel, create_engine

from sotrplib.coadd_cli import main
from sotrplib.config.coadd import CoaddSettings
from sotrplib.maps.core import CoaddRhoAndKappaMap
from sotrplib.maps.database import depth_one_map_ids_of_coadds

T0 = datetime(2025, 9, 4, tzinfo=timezone.utc)


@pytest.fixture
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'mapcat.sqlite'}")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def _depth_one(name: str, day: int) -> DepthOneMapTable:
    t = T0 + timedelta(days=day)
    return DepthOneMapTable(
        map_name=name,
        mean_time_path=None,
        tube_slot="i1",
        frequency="f090",
        ctime=t,
        start_time=t,
        stop_time=t + timedelta(hours=4),
    )


def _coadd(name: str, maps: list[DepthOneMapTable]) -> DepthOneCoaddTable:
    return DepthOneCoaddTable(
        coadd_name=name,
        coadd_type="weekly",
        map_path=f"{name}_flux.fits",
        ivar_path=None,
        mean_time_path=None,
        frequency="f090",
        ctime=maps[0].ctime,
        start_time=maps[0].start_time,
        stop_time=maps[-1].stop_time,
        maps=maps,
    )


def test_depth_one_map_ids_of_coadds(session):
    week1 = [_depth_one(f"w1_{d}", d) for d in range(3)]
    week2 = [_depth_one(f"w2_{d}", 7 + d) for d in range(2)]
    c1, c2 = _coadd("week1", week1), _coadd("week2", week2)
    session.add_all([c1, c2])
    session.commit()

    ids = depth_one_map_ids_of_coadds([c1.coadd_id, c2.coadd_id], session=session)
    assert sorted(ids) == sorted(m.map_id for m in week1 + week2)


def test_depth_one_map_ids_of_coadds_rejects_shared_maps(session):
    shared = _depth_one("shared", 6)
    c1 = _coadd("week1", [_depth_one("a", 0), shared])
    c2 = _coadd("week2", [shared, _depth_one("b", 8)])
    session.add_all([c1, c2])
    session.commit()

    with pytest.raises(ValueError, match="more than one input coadd"):
        depth_one_map_ids_of_coadds([c1.coadd_id, c2.coadd_id], session=session)


def _settings(map_type: str, preprocessors: list | None = None) -> dict:
    return {
        "maps": {
            "map_generator_type": "mapcat_database",
            "map_type": map_type,
            **({"coadd_type": "weekly"} if map_type == "coadd_rhokappa" else {}),
        },
        "preprocessors": preprocessors or [],
    }


def test_coadd_settings_accept_coadd_inputs():
    config = CoaddSettings.model_validate(_settings("coadd_rhokappa"))
    assert config.inputs_are_coadds
    assert config.maps.coadd_type == "weekly"
    assert not CoaddSettings.model_validate(_settings("intensity")).inputs_are_coadds


def test_coadd_settings_reject_preprocessors_for_coadd_inputs():
    with pytest.raises(ValidationError, match="already"):
        CoaddSettings.model_validate(
            _settings("coadd_rhokappa", [{"preprocessor_type": "kappa_rho"}])
        )


def test_coadd_settings_reject_other_map_types():
    with pytest.raises(ValidationError, match="must be 'intensity' or"):
        CoaddSettings.model_validate(_settings("rhokappa"))


def _write_coadd(tmp_path, hits_value: int):
    shape, wcs = enmap.geometry(
        pos=np.deg2rad([[-1, 1], [1, -1]]), res=np.deg2rad(0.5 / 60)
    )
    paths = {}
    for field, value in (("rho", 2.0), ("kappa", 4.0), ("hits", hits_value)):
        paths[field] = tmp_path / f"f090_i1_1756951951_{field}.fits"
        enmap.write_map(str(paths[field]), enmap.full(shape, wcs, value))
    return paths


def test_coadd_map_reads_hits_file(tmp_path):
    paths = _write_coadd(tmp_path, hits_value=7)
    m = CoaddRhoAndKappaMap(
        rho_filename=paths["rho"],
        kappa_filename=paths["kappa"],
        start_time=Time(T0),
        end_time=Time(T0 + timedelta(days=7)),
        frequency="f090",
        hits_filename=paths["hits"],
    )
    m.build()
    assert np.all(m.hits == 7)

    no_hits = CoaddRhoAndKappaMap(
        rho_filename=paths["rho"],
        kappa_filename=paths["kappa"],
        start_time=Time(T0),
        end_time=Time(T0 + timedelta(days=7)),
        frequency="f090",
    )
    no_hits.build()
    assert np.all(no_hits.hits == 1)


@pytest.mark.parametrize("track_processing", [False, True])
def test_main_links_coadd_of_coadds_to_depth_one_maps(track_processing):
    """The new coadd links to the depth-1 maps of its input coadds, and the
    input coadds get status rows with map_type "coadd"."""
    config = MagicMock()
    config.log_level = logging.INFO
    config.inputs_are_coadds = True
    config.mapcat_registration.enabled = True
    reader = MagicMock()
    reader.track_processing = track_processing
    reader.map_ids = ["week1", "week2"]
    reader.__iter__.return_value = iter(
        [MagicMock(mapcat_id="week1"), MagicMock(mapcat_id="week2")]
    )
    config.to_dependencies.return_value = {
        "maps": reader,
        "preprocessors": [],
        "coadder": MagicMock(),
        "map_outputs": [],
    }
    ends = []

    with (
        patch("sotrplib.coadd_cli.parse_args"),
        patch("sotrplib.coadd_cli.CoaddSettings.from_file", return_value=config),
        patch("sotrplib.coadd_cli._check_registration_paths"),
        patch(
            "sotrplib.coadd_cli.stream_coadd",
            return_value=(MagicMock(), ["week1", "week2"], []),
        ),
        patch(
            "sotrplib.coadd_cli.depth_one_map_ids_of_coadds",
            return_value=["d1", "d2", "d3"],
        ) as links,
        patch("sotrplib.coadd_cli.register_coadd", return_value="month") as register,
        patch("sotrplib.coadd_cli.set_processing_start"),
        patch(
            "sotrplib.coadd_cli.set_processing_end",
            side_effect=lambda i, **kw: ends.append((i, kw["map_type"], kw["status"])),
        ),
    ):
        main()

    # Checked before the coadd work, then used for the links.
    assert links.call_count == 2
    assert register.call_args.kwargs["map_ids"] == ["d1", "d2", "d3"]
    expected = [("month", "coadd", "completed")]
    if track_processing:
        expected += [
            ("week1", "coadd", "completed"),
            ("week2", "coadd", "completed"),
        ]
    assert ends == expected
