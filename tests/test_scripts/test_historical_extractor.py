"""
Tests for the map selection and the configs of the mapcat historical
lightcurve extractor script.
"""

import importlib.util
import json
import sqlite3
import uuid
from pathlib import Path

import pytest

from sotrplib.config.config import Settings

SCRIPT = (
    Path(__file__).parents[2]
    / "scripts"
    / "historical_lightcurve_extractor"
    / "slurm_wrapper_mapcat_historical_extractor.py"
)


@pytest.fixture(scope="module")
def extractor():
    spec = importlib.util.spec_from_file_location("historical_extractor", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MAPS = [
    # name, tube, band, start, covers the source
    ("a", "i1", "f090", "2025-09-08 05:00:00", True),
    ("b", "i3", "f150", "2025-09-09 05:00:00", True),
    ("c", "i1", "f090", "2025-09-10 05:00:00", True),
    ("d", "i1", "f090", "2025-09-12 05:00:00", True),
    ("e", "i6", "f090", "2025-09-09 06:00:00", False),
    ("f", "i4", "f220", "2025-09-09 07:00:00", True),
]


def _mapcat(path: Path, extractor, with_coverage: bool = True) -> dict[str, str]:
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE depth_one_maps (map_id TEXT, map_name TEXT, tube_slot TEXT, "
        "frequency TEXT, start_time TEXT)"
    )
    con.execute("CREATE TABLE depth_one_sky_coverage (x INT, y INT, map_id TEXT)")
    x, y = extractor.coverage_index(348.347, 2.676)
    ids = {}
    for name, tube, band, start, covers in MAPS:
        map_id = uuid.uuid4().hex
        ids[name] = str(uuid.UUID(map_id))
        con.execute(
            "INSERT INTO depth_one_maps VALUES (?, ?, ?, ?, ?)",
            (map_id, name, tube, band, start),
        )
        if with_coverage:
            cx = x if covers else (x + 5) % 36
            con.execute(
                "INSERT INTO depth_one_sky_coverage VALUES (?, ?, ?)", (cx, y, map_id)
            )
    con.commit()
    con.close()
    return ids


def test_maps_covering_applies_coverage_bands_tubes_and_time(tmp_path, extractor):
    ids = _mapcat(tmp_path / "mapcat.sqlite", extractor)
    maps = extractor.maps_covering(
        str(tmp_path / "mapcat.sqlite"),
        348.347,
        2.676,
        bands=["f090", "f150"],
        tubes=["i1", "i3", "i6"],
        start_time="2025-09-09T00:00:00",
        end_time="2025-09-11T00:00:00",
    )
    # a: before start_time. d: after end_time. e: no coverage. f: band.
    assert [m["name"] for m in maps] == ["b", "c"]
    assert [m["map_id"] for m in maps] == [ids["b"], ids["c"]]


def test_maps_covering_without_coverage_table_rows(tmp_path, extractor, capsys):
    _mapcat(tmp_path / "mapcat.sqlite", extractor, with_coverage=False)
    maps = extractor.maps_covering(
        str(tmp_path / "mapcat.sqlite"),
        348.347,
        2.676,
        bands=["f090"],
        tubes=["i1", "i6"],
        end_time="2025-09-11T00:00:00",
    )
    assert [m["name"] for m in maps] == ["a", "e", "c"]
    assert "WARNING" in capsys.readouterr().out


def test_config_reads_only_the_maps_and_does_not_track_processing(tmp_path, extractor):
    args = extractor.parser().parse_args(
        ["--out-dir", str(tmp_path), "--code-dir", "/code", "--lightcurvedb"]
    )
    source = extractor.Source(name="SZ Psc", ra=348.347, dec=2.676, socat_id="x")
    maps = [
        {"map_id": str(uuid.uuid4()), "band": "f090"},
        {"map_id": str(uuid.uuid4()), "band": "f150"},
    ]
    raw = extractor.config(source, maps, str(tmp_path) + "/", args)

    settings = Settings.model_validate(json.loads(json.dumps(raw)))
    reader = settings.maps
    assert reader.track_processing is False
    assert reader.rerun is False
    assert [str(i) for i in reader.map_ids] == [m["map_id"] for m in maps]
    assert reader.number_to_read == 2
    lower, upper = reader.sky_box
    assert lower.ra.deg == pytest.approx(347.847)
    assert upper.dec.deg == pytest.approx(3.176)
    assert set(settings.preprocessors[1].beam1d) == {"f090", "f150"}
    assert [o.output_type for o in settings.source_outputs] == [
        "pickle",
        "lightcurvedb",
    ]


def test_copy_socat_sources_keeps_the_source_id(tmp_path, extractor):
    from astropy import units as u
    from astropy.coordinates import ICRS

    source_db = str(tmp_path / "socat.db")
    created = extractor.socat_client(source_db).create_source(
        position=ICRS(ra=348.347 * u.deg, dec=2.676 * u.deg),
        flux=800 * u.mJy,
        name="SOT J231323.3+024034",
        flags={"monitored": False},
    )
    target_db = str(tmp_path / "jobs_socat.db")
    sources = extractor.copy_socat_sources(
        source_db, [str(created.source_id)], target_db
    )

    assert sources[0].socat_id == str(created.source_id)
    assert sources[0].flux_jy == pytest.approx(0.8)
    copied = extractor.socat_client(target_db).get_source(source_id=created.source_id)
    assert copied.name == "SOT J231323.3+024034"
    assert copied.monitored

    # A second copy does not make a duplicate.
    extractor.copy_socat_sources(source_db, [str(created.source_id)], target_db)
    con = sqlite3.connect(target_db)
    assert con.execute("SELECT COUNT(*) FROM fixed_sources").fetchone() == (1,)
