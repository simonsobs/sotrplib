"""
Tests for the mock real-time run scripts (scripts/mock_real_time/): the
feeder, the schedule, the generated configs and the ledger.
"""

import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).parents[2] / "scripts" / "mock_real_time"


@pytest.fixture(scope="module")
def mock():
    sys.path.insert(0, str(SCRIPT_DIR))
    try:
        import configs
        import feed
        import mockrt

        yield {"configs": configs, "feed": feed, "mockrt": mockrt}
    finally:
        sys.path.remove(str(SCRIPT_DIR))


T0 = datetime(2025, 9, 4, tzinfo=timezone.utc)


def _source_mapcat(path: Path):
    """A small mapcat with the tables that the feeder uses."""
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL);
        INSERT INTO alembic_version VALUES ('abc123');
        CREATE TABLE depth_one_maps (map_id CHAR(32) PRIMARY KEY, map_name TEXT,
            tube_slot TEXT, frequency TEXT, start_time DATETIME, stop_time DATETIME);
        CREATE INDEX ix_maps_start ON depth_one_maps (start_time);
        CREATE TABLE tod_depth_one (tod_id INTEGER PRIMARY KEY, obs_id TEXT);
        CREATE TABLE link_tod_to_depth_one_map (tod_id INTEGER, map_id CHAR(32),
            PRIMARY KEY (tod_id, map_id));
        CREATE TABLE depth_one_sky_coverage (x INT, y INT, map_id CHAR(32));
        CREATE TABLE time_domain_processing (processing_status_id INTEGER PRIMARY KEY,
            map_id CHAR(32), processing_status TEXT);
        """
    )
    maps = [
        ("m0", "17569/depth1_1756951951_i1_f090", "i1", "f090", 0, 2),
        ("m1", "17569/depth1_1756951951_c1_f220", "c1", "f220", 0, 2),
        ("m2", "17570/depth1_1757000000_i3_f150", "i3", "f150", 0, 20),
        ("m3", "17570/depth1_1757050000_i1_f090", "i1", "f090", 1, 3),
    ]
    for map_id, name, tube, band, day, hour in maps:
        t = T0 + timedelta(days=day, hours=hour)
        con.execute(
            "INSERT INTO depth_one_maps VALUES (?, ?, ?, ?, ?, ?)",
            (map_id, name, tube, band, t.strftime("%Y-%m-%d %H:%M:%S.000000"),
             (t + timedelta(hours=4)).strftime("%Y-%m-%d %H:%M:%S.000000")),
        )  # fmt: skip
    con.execute("INSERT INTO tod_depth_one VALUES (1, 'obs_a')")
    con.execute("INSERT INTO link_tod_to_depth_one_map VALUES (1, 'm0')")
    con.execute("INSERT INTO link_tod_to_depth_one_map VALUES (1, 'm1')")
    con.execute("INSERT INTO time_domain_processing VALUES (1, 'm0', 'completed')")
    con.commit()
    con.close()


def test_feed_copies_one_day_once(tmp_path, mock):
    feed = mock["feed"]
    source, live = tmp_path / "source.sqlite", tmp_path / "live.sqlite"
    _source_mapcat(source)
    feed.create_live_mapcat(source, live)

    con = sqlite3.connect(live)
    assert con.execute("SELECT version_num FROM alembic_version").fetchall() == [
        ("abc123",)
    ]
    assert con.execute("SELECT COUNT(*) FROM depth_one_maps").fetchone() == (0,)
    con.close()

    day0 = feed.feed_window(source, live, T0, T0 + timedelta(days=1))
    assert sorted(day0) == sorted(
        [
            "17569/depth1_1756951951_i1_f090",
            "17569/depth1_1756951951_c1_f220",
            "17570/depth1_1757000000_i3_f150",
        ]
    )
    # A second feed of the same day copies nothing.
    assert feed.feed_window(source, live, T0, T0 + timedelta(days=1)) == []

    con = sqlite3.connect(live)
    assert con.execute("SELECT COUNT(*) FROM depth_one_maps").fetchone() == (3,)
    assert con.execute("SELECT COUNT(*) FROM tod_depth_one").fetchone() == (1,)
    assert con.execute("SELECT COUNT(*) FROM link_tod_to_depth_one_map").fetchone() == (
        2,
    )
    # The processing status is not copied.
    assert con.execute("SELECT COUNT(*) FROM time_domain_processing").fetchone() == (0,)
    con.close()


def test_reset_unfinished_status_of_one_day(tmp_path, mock):
    feed = mock["feed"]
    source, live = tmp_path / "source.sqlite", tmp_path / "live.sqlite"
    _source_mapcat(source)
    feed.create_live_mapcat(source, live)
    feed.feed_window(source, live, T0, T0 + timedelta(days=2))
    con = sqlite3.connect(live)
    con.executemany(
        "INSERT INTO time_domain_processing (map_id, processing_status) VALUES (?, ?)",
        [
            ("m0", "processing"),
            ("m1", "failed"),
            ("m2", "completed"),
            ("m3", "processing"),
        ],
    )
    con.commit()
    con.close()

    assert feed.reset_unfinished_status(live, T0, T0 + timedelta(days=1)) == 2

    con = sqlite3.connect(live)
    rows = con.execute(
        "SELECT map_id, processing_status FROM time_domain_processing ORDER BY map_id"
    ).fetchall()
    con.close()
    # m2 (completed) and m3 (another day) stay.
    assert rows == [("m2", "completed"), ("m3", "processing")]


def test_feed_filters_bands_tubes_and_excluded_maps(tmp_path, mock):
    feed = mock["feed"]
    source, live = tmp_path / "source.sqlite", tmp_path / "live.sqlite"
    _source_mapcat(source)
    feed.create_live_mapcat(source, live)

    exclude_csv = tmp_path / "shifted.csv"
    exclude_csv.write_text("band,array,ctime,shift\nf220,c1,1756951951,36.5\n")
    exclude = feed.read_exclude_list(exclude_csv)
    assert exclude == {"depth1_1756951951_c1_f220"}

    new = feed.feed_window(
        source, live, T0, T0 + timedelta(days=2),
        bands=["f090", "f220"], tubes=["i1", "c1"], exclude=exclude,
    )  # fmt: skip
    assert sorted(new) == [
        "17569/depth1_1756951951_i1_f090",
        "17570/depth1_1757050000_i1_f090",
    ]


def _run(mock, tmp_path, **kw):
    RunConfig = mock["configs"].RunConfig
    return RunConfig(
        run_dir=str(tmp_path / "run"),
        source_mapcat="source.sqlite",
        source_socat="socat.db",
        depth_one_parent="/maps",
        code_dir="/code",
        start_time="2025-09-04T00:00:00Z",
        n_days=kw.pop("n_days", 37),
        **kw,
    )


def test_schedule_weeks_and_months(tmp_path, mock):
    run = _run(mock, tmp_path)
    assert run.day_window(1) == (T0 + timedelta(days=1), T0 + timedelta(days=2))
    ends = {
        d: run.ends_week(d) for d in range(run.n_days) if run.ends_week(d) is not None
    }
    # Weeks end on days 6, 13, 20, 27, 34; the last day ends a short week 5.
    assert ends == {6: 0, 13: 1, 20: 2, 27: 3, 34: 4, 36: 5}
    assert run.week_window(5) == (T0 + timedelta(days=35), T0 + timedelta(days=37))
    # One full month (weeks 0-3, 28 days). Weeks 4-5 do not make a full month.
    assert [w for w in range(6) if run.ends_month(w) is not None] == [3]
    assert run.month_window(0) == (T0, T0 + timedelta(days=28))


def test_daily_config(tmp_path, mock):
    configs = mock["configs"]
    run = _run(mock, tmp_path)
    config = configs.daily_config(run, 6)

    maps = config["maps"]
    assert (maps["start_time"], maps["end_time"]) == (
        "2025-09-10T00:00:00Z",
        "2025-09-11T00:00:00Z",
    )
    assert maps["time_binning"] == "left-bound" and maps["track_processing"]
    (socat,) = config["source_catalogs"]
    assert socat["update"]["record_directory"].endswith("day006_20250910/socat_updates")
    assert config["map_matcher"]["min_arrays"] == 2
    assert [o["output_type"] for o in config["source_outputs"]] == [
        "pickle",
        "lightcurvedb",
    ]
    assert set(config["preprocessors"][1]["beam1d"]) == {"f090", "f150", "f220", "f280"}

    run.lightcurvedb = False
    no_lcdb = configs.daily_config(run, 6)
    assert [o["output_type"] for o in no_lcdb["source_outputs"]] == ["pickle"]


def test_coadd_analysis_config_confirms_with_two_bands(tmp_path, mock):
    configs = mock["configs"]
    run = _run(mock, tmp_path)
    start, end = run.week_window(0)
    config = configs.coadd_analysis_config(run, "weekly", start, end, run.week_dir(0))

    assert config["maps"]["map_type"] == "coadd_rhokappa"
    assert config["maps"]["coadd_type"] == "weekly"
    assert config["preprocessors"] == []
    assert config["map_matcher"]["min_arrays"] == 1
    assert config["map_matcher"]["min_bands"] == 2


def test_init_writes_run_dir(tmp_path, mock):
    source = tmp_path / "source.sqlite"
    _source_mapcat(source)
    socat = tmp_path / "socat.db"
    socat.write_bytes(b"")
    mockrt = mock["mockrt"]
    mockrt.main(
        [
            "init",
            "--run-dir", str(tmp_path / "run"),
            "--source-mapcat", str(source),
            "--source-socat", str(socat),
            "--depth-one-parent", "/maps",
            "--code-dir", str(tmp_path),
            "--start-time", "2025-09-04T00:00:00Z",
            "--n-days", "3",
            "--runner", "basic",
        ]
    )  # fmt: skip
    run = mock["configs"].RunConfig.load(tmp_path / "run")
    assert run.n_days == 3 and run.runner == "basic"
    assert run.live_mapcat.exists() and run.live_socat.exists()
    env = run.env_file.read_text()
    assert f"MAPCAT_DATABASE_NAME={run.live_mapcat}" in env
    assert f"socat_model_database_name={run.live_socat}" in env
    assert "services.env" in env

    with pytest.raises(SystemExit):
        mockrt.main(
            [
                "init",
                "--run-dir", str(tmp_path / "run"),
                "--source-mapcat", str(source),
                "--source-socat", str(socat),
                "--depth-one-parent", "/maps",
                "--code-dir", str(tmp_path),
                "--start-time", "2025-09-04T00:00:00Z",
                "--n-days", "3",
            ]
        )  # fmt: skip


def test_ledger_runs_a_step_once(tmp_path, mock):
    mockrt = mock["mockrt"]
    ledger = mockrt.Ledger(tmp_path / "ledger.sqlite")
    calls = []

    def work(x):
        calls.append(x)
        return {"x": x}

    mockrt.step(ledger, "day000:feed", work, 1)
    mockrt.step(ledger, "day000:feed", work, 2)
    assert calls == [1]
    assert ledger.done("day000:feed")

    def broken():
        raise ValueError("no maps")

    with pytest.raises(ValueError):
        mockrt.step(ledger, "day000:daily", broken)
    assert ledger.get("day000:daily") == "failed"
    # A failed step runs again.
    mockrt.step(ledger, "day000:daily", work, 3)
    assert calls == [1, 3]


def test_added_sources_reads_socat_update_records(tmp_path, mock):
    mockrt = mock["mockrt"]
    (tmp_path / "socat_update_20250910T052435.json").write_text(
        json.dumps(
            {
                "maps": [],
                "groups": [
                    {"action": "added", "socat_id": "a"},
                    {"action": "already_in_catalog", "socat_id": "b"},
                    {"action": "monitored", "socat_id": "c"},
                ],
            }
        )
    )
    # A rerun: the sources of the first run are already in the catalog.
    (tmp_path / "socat_update_20250910T052435_rerun.json").write_text(
        json.dumps(
            {
                "maps": [],
                "groups": [
                    {"action": "already_in_catalog", "socat_id": "a"},
                    {"action": "added", "socat_id": "d"},
                ],
            }
        )
    )
    assert mockrt.added_sources(tmp_path) == ["a", "c", "d"]
    assert mockrt.added_sources(tmp_path / "missing") == []


def test_slurm_scripts(tmp_path, mock):
    configs = mock["configs"]
    run = _run(mock, tmp_path, postgres_image="/img/postgres_16.sif")
    tick = configs.slurm_script(
        run, "mockrt_tick000", "echo hi", 8, "128G", "04:00:00", prefect=True
    )
    assert "source " + str(run.env_file) in tick
    assert "prefect server start" in tick
    assert "ln -s" in tick and "de440s.bsp" in tick

    services = configs.services_script(run)
    assert "apptainer run" in services and "/img/postgres_16.sif" in services
    assert "lightcurvedb-setup" in services
    assert str(run.services_env) in services
