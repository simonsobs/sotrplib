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


def _write_observations(run, hours):
    """observations.json with one observation at each of the hours after T0."""
    run.root.mkdir(parents=True, exist_ok=True)
    run.observations_file.write_text(
        json.dumps([(T0 + timedelta(hours=h)).isoformat() for h in hours])
    )


def test_list_observations(tmp_path, mock):
    feed = mock["feed"]
    source = tmp_path / "source.sqlite"
    _source_mapcat(source)
    # m0 and m1 are one observation (same start time).
    times = feed.list_observations(source, T0, T0 + timedelta(days=2))
    assert times == [
        T0 + timedelta(hours=2),
        T0 + timedelta(hours=20),
        T0 + timedelta(days=1, hours=3),
    ]
    # An observation with no map left after the filters is not in the list.
    times = feed.list_observations(
        source, T0, T0 + timedelta(days=2), bands=["f090", "f220"]
    )
    assert times == [T0 + timedelta(hours=2), T0 + timedelta(days=1, hours=3)]


def test_schedule_observations_weeks_and_months(tmp_path, mock):
    run = _run(mock, tmp_path)
    # Observations on days 0, 0, 6, 8 and 36 (hours 2, 20, 146, 194, 866).
    _write_observations(run, [2, 20, 146, 194, 866])
    start, end = run.obs_window(1)
    assert start == T0 + timedelta(hours=20) and end - start == timedelta(minutes=1)
    assert run.obs_dir(1).name == "obs0001_20250904T200000"
    ends = {o: run.week_ending_at(o) for o in range(5)}
    # Obs 2 (day 6) ends week 0, obs 3 (day 8) ends week 1, the last obs ends
    # its week (5).
    assert ends == {0: None, 1: None, 2: 0, 3: 1, 4: 5}
    assert run.week_window(5) == (T0 + timedelta(days=35), T0 + timedelta(days=37))
    # One full month (weeks 0-3, 28 days). Weeks 4-5 do not make a full month.
    assert [w for w in range(6) if run.ends_month(w) is not None] == [3]
    assert run.month_window(0) == (T0, T0 + timedelta(days=28))


def test_obs_config(tmp_path, mock):
    configs = mock["configs"]
    run = _run(mock, tmp_path)
    _write_observations(run, [2, 146])
    config = configs.obs_config(run, 1)

    maps = config["maps"]
    assert (maps["start_time"], maps["end_time"]) == (
        "2025-09-10T02:00:00Z",
        "2025-09-10T02:01:00Z",
    )
    assert maps["time_binning"] == "left-bound" and maps["track_processing"]
    (socat,) = config["source_catalogs"]
    assert socat["update"]["record_directory"].endswith(
        "obs0001_20250910T020000/socat_updates"
    )
    assert config["map_matcher"]["min_arrays"] == 2
    assert [o["output_type"] for o in config["source_outputs"]] == [
        "pickle",
        "lightcurvedb",
    ]
    assert set(config["preprocessors"][1]["beam1d"]) == {"f090", "f150", "f220", "f280"}

    run.lightcurvedb = False
    no_lcdb = configs.obs_config(run, 1)
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
    assert len(run.observations()) == 3

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


def test_backfill_measures_each_source_once(tmp_path, mock, monkeypatch):
    """lightcurvedb keeps duplicates, so a rerun of the backfill step must not
    measure a source that is done again."""
    mockrt = mock["mockrt"]
    run = _run(mock, tmp_path)
    ledger = mockrt.Ledger(tmp_path / "ledger.sqlite")
    record_dir = tmp_path / "socat_updates"
    record_dir.mkdir()
    (record_dir / "socat_update_20250910T052435.json").write_text(
        json.dumps({"maps": [], "groups": [{"action": "added", "socat_id": "a"}]})
    )

    measured = []

    class FakeExtractor:
        @staticmethod
        def main(argv):
            measured.append(argv[argv.index("--socat-id") + 1])
            return []

    monkeypatch.setattr(mockrt, "_load_script", lambda *args: FakeExtractor)
    end = datetime(2025, 9, 11, tzinfo=timezone.utc)

    out = tmp_path / "backfill"
    mockrt.do_backfill(run, ledger, record_dir, out, end)
    assert measured == ["a"]
    assert ledger.done("backfill:a")

    # A second source is added. A rerun measures only the new source.
    (record_dir / "socat_update_20250910T052435_rerun.json").write_text(
        json.dumps({"maps": [], "groups": [{"action": "added", "socat_id": "b"}]})
    )
    result = mockrt.do_backfill(run, ledger, record_dir, out, end)
    assert measured == ["a", "b"]
    assert result["socat_ids"] == ["b"]
    assert mockrt.do_backfill(run, ledger, record_dir, out, end) == {"n_sources": 0}


def _fake_sbatch(mockrt, monkeypatch):
    """Record the submitted scripts and their dependencies; return job ids."""
    jobs = []

    def sbatch(script, dependency=None, begin=None):
        jobs.append((Path(script).name, dependency, Path(script).read_text()))
        return str(len(jobs))

    monkeypatch.setattr(mockrt, "sbatch", sbatch)
    return jobs


class _FakeWeeks:
    @staticmethod
    def build_config(**kwargs):
        return {}


def test_sequential_week_analysis_submits_next_tick(tmp_path, mock, monkeypatch):
    mockrt = mock["mockrt"]
    run = _run(mock, tmp_path)
    run.slurm_dir.mkdir(parents=True)
    jobs = _fake_sbatch(mockrt, monkeypatch)
    monkeypatch.setattr(mockrt, "_load_script", lambda *args: _FakeWeeks)

    mockrt.submit_week(run, 0, next_obs=7)
    name, dependency, text = jobs[-1]
    assert name == "mockrt_week00_analysis.slurm"
    assert dependency == ["1", "2", "3", "4"]
    assert "--next-obs 7" in text


def test_sequential_month_waits_for_week_analysis(tmp_path, mock, monkeypatch):
    mockrt = mock["mockrt"]
    run = _run(mock, tmp_path)
    run.slurm_dir.mkdir(parents=True)
    jobs = _fake_sbatch(mockrt, monkeypatch)
    monkeypatch.setattr(mockrt, "_load_script", lambda *args: _FakeWeeks)

    # Week 3 ends month 0: the month job submits the next tick, not the week.
    detail = mockrt.submit_week(run, 3, next_obs=28)
    analysis, planner = jobs[-2], jobs[-1]
    assert "--next-obs" not in analysis[2]
    assert planner[0] == "mockrt_month00_plan.slurm"
    assert planner[1] == [detail["analysis_job"]]
    assert "--next-obs 28" in planner[2]

    # Not sequential: the month starts after the weekly coadds.
    jobs.clear()
    detail = mockrt.submit_week(run, 3)
    assert jobs[-1][1] == detail["coadd_jobs"]
    assert all("--next-obs" not in text for _, _, text in jobs)


@pytest.mark.parametrize("sequential", [True, False])
def test_tick_at_week_end(tmp_path, mock, monkeypatch, sequential):
    mockrt = mock["mockrt"]
    run = _run(mock, tmp_path, sequential=sequential)
    run.root.mkdir(parents=True)
    run.save()
    # Obs 1 (day 6) is the last observation of week 0.
    _write_observations(run, [2, 146, 194])
    calls = []
    monkeypatch.setattr(mockrt, "do_feed", lambda run, obs: {})
    monkeypatch.setattr(mockrt, "do_run", lambda run, obs: {})
    monkeypatch.setattr(mockrt, "do_backfill", lambda *args: {})
    monkeypatch.setattr(
        mockrt,
        "submit_week",
        lambda run, week, next_obs=None: calls.append(("week", week, next_obs)),
    )
    monkeypatch.setattr(
        mockrt, "submit_tick", lambda run, obs: calls.append(("tick", obs))
    )
    mockrt.main(["tick", "--obs", "1", "--run-dir", str(run.root)])
    if sequential:
        assert calls == [("week", 0, 2)]
    else:
        assert calls == [("week", 0, None), ("tick", 2)]

    # Obs 0 does not end a week: only the next tick.
    calls.clear()
    mockrt.main(["tick", "--obs", "0", "--run-dir", str(run.root)])
    assert calls == [("tick", 1)]


def _records(record_dir, ids):
    record_dir.mkdir()
    (record_dir / "socat_update_20250910T052435.json").write_text(
        json.dumps(
            {"maps": [], "groups": [{"action": "added", "socat_id": i} for i in ids]}
        )
    )


def test_backfill_runs_sources_in_parallel(tmp_path, mock, monkeypatch):
    import threading

    mockrt = mock["mockrt"]
    run = _run(mock, tmp_path, backfill_workers=4)
    ledger = mockrt.Ledger(tmp_path / "ledger.sqlite")
    _records(tmp_path / "socat_updates", ["a", "b", "c", "d"])
    monkeypatch.setattr(mockrt, "_load_script", lambda *args: None)

    # Each source waits until all four run at the same time.
    barrier = threading.Barrier(4, timeout=10)
    measured = []

    def backfill_source(run, extractor, socat_id, out_dir, end, lock=None):
        barrier.wait()
        measured.append(socat_id)

    monkeypatch.setattr(mockrt, "backfill_source", backfill_source)
    end = datetime(2025, 9, 11, tzinfo=timezone.utc)
    result = mockrt.do_backfill(
        run, ledger, tmp_path / "socat_updates", tmp_path / "backfill", end
    )
    assert sorted(measured) == ["a", "b", "c", "d"]
    assert result["n_sources"] == 4
    assert all(ledger.done(f"backfill:{i}") for i in "abcd")


def test_backfill_failure_does_not_stop_other_sources(tmp_path, mock, monkeypatch):
    mockrt = mock["mockrt"]
    run = _run(mock, tmp_path, backfill_workers=2)
    ledger = mockrt.Ledger(tmp_path / "ledger.sqlite")
    _records(tmp_path / "socat_updates", ["a", "bad", "c"])
    monkeypatch.setattr(mockrt, "_load_script", lambda *args: None)

    def backfill_source(run, extractor, socat_id, out_dir, end, lock=None):
        if socat_id == "bad":
            raise ValueError("no maps")

    monkeypatch.setattr(mockrt, "backfill_source", backfill_source)
    end = datetime(2025, 9, 11, tzinfo=timezone.utc)
    with pytest.raises(RuntimeError, match="1 of 3 backfills failed"):
        mockrt.do_backfill(
            run, ledger, tmp_path / "socat_updates", tmp_path / "backfill", end
        )
    assert ledger.done("backfill:a") and ledger.done("backfill:c")
    assert ledger.get("backfill:bad") == "failed"

    # A rerun measures only the failed source.
    rerun = []
    monkeypatch.setattr(
        mockrt,
        "backfill_source",
        lambda run, extractor, socat_id, *args, **kwargs: rerun.append(socat_id),
    )
    mockrt.do_backfill(
        run, ledger, tmp_path / "socat_updates", tmp_path / "backfill", end
    )
    assert rerun == ["bad"]
