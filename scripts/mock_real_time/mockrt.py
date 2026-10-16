#!/usr/bin/env python3
"""
Mock real-time run of the time-domain pipeline.

A mock real-time run replays the depth-1 maps of a source mapcat, one
simulated day at a time, into a live mapcat. For each day ("tick"):

1. feed: copy the maps of the day into the live mapcat, and fill their sky
   coverage.
2. daily: run sotrp on the new maps of the day (all bands in one run, multi-
   array map matching). Forced photometry goes to lightcurvedb. The socat
   update adds the confirmed map-match groups to the live socat.
3. backfill: for each new socat source, measure the source in all the
   earlier maps (historical lightcurve extractor) and write to lightcurvedb.
4. At the end of each week: submit the weekly coadd jobs (one for each band)
   and a job that runs sotrp on the weekly coadds (two-band map matching,
   socat update, backfill). At the end of each month (month_weeks weeks): a
   job makes the monthly coadds from the weekly coadds, and a job runs sotrp
   on them.
5. Submit the tick of the next day.

A ledger (ledger.sqlite) records each step. A step that is complete is not
run again, so a failed tick can be submitted again.

    python scripts/mock_real_time/mockrt.py init --run-dir /path/to/run ...
    python scripts/mock_real_time/mockrt.py services --run-dir /path/to/run --submit
    python scripts/mock_real_time/mockrt.py start --run-dir /path/to/run
    python scripts/mock_real_time/mockrt.py status --run-dir /path/to/run

See docs/mock_real_time.md.
"""

import argparse
import importlib.util
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import feed  # noqa: E402
from configs import (  # noqa: E402
    RunConfig,
    coadd_analysis_config,
    daily_config,
    env_text,
    iso,
    services_script,
    slurm_script,
)

# --- ledger --------------------------------------------------------------------


class Ledger:
    """The status of each step of the run (SQLite)."""

    def __init__(self, path: Path):
        self.path = path
        with self._connect() as con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS steps (key TEXT PRIMARY KEY, "
                "status TEXT, updated TEXT, detail TEXT)"
            )

    def _connect(self):
        return sqlite3.connect(self.path, timeout=60)

    def get(self, key: str) -> str | None:
        with self._connect() as con:
            row = con.execute(
                "SELECT status FROM steps WHERE key = ?", (key,)
            ).fetchone()
        return row[0] if row else None

    def done(self, key: str) -> bool:
        return self.get(key) == "done"

    def set(self, key: str, status: str, detail: str = "") -> None:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._connect() as con:
            con.execute(
                "INSERT INTO steps VALUES (?, ?, ?, ?) ON CONFLICT(key) DO UPDATE "
                "SET status = excluded.status, updated = excluded.updated, "
                "detail = excluded.detail",
                (key, status, now, detail),
            )

    def rows(self) -> list[tuple]:
        with self._connect() as con:
            return con.execute("SELECT * FROM steps ORDER BY updated").fetchall()


def step(ledger: Ledger, key: str, func, *args, **kwargs):
    """Run func once: skip it if the ledger has the key as done."""
    if ledger.done(key):
        print(f"[{key}] already done")
        return None
    ledger.set(key, "running")
    try:
        detail = func(*args, **kwargs)
    except BaseException as err:
        ledger.set(key, "failed", repr(err)[:500])
        raise
    ledger.set(key, "done", json.dumps(detail, default=str)[:2000] if detail else "")
    print(f"[{key}] done")
    return detail


# --- helpers -------------------------------------------------------------------


def _load_script(code_dir: str, relative: str, name: str):
    """Import a script of the code directory (scripts/ is not a package)."""
    path = Path(code_dir) / relative
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_sotrp(
    run: RunConfig, config: dict, config_file: Path, log_file: Path, env=None
):
    config_file.parent.mkdir(parents=True, exist_ok=True)
    # The pickle and json outputs do not make their directories.
    for output in config.get("source_outputs", []) + config.get(
        "map_match_outputs", []
    ):
        if "directory" in output:
            Path(output["directory"]).mkdir(parents=True, exist_ok=True)
    config_file.write_text(json.dumps(config, indent=2))
    with open(log_file, "w") as log:
        result = subprocess.run(
            [
                run.python,
                "-c",
                "from sotrplib.cli import main; main()",
                "-c",
                str(config_file),
            ],
            cwd=run.code_dir,
            env={**os.environ, **(env or {})},
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if result.returncode != 0:
        raise RuntimeError(f"sotrp failed ({result.returncode}), see {log_file}")


def sbatch(
    script: Path, dependency: list[str] | None = None, begin: str | None = None
) -> str:
    cmd = ["sbatch", "--parsable"]
    if dependency:
        cmd.append("--dependency=afterok:" + ":".join(dependency))
    if begin:
        cmd.append(f"--begin={begin}")
    cmd.append(str(script))
    job_id = subprocess.run(
        cmd, check=True, capture_output=True, text=True
    ).stdout.strip()
    print(f"submitted {script.name}: {job_id}")
    return job_id.split(";")[0]


def write_slurm(
    run: RunConfig, name: str, command: str, kind: str = "tick", prefect=False
) -> Path:
    cpus, mem, time = {
        "tick": (run.tick_cpus, run.tick_mem, run.tick_time),
        "coadd": (run.coadd_cpus, run.coadd_mem, run.coadd_time),
        "small": (1, "4G", "00:30:00"),
    }[kind]
    path = run.slurm_dir / f"{name}.slurm"
    path.write_text(slurm_script(run, name, command, cpus, mem, time, prefect=prefect))
    return path


def mockrt_command(run: RunConfig, *args: str) -> str:
    script = Path(run.code_dir) / "scripts" / "mock_real_time" / "mockrt.py"
    return " ".join([run.python, str(script), *args, "--run-dir", str(run.root)])


# --- steps -----------------------------------------------------------------------


def do_feed(run: RunConfig, day: int) -> dict:
    start, end = run.day_window(day)
    new = feed.feed_window(
        Path(run.source_mapcat),
        run.live_mapcat,
        start,
        end,
        bands=run.bands,
        tubes=run.tubes,
        exclude=feed.read_exclude_list(run.exclude_maps_csv),
    )
    n_coverage = feed.update_sky_coverage() if new else 0
    return {"n_maps": len(new), "n_coverage_rows": n_coverage}


def do_daily(run: RunConfig, day: int) -> dict:
    out = run.day_dir(day)
    start, end = run.day_window(day)
    n_reset = feed.reset_unfinished_status(run.live_mapcat, start, end)
    if n_reset:
        print(f"day {day}: reset {n_reset} unfinished map status rows of a failed run")
    run_sotrp(run, daily_config(run, day), out / "config.json", out / "sotrp.log")
    return {"dir": str(out), "n_reset": n_reset}


def added_sources(record_dir: Path) -> list[str]:
    """
    The socat ids that the socat updates in record_dir added, or set to
    monitored. Reads all the records (a rerun writes a new record, in which
    the sources of the first run are "already_in_catalog").
    """
    ids = []
    for path in sorted(record_dir.glob("socat_update_*.json")):
        for group in json.loads(path.read_text())["groups"]:
            if (
                group["action"] in ("added", "monitored")
                and group["socat_id"] not in ids
            ):
                ids.append(group["socat_id"])
    return ids


def do_backfill(
    run: RunConfig, ledger: Ledger, record_dir: Path, out_dir: Path, end: datetime
) -> dict:
    """
    Measure each source that the socat update added in all the live maps
    that start before `end` (historical lightcurve extractor, lightcurvedb
    output).

    lightcurvedb keeps each measurement that it gets, also a duplicate. Thus,
    the ledger records each source that is done (key "backfill:<socat_id>"),
    and a rerun of the step does not measure the source again.
    """
    socat_ids = [
        socat_id
        for socat_id in added_sources(record_dir)
        if not ledger.done(f"backfill:{socat_id}")
    ]
    if not socat_ids:
        return {"n_sources": 0}
    extractor = _load_script(
        run.code_dir,
        "scripts/historical_lightcurve_extractor/slurm_wrapper_mapcat_historical_extractor.py",
        "historical_extractor",
    )
    for socat_id in socat_ids:
        step(
            ledger, f"backfill:{socat_id}", backfill_source, run, extractor,
            socat_id, out_dir / socat_id, end,
        )  # fmt: skip
    return {"n_sources": len(socat_ids), "socat_ids": socat_ids}


def backfill_source(
    run: RunConfig, extractor, socat_id: str, out_dir: Path, end: datetime
) -> None:
    """
    Measure one socat source in the live maps that start before `end`. Each
    source has its own out_dir, thus its own extractor socat database with
    only this source in it.
    """
    argv = [
        "--mapcat", str(run.live_mapcat),
        "--copy-from-socat-db", str(run.live_socat),
        "--socat-id", socat_id,
        "--end-time", iso(end).rstrip("Z"),
        "--bands", *run.bands,
        "--optics-tubes", *run.tubes,
        "--out-dir", str(out_dir),
        "--code-dir", run.code_dir,
        "--beam-profile", run.beam_profile,
    ]  # fmt: skip
    if run.lightcurvedb:
        argv.append("--lightcurvedb")
    slurm_files = extractor.main(argv)
    env = {
        "socat_model_database_name": str(out_dir / "extractor_socat.db"),
        "sotrp_runner": "basic",
    }
    for slurm_file in slurm_files:
        config_file = Path(str(slurm_file).replace("_sub.slurm", "_config.json"))
        config = json.loads(config_file.read_text())
        run_sotrp(run, config, config_file, config_file.with_suffix(".log"), env=env)


def submit_week(run: RunConfig, week: int) -> dict:
    """Submit the weekly coadd job of each band, and the analysis job after
    them. At the end of a month, also the month job."""
    weeks_mod = _load_script(
        run.code_dir, "scripts/coadding/submit_week_coadds.py", "submit_week_coadds"
    )
    start, end = run.week_window(week)
    out_dir = run.coadd_parent / "weekly" / f"{start:%Y%m%d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    config_dir = run.week_dir(week) / "coadd_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    job_ids = []
    for band in run.bands:
        config = weeks_mod.build_config(
            start_time=start.timestamp(),
            end_time=end.timestamp(),
            frequency=band,
            array=None,
            instrument="LAT",
            output_dir=out_dir,
            beam1d=run.beam_profile.format(band=band),
            fields=["rho", "kappa", "flux", "snr", "hits", "time_mean"],
            rerun=False,
            coadd_name=f"{band}_{int(start.timestamp())}_{int(end.timestamp())}_weekly",
            coadd_type="weekly",
            register=True,
            use_socat=True,
            ephem_file_path=None,
            asteroid_mask_radius="10 arcmin",
        )
        config_file = config_dir / f"{band}.json"
        config_file.write_text(json.dumps(config, indent=2))
        name = f"mockrt_week{week:02d}_{band}"
        command = (
            f'{run.python} -c "from sotrplib.coadd_cli import main; main()" '
            f"-c {config_file} > {config_dir / band}.log 2>&1"
        )
        job_ids.append(sbatch(write_slurm(run, name, command, kind="coadd")))

    analysis = write_slurm(
        run,
        f"mockrt_week{week:02d}_analysis",
        mockrt_command(run, "coadd-analysis", "--kind", "week", "--index", str(week)),
        kind="tick",
        prefect=run.runner == "prefect",
    )
    detail = {
        "coadd_jobs": job_ids,
        "analysis_job": sbatch(analysis, dependency=job_ids),
    }

    month = run.ends_month(week)
    if month is not None:
        planner = write_slurm(
            run,
            f"mockrt_month{month:02d}_plan",
            mockrt_command(run, "month", "--month", str(month)),
            kind="small",
        )
        detail["month_job"] = sbatch(planner, dependency=job_ids)
    return detail


def submit_month(run: RunConfig, month: int) -> dict:
    """Submit the monthly coadd job of each band (a coadd of the weekly
    coadds), and the analysis job after them."""
    months_mod = _load_script(
        run.code_dir, "scripts/coadding/submit_month_coadds.py", "submit_month_coadds"
    )
    start, end = run.month_window(month)
    out_dir = run.coadd_parent / "monthly" / f"{start:%Y%m%d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    config_dir = run.month_dir(month) / "coadd_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    job_ids = []
    for band in run.bands:
        have, make = months_mod.plan_weeks(
            run.live_mapcat,
            band,
            "weekly",
            start.timestamp(),
            end.timestamp(),
            run.week_days,
        )
        if make:
            raise RuntimeError(
                f"{band}: {len(make)} week(s) of month {month} have maps but no "
                "registered weekly coadd. Check the weekly coadd jobs."
            )
        if not have:
            print(f"{band}: no weekly coadds in month {month}, no monthly coadd.")
            continue
        config = months_mod.build_month_config(
            start_time=start.timestamp(),
            end_time=end.timestamp(),
            frequency=band,
            instrument="LAT",
            output_dir=out_dir,
            fields=["rho", "kappa", "flux", "snr", "hits", "time_mean"],
            coadd_name=f"{band}_{int(start.timestamp())}_{int(end.timestamp())}_monthly",
            coadd_type="monthly",
            week_coadd_type="weekly",
            register=True,
        )
        config_file = config_dir / f"{band}.json"
        config_file.write_text(json.dumps(config, indent=2))
        command = (
            f'{run.python} -c "from sotrplib.coadd_cli import main; main()" '
            f"-c {config_file} > {config_dir / band}.log 2>&1"
        )
        job_ids.append(
            sbatch(
                write_slurm(
                    run, f"mockrt_month{month:02d}_{band}", command, kind="coadd"
                )
            )
        )

    analysis = write_slurm(
        run,
        f"mockrt_month{month:02d}_analysis",
        mockrt_command(run, "coadd-analysis", "--kind", "month", "--index", str(month)),
        kind="tick",
        prefect=run.runner == "prefect",
    )
    return {
        "coadd_jobs": job_ids,
        "analysis_job": sbatch(analysis, dependency=job_ids or None),
    }


def submit_tick(run: RunConfig, day: int) -> str:
    script = write_slurm(
        run,
        f"mockrt_tick{day:03d}",
        mockrt_command(run, "tick", "--day", str(day)),
        kind="tick",
        prefect=run.runner == "prefect",
    )
    begin = f"now+{int(run.pace_minutes * 60)}" if run.pace_minutes > 0 else None
    return sbatch(script, begin=begin)


# --- commands --------------------------------------------------------------------


def cmd_init(args) -> None:
    root = Path(args.run_dir).resolve()
    if (root / "mockrt.json").exists() and not args.force:
        sys.exit(f"{root}/mockrt.json exists. Use --force to write it again.")
    root.mkdir(parents=True, exist_ok=True)
    run = RunConfig(
        run_dir=str(root),
        source_mapcat=str(Path(args.source_mapcat).resolve()),
        source_socat=str(Path(args.source_socat).resolve()),
        depth_one_parent=args.depth_one_parent,
        code_dir=str(Path(args.code_dir).resolve()),
        start_time=args.start_time,
        n_days=args.n_days,
        venv=args.venv or "",
        lightserve_venv=args.lightserve_venv or "",
        exclude_maps_csv=args.exclude_maps_csv,
        runner=args.runner,
        pace_minutes=args.pace_minutes,
        lightcurvedb=not args.no_lightcurvedb,
        postgres_image=args.postgres_image or "",
    )
    for d in (
        run.slurm_dir,
        run.coadd_parent,
        root / "days",
        root / "weeks",
        root / "months",
    ):
        d.mkdir(parents=True, exist_ok=True)
    feed.create_live_mapcat(Path(run.source_mapcat), run.live_mapcat)
    if not run.live_socat.exists():
        shutil.copy2(run.source_socat, run.live_socat)
    run.env_file.write_text(env_text(run))
    Ledger(run.ledger)
    print("Run settings:", run.save())
    print(f"Days 0..{run.n_days - 1} from {run.start_time}")


def cmd_services(args) -> None:
    run = RunConfig.load(args.run_dir)
    if not run.postgres_image:
        sys.exit(
            "Set postgres_image in mockrt.json (apptainer pull postgres_16.sif docker://postgres:16)."
        )
    path = run.slurm_dir / "services.slurm"
    path.write_text(services_script(run))
    print("Services job:", path)
    if args.submit:
        sbatch(path)


def cmd_start(args) -> None:
    run = RunConfig.load(args.run_dir)
    if run.lightcurvedb and not run.services_env.exists():
        sys.exit(
            "services.env does not exist: start the services job first (mockrt.py services --submit)."
        )
    submit_tick(run, args.day)


def cmd_tick(args) -> None:
    run = RunConfig.load(args.run_dir)
    ledger = Ledger(run.ledger)
    day = args.day
    _, end = run.day_window(day)
    step(ledger, f"day{day:03d}:feed", do_feed, run, day)
    step(ledger, f"day{day:03d}:daily", do_daily, run, day)
    out = run.day_dir(day)
    step(
        ledger, f"day{day:03d}:backfill", do_backfill, run, ledger,
        out / "socat_updates", out / "backfill", end,
    )  # fmt: skip
    week = run.ends_week(day)
    if week is not None and not args.no_coadds:
        step(ledger, f"week{week:02d}:submit", submit_week, run, week)
    if not args.no_next and day + 1 < run.n_days:
        step(ledger, f"day{day + 1:03d}:submit", submit_tick, run, day + 1)


def cmd_feed(args) -> None:
    """Feed days without running the pipeline on them (for example, to start
    a test in the middle of the data)."""
    run = RunConfig.load(args.run_dir)
    ledger = Ledger(run.ledger)
    for day in range(args.first, args.last + 1):
        step(ledger, f"day{day:03d}:feed", do_feed, run, day)


def cmd_coadd_analysis(args) -> None:
    run = RunConfig.load(args.run_dir)
    ledger = Ledger(run.ledger)
    if args.kind == "week":
        start, end = run.week_window(args.index)
        out, coadd_type, key = (
            run.week_dir(args.index),
            "weekly",
            f"week{args.index:02d}",
        )
    else:
        start, end = run.month_window(args.index)
        out, coadd_type, key = (
            run.month_dir(args.index),
            "monthly",
            f"month{args.index:02d}",
        )
    config = coadd_analysis_config(run, coadd_type, start, end, out)
    step(
        ledger,
        f"{key}:analysis",
        run_sotrp,
        run,
        config,
        out / "config.json",
        out / "sotrp.log",
    )
    step(
        ledger, f"{key}:backfill", do_backfill, run, ledger,
        out / "socat_updates", out / "backfill", end,
    )  # fmt: skip


def cmd_week(args) -> None:
    run = RunConfig.load(args.run_dir)
    step(Ledger(run.ledger), f"week{args.week:02d}:submit", submit_week, run, args.week)


def cmd_month(args) -> None:
    run = RunConfig.load(args.run_dir)
    step(
        Ledger(run.ledger),
        f"month{args.month:02d}:submit",
        submit_month,
        run,
        args.month,
    )


def cmd_status(args) -> None:
    run = RunConfig.load(args.run_dir)
    for key, status, updated, detail in Ledger(run.ledger).rows():
        print(f"{updated}  {key:<20} {status:<8} {detail[:100]}")
    con = sqlite3.connect(f"file:{run.live_mapcat}?mode=ro", uri=True)
    n_maps = con.execute("SELECT COUNT(*) FROM depth_one_maps").fetchone()[0]
    n_coadds = con.execute(
        "SELECT coadd_type, COUNT(*) FROM depth_one_coadds GROUP BY coadd_type"
    ).fetchall()
    con.close()
    print(f"live mapcat: {n_maps} depth-1 maps, coadds {dict(n_coadds)}")
    print("services:", "up" if run.services_env.exists() else "not running")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="Make the run directory and the live databases.")
    init.add_argument("--source-mapcat", required=True)
    init.add_argument("--source-socat", required=True)
    init.add_argument("--depth-one-parent", required=True)
    init.add_argument(
        "--code-dir", required=True, help="sotrplib checkout that the jobs run."
    )
    init.add_argument("--start-time", required=True, help="Start of day 0 (ISO, UTC).")
    init.add_argument("--n-days", type=int, required=True)
    init.add_argument("--venv", default=None)
    init.add_argument("--lightserve-venv", default=None)
    init.add_argument("--exclude-maps-csv", default=None)
    init.add_argument("--runner", choices=["prefect", "basic"], default="prefect")
    init.add_argument("--pace-minutes", type=float, default=0.0)
    init.add_argument("--postgres-image", default=None)
    init.add_argument("--no-lightcurvedb", action="store_true")
    init.add_argument("--force", action="store_true")

    services = sub.add_parser("services", help="Write (and submit) the services job.")
    services.add_argument("--submit", action="store_true")

    start = sub.add_parser("start", help="Submit the tick of a day (default: day 0).")
    start.add_argument("--day", type=int, default=0)

    tick = sub.add_parser("tick", help="Run one simulated day (in a job).")
    tick.add_argument("--day", type=int, required=True)
    tick.add_argument(
        "--no-next", action="store_true", help="Do not submit the next tick."
    )
    tick.add_argument(
        "--no-coadds",
        action="store_true",
        help="Do not submit the week jobs (for tests).",
    )

    feed_days = sub.add_parser("feed", help="Feed days only (no pipeline run).")
    feed_days.add_argument("--first", type=int, required=True)
    feed_days.add_argument("--last", type=int, required=True)

    analysis = sub.add_parser(
        "coadd-analysis", help="Run sotrp on the coadds of a week or a month."
    )
    analysis.add_argument("--kind", choices=["week", "month"], required=True)
    analysis.add_argument("--index", type=int, required=True)

    week = sub.add_parser(
        "week", help="Submit the jobs of a week (normally from a tick)."
    )
    week.add_argument("--week", type=int, required=True)

    month = sub.add_parser(
        "month", help="Submit the jobs of a month (normally from a job)."
    )
    month.add_argument("--month", type=int, required=True)

    sub.add_parser("status", help="Show the ledger and the live mapcat.")

    for sp in (
        init,
        services,
        start,
        tick,
        feed_days,
        analysis,
        week,
        month,
        sub.choices["status"],
    ):
        sp.add_argument("--run-dir", required=True)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    {
        "init": cmd_init,
        "services": cmd_services,
        "start": cmd_start,
        "tick": cmd_tick,
        "feed": cmd_feed,
        "coadd-analysis": cmd_coadd_analysis,
        "week": cmd_week,
        "month": cmd_month,
        "status": cmd_status,
    }[args.command](args)


if __name__ == "__main__":
    main()
