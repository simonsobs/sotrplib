#!/usr/bin/env python3
"""
Write the sotrp-coadd configs and SLURM scripts for a monthly coadd of each
frequency. Add --submit to submit the jobs.

By default, a monthly coadd is a coadd of weekly coadds:

1. The script divides the month into week windows (--week-days, from
   --start-time). The last window can be shorter.
2. For each window, it looks in mapcat for a registered weekly coadd
   (--week-coadd-type) of the frequency that starts in the window. If there
   is no weekly coadd, and depth-1 maps start in the window, the script
   writes a job that makes the weekly coadd from the depth-1 maps.
3. It writes a job that makes the monthly coadd from the weekly coadds that
   start in the month. This job starts after the weekly coadd jobs.

The monthly coadd is registered in mapcat (--coadd-type) and linked to the
depth-1 maps of its weekly coadds. If two weekly coadds contain the same
depth-1 map, the monthly job stops before the coadd work.

With --no-week-coadd, the script writes one job for each frequency that
makes the monthly coadd directly from the depth-1 maps of the month. This
is slower: each depth-1 map is preprocessed in the job.

The script writes submit.sh, which submits the jobs with their
dependencies. --submit runs it.
"""

import argparse
import json
import sqlite3
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

from submit_week_coadds import (
    REPO_DIR,
    SLURM_HEADER,
    build_config,
    iso,
    window_bounds,
    window_dirname,
)


def parse_time(value: str) -> float:
    t = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t.timestamp()


def db_time(unix_time: float) -> str:
    """The format of the DateTime columns of mapcat (SQLite, UTC)."""
    return datetime.fromtimestamp(unix_time, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


# A depth-1 map that starts before the end of a window can stop after it.
# Thus, a weekly coadd of the window can stop up to this long after the window.
MAP_SLACK = timedelta(days=1).total_seconds()


def _unix(value) -> float:
    return (
        datetime.fromisoformat(str(value).split("+")[0])
        .replace(tzinfo=timezone.utc)
        .timestamp()
    )


def registered_weeks(
    database_name: Path,
    frequency: str,
    coadd_type: str,
    start: float,
    stop: float,
) -> list[tuple[str, float, float]]:
    """Return (name, start, stop) of the registered coadds of this type that
    start in [start, stop)."""
    con = sqlite3.connect(f"file:{database_name}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT coadd_name, start_time, stop_time FROM depth_one_coadds "
            "WHERE frequency = ? AND coadd_type = ? AND start_time >= ? "
            "AND start_time < ?",
            (frequency, coadd_type, db_time(start), db_time(stop)),
        ).fetchall()
    finally:
        con.close()
    return [(name, _unix(s), _unix(e)) for name, s, e in rows]


def n_depth_one_maps(
    database_name: Path, frequency: str, start: float, stop: float
) -> int:
    """Return the number of depth-1 maps of the frequency that start in
    [start, stop) (the left-bound time binning of the coadd jobs)."""
    con = sqlite3.connect(f"file:{database_name}?mode=ro", uri=True)
    try:
        (n,) = con.execute(
            "SELECT COUNT(*) FROM depth_one_maps WHERE frequency = ? "
            "AND start_time >= ? AND start_time < ?",
            (frequency, db_time(start), db_time(stop)),
        ).fetchone()
    finally:
        con.close()
    return n


def plan_weeks(
    database_name: Path,
    frequency: str,
    week_coadd_type: str,
    start: float,
    stop: float,
    week_days: float,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """
    Return (registered windows, windows to coadd). A window is registered if
    a weekly coadd starts in it and stops not more than MAP_SLACK after it.
    A window without a weekly coadd is to coadd if depth-1 maps start in it.

    Raises ValueError if a registered weekly coadd starts in a window but
    stops later. The monthly coadd would then contain maps of a different
    window (for example, maps after the end of the month).
    """
    registered = registered_weeks(
        database_name, frequency, week_coadd_type, start, stop
    )
    have, make = [], []
    for w_start, w_stop in window_bounds(start, stop, week_days):
        in_window = [r for r in registered if w_start <= r[1] < w_stop]
        too_long = [r for r in in_window if r[2] > w_stop + MAP_SLACK]
        if too_long:
            name, _, r_stop = too_long[0]
            raise ValueError(
                f"{frequency}: the registered {week_coadd_type} coadd {name} "
                f"starts in the window {iso(w_start)} to {iso(w_stop)} but stops "
                f"at {iso(r_stop)}. Its maps are not all in the window. Use week "
                "windows that align with the registered coadds (--start-time, "
                "--week-days), or use --no-week-coadd."
            )
        if in_window:
            have.append((w_start, w_stop))
        elif n_depth_one_maps(database_name, frequency, w_start, w_stop) > 0:
            make.append((w_start, w_stop))
    return have, make


def build_month_config(
    start_time: float,
    end_time: float,
    frequency: str,
    instrument: str,
    output_dir: Path,
    fields: list[str],
    coadd_name: str,
    coadd_type: str,
    week_coadd_type: str,
    register: bool,
) -> dict:
    """The sotrp-coadd config of a coadd of the weekly coadds in the month."""
    return {
        "instrument": instrument,
        "maps": {
            "map_generator_type": "mapcat_database",
            "map_type": "coadd_rhokappa",
            "coadd_type": week_coadd_type,
            "frequency": frequency,
            "instrument": instrument,
            "start_time": iso(start_time),
            "end_time": iso(end_time),
            "time_binning": "left-bound",
        },
        "preprocessors": [],
        "map_coadder": {
            "coadd_type": "rhokappa",
            "frequencies": [frequency],
            "instrument": instrument,
        },
        "map_outputs": [
            {"output_type": "maps", "directory": str(output_dir), "fields": fields}
        ],
        "mapcat_registration": {
            "coadd_name": coadd_name,
            "coadd_type": coadd_type,
            "enabled": register,
        },
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--database-name", type=Path, required=True, help="mapcat SQLite file."
    )
    p.add_argument(
        "--depth-one-parent",
        type=Path,
        required=True,
        help="MAPCAT_DEPTH_ONE_PARENT of the depth-1 map paths.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for the monthly coadd FITS files (in a YYYYMMDD subdirectory).",
    )
    p.add_argument(
        "--week-output-dir",
        type=Path,
        default=None,
        help="Directory for new weekly coadd FITS files. Default: <output-dir>/weeks.",
    )
    p.add_argument(
        "--coadd-parent",
        type=Path,
        default=None,
        help="MAPCAT_DEPTH_ONE_COADD_PARENT. Must contain --output-dir, "
        "--week-output-dir and the registered weekly coadds. Default: --output-dir.",
    )
    p.add_argument("--start-time", required=True, help="Start of the month (ISO, UTC).")
    p.add_argument(
        "--end-time",
        default=None,
        help="End of the month (ISO, UTC). Default: --start-time + --month-days.",
    )
    p.add_argument(
        "--month-days", type=float, default=30.0, help="Length of the month."
    )
    p.add_argument(
        "--week-days", type=float, default=7.0, help="Length of a week window."
    )
    p.add_argument("--frequencies", nargs="+", default=["f090", "f150", "f220", "f280"])
    p.add_argument(
        "--no-week-coadd",
        action="store_true",
        help="Make each monthly coadd directly from the depth-1 maps.",
    )
    p.add_argument(
        "--week-coadd-type",
        default="weekly",
        help="coadd_type of the weekly coadds (to find them and to register new ones).",
    )
    p.add_argument(
        "--coadd-type", default="monthly", help="coadd_type of the monthly coadds."
    )
    p.add_argument("--instrument", default="LAT")
    p.add_argument(
        "--fields",
        nargs="+",
        default=["rho", "kappa", "flux", "snr", "hits", "time_mean"],
    )
    p.add_argument(
        "--beam1d-template",
        default="profile_{frequency}_1756699200_20000000000.txt",
        help="Matched-filter beam profile (relative to --repo-dir) of each frequency, "
        "for the depth-1 coadd jobs.",
    )
    p.add_argument("--socat-db-path", type=Path, default=None)
    p.add_argument("--use-socat", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--ephem-file-path",
        default="sotrplib/solar_system/JPL_batched_ephemerides_2023-01-01_2033-01-01.parquet",
        help="Ephemeris parquet for the asteroid mask. '' to use SOCat only.",
    )
    p.add_argument("--asteroid-mask-radius", default="10 arcmin")
    p.add_argument("--repo-dir", type=Path, default=REPO_DIR)
    p.add_argument(
        "--register-coadds", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument(
        "--slurm-dir", type=Path, default=None, help="Default: <output-dir>/slurm."
    )
    p.add_argument("--account", default="simonsobs")
    p.add_argument("--cpus", type=int, default=4)
    p.add_argument("--mem-per-cpu", default="16G")
    p.add_argument(
        "--time",
        default="08:00:00",
        help="Time limit of a job that coadds depth-1 maps (a week, or a month "
        "with --no-week-coadd).",
    )
    p.add_argument(
        "--month-time",
        default="01:00:00",
        help="Time limit of a coadd of weekly coadds.",
    )
    p.add_argument("--submit", action="store_true", help="Run submit.sh.")
    return p.parse_args(argv)


def slurm_text(args, tag: str, config_file: Path, slurm_dir: Path, time: str) -> str:
    socat_env = ""
    if args.use_socat:
        socat_env = "export socat_client_client_type=db\n"
        if args.socat_db_path:
            socat_env += f"export socat_model_database_name={args.socat_db_path}\n"
    return SLURM_HEADER.format(
        jobname=tag,
        account=args.account,
        cpus=args.cpus,
        mem_per_cpu=args.mem_per_cpu,
        time=time,
        slurm_out_dir=slurm_dir,
        repo_dir=args.repo_dir,
        depth_one_parent=args.depth_one_parent,
        coadd_parent=args.coadd_parent or args.output_dir,
        database_name=args.database_name,
        socat_env=socat_env,
        config_file=config_file,
        log_file=slurm_dir / f"{tag}.log",
    )


def depth_one_config(args, frequency, w_start, w_stop, output_dir, coadd_type) -> dict:
    beam1d = None
    if args.beam1d_template:
        name = args.beam1d_template.format(frequency=frequency)
        if (args.repo_dir / name).exists():
            beam1d = name
    return build_config(
        start_time=w_start,
        end_time=w_stop,
        frequency=frequency,
        array=None,
        instrument=args.instrument,
        output_dir=output_dir,
        beam1d=beam1d,
        fields=args.fields,
        rerun=False,
        coadd_name=f"{frequency}_{int(w_start)}_{int(w_stop)}",
        coadd_type=coadd_type,
        register=args.register_coadds,
        use_socat=args.use_socat,
        ephem_file_path=args.ephem_file_path or None,
        asteroid_mask_radius=args.asteroid_mask_radius,
    )


def main(argv=None) -> Path:
    args = parse_args(argv)
    start = parse_time(args.start_time)
    stop = (
        parse_time(args.end_time)
        if args.end_time
        else start + timedelta(days=args.month_days).total_seconds()
    )
    week_output_dir = args.week_output_dir or args.output_dir / "weeks"
    slurm_dir = args.slurm_dir or args.output_dir / "slurm"
    month_dir = args.output_dir / window_dirname(start)
    for d in (args.output_dir, slurm_dir, month_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"Month {iso(start)} to {iso(stop)}")
    lines = ["#!/bin/bash", "set -e", f"cd {slurm_dir}"]

    for frequency in args.frequencies:
        month_tag = f"month_{window_dirname(start)}_{frequency}"
        if args.no_week_coadd:
            config = depth_one_config(
                args, frequency, start, stop, month_dir, args.coadd_type
            )
            config_file = slurm_dir / f"{month_tag}.json"
            config_file.write_text(json.dumps(config, indent=2))
            (slurm_dir / f"{month_tag}.slurm").write_text(
                slurm_text(args, month_tag, config_file, slurm_dir, args.time)
            )
            lines.append(f"sbatch {month_tag}.slurm")
            print(f"{frequency}: monthly coadd from depth-1 maps")
            continue

        try:
            have, make = plan_weeks(
                args.database_name,
                frequency,
                args.week_coadd_type,
                start,
                stop,
                args.week_days,
            )
        except ValueError as err:
            raise SystemExit(f"ERROR: {err}") from None
        if not have and not make:
            print(f"{frequency}: no weekly coadds and no depth-1 maps, no job.")
            continue

        deps = f"deps_{frequency}"
        lines.append(f'{deps}=""')
        for w_start, w_stop in make:
            week_tag = f"week_{window_dirname(w_start)}_{frequency}"
            week_dir = week_output_dir / window_dirname(w_start)
            week_dir.mkdir(parents=True, exist_ok=True)
            config = depth_one_config(
                args, frequency, w_start, w_stop, week_dir, args.week_coadd_type
            )
            config_file = slurm_dir / f"{week_tag}.json"
            config_file.write_text(json.dumps(config, indent=2))
            (slurm_dir / f"{week_tag}.slurm").write_text(
                slurm_text(args, week_tag, config_file, slurm_dir, args.time)
            )
            lines.append(f'{deps}="${{{deps}}}:$(sbatch --parsable {week_tag}.slurm)"')

        config = build_month_config(
            start_time=start,
            end_time=stop,
            frequency=frequency,
            instrument=args.instrument,
            output_dir=month_dir,
            fields=args.fields,
            coadd_name=f"{frequency}_{int(start)}_{int(stop)}_{args.coadd_type}",
            coadd_type=args.coadd_type,
            week_coadd_type=args.week_coadd_type,
            register=args.register_coadds,
        )
        config_file = slurm_dir / f"{month_tag}.json"
        config_file.write_text(json.dumps(config, indent=2))
        (slurm_dir / f"{month_tag}.slurm").write_text(
            slurm_text(args, month_tag, config_file, slurm_dir, args.month_time)
        )
        lines.append(
            f"sbatch ${{{deps}:+--dependency=afterok${{{deps}}}}} {month_tag}.slurm"
        )
        print(
            f"{frequency}: {len(have)} registered weekly coadd(s), "
            f"{len(make)} new weekly coadd job(s), 1 monthly coadd job"
        )

    submit = slurm_dir / "submit.sh"
    submit.write_text("\n".join(lines) + "\n")
    submit.chmod(0o755)
    print("Configs and SLURM scripts:", slurm_dir)
    print("Submit with:", submit)
    if args.submit:
        subprocess.run(["bash", str(submit)], check=True)
    return submit


if __name__ == "__main__":
    main()
