#!/usr/bin/env python3
"""
Write one sotrp-coadd config and one SLURM script for each time window
(default: 7 days) and each frequency. Add --submit to submit the jobs.

Each job makes a coadd of the depth-1 maps in its window. The job writes
the coadd to FITS and registers it in mapcat (default). See
sotrplib.maps.map_coadding.stream_coadd and docs/coadding/scripts.md.
"""

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from getpass import getuser
from pathlib import Path

USER = getuser()

REPO_DIR = Path(__file__).resolve().parents[2]


def get_time_range(database_name: Path) -> tuple[float, float]:
    """Return the first start_time and the last stop_time of the depth-1 maps."""
    con = sqlite3.connect(str(database_name))
    try:
        cur = con.cursor()
        cur.execute("SELECT MIN(start_time), MAX(stop_time) FROM depth_one_maps")
        start, stop = cur.fetchone()
    finally:
        con.close()
    if start is None or stop is None:
        raise ValueError(f"No depth_one_maps rows found in {database_name}")
    # SQLite gives DateTime strings. Change them to unix times.
    return (
        datetime.fromisoformat(start).replace(tzinfo=timezone.utc).timestamp(),
        datetime.fromisoformat(stop).replace(tzinfo=timezone.utc).timestamp(),
    )


def iso(unix_time: float) -> str:
    return datetime.fromtimestamp(unix_time, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def window_dirname(unix_time: float) -> str:
    """Return the subdirectory name for a window: its UTC start date (YYYYMMDD)."""
    return datetime.fromtimestamp(unix_time, tz=timezone.utc).strftime("%Y%m%d")


def window_bounds(
    start: float, stop: float, window_days: float
) -> list[tuple[float, float]]:
    """Return windows of `window_days` days from `start` to `stop`."""
    window = window_days * 86400.0
    windows = []
    t = start
    while t < stop:
        windows.append((t, min(t + window, stop)))
        t += window
    return windows


def build_config(
    start_time: float,
    end_time: float,
    frequency: str,
    array: str | None,
    instrument: str,
    output_dir: Path,
    beam1d: Path | None,
    fields: list[str],
    rerun: bool,
    coadd_name: str,
    coadd_type: str,
    register: bool,
    use_socat: bool,
    ephem_file_path: Path | None,
    asteroid_mask_radius: str,
) -> dict:
    matched_filter: dict = {
        "preprocessor_type": "matched_filter",
        "band_height": "1 deg",
        "shrink_holes": "5 arcmin",
        "noisemask_lim": 0.1,
        "noisemask_radius": "10 arcmin",
        "apod_holes": "10 arcmin",
    }
    if beam1d is not None:
        matched_filter["beam1d"] = str(beam1d)

    preprocessors = [
        {"preprocessor_type": "planet_mask", "mask_radius": "15 arcmin"},
    ]
    if use_socat or ephem_file_path is not None:
        asteroid_mask: dict = {
            "preprocessor_type": "asteroid_mask",
            "use_socat": use_socat,
            "mask_radius": asteroid_mask_radius,
        }
        if ephem_file_path is not None:
            # interpolate_ephem needs ephemeris rows up to 0.5 day before
            # and after each time. Thus, load 1 day more at each end.
            pad = 1.0 * 86400.0
            asteroid_mask["ephem_file_path"] = str(ephem_file_path)
            asteroid_mask["start_time"] = iso(start_time - pad)
            asteroid_mask["end_time"] = iso(end_time + pad)
        preprocessors.append(asteroid_mask)
    preprocessors.extend(
        [
            matched_filter,
            {"preprocessor_type": "kappa_rho"},
            {
                "preprocessor_type": "edge_mask",
                "mask_on": "kappa",
                "edge_width": "10.0 arcmin",
            },
        ]
    )

    return {
        "instrument": instrument,
        "maps": {
            "map_generator_type": "mapcat_database",
            "map_type": "intensity",
            "frequency": frequency,
            "array": array,
            "instrument": instrument,
            "start_time": iso(start_time),
            "end_time": iso(end_time),
            "rerun": rerun,
            "time_binning": "left-bound",
        },
        "preprocessors": preprocessors,
        "map_coadder": {
            "coadd_type": "rhokappa",
            "frequencies": [frequency],
            "instrument": instrument,
        },
        "map_outputs": [
            {
                "output_type": "maps",
                "directory": str(output_dir),
                "fields": fields,
            }
        ],
        "mapcat_registration": {
            "coadd_name": coadd_name,
            "coadd_type": coadd_type,
            "enabled": register,
        },
    }


SLURM_HEADER = """#!/bin/bash
#SBATCH --job-name={jobname}
#SBATCH -A {account}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem-per-cpu={mem_per_cpu}
#SBATCH --time={time}
#SBATCH --output={slurm_out_dir}/%x.out

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
cd {repo_dir}
source .venv/bin/activate

export MAPCAT_DEPTH_ONE_PARENT={depth_one_parent}
export MAPCAT_DEPTH_ONE_COADD_PARENT={coadd_parent}
export MAPCAT_DATABASE_NAME={database_name}
{socat_env}
srun --overlap sotrp-coadd -c {config_file} > {log_file} 2>&1
"""


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument(
        "--database-name",
        type=Path,
        required=True,
        help="Path to the mapcat sqlite database, e.g. .../out_deep56/mapcat.sqlite",
    )
    p.add_argument(
        "--depth-one-parent",
        type=Path,
        required=True,
        help="Parent directory that depth-1 map paths in the database are relative to.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write coadded map FITS outputs to. Each window's coadds go in "
        "a YYYYMMDD subdirectory named for the window's UTC start date.",
    )
    p.add_argument(
        "--coadd-parent",
        type=Path,
        default=None,
        help="Parent directory that registered coadd paths are stored relative to "
        "(MAPCAT_DEPTH_ONE_COADD_PARENT). Must contain --output-dir. Default: --output-dir.",
    )
    p.add_argument(
        "--window-days",
        type=float,
        default=7.0,
        help="Width of each coadd window in days, anchored to the first observation.",
    )
    p.add_argument(
        "--start-time",
        type=str,
        default=None,
        help="ISO8601 start time override. Default: earliest observation in the database.",
    )
    p.add_argument(
        "--end-time",
        type=str,
        default=None,
        help="ISO8601 end time override. Default: latest observation in the database.",
    )
    p.add_argument(
        "--frequencies",
        nargs="+",
        default=["f090", "f150", "f220", "f280"],
        help="Frequency bands to coadd, one job per (window, frequency).",
    )
    p.add_argument(
        "--array",
        type=str,
        default=None,
        help="Use the maps of this array only. Default: all arrays, "
        "in one coadd for each frequency. Mapcat stores no array for a coadd.",
    )
    p.add_argument(
        "--instrument",
        type=str,
        default="LAT",
        help="Instrument tag stored on the coadded maps.",
    )
    p.add_argument(
        "--fields",
        nargs="+",
        default=["rho", "kappa", "flux", "snr", "hits", "time_mean"],
        help="Coadd fields to write to FITS. You can request rho/kappa "
        "and flux/snr together.",
    )
    p.add_argument(
        "--beam1d-template",
        type=str,
        default="profile_{frequency}_1756699200_20000000000.txt",
        help="Path template (relative to --repo-dir) for the matched-filter 1D beam profile "
        "per frequency. Only used if the resolved file exists; set to '' to disable.",
    )
    p.add_argument(
        "--socat-db-path",
        type=Path,
        default=None,
        help="Path to a SOCat sqlite database for the asteroid mask. "
        "If not given, the job uses the SOCat settings in your environment, if they exist.",
    )
    p.add_argument(
        "--use-socat",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use SOCat for asteroid positions. If SOCat is not "
        "configured or a query fails, use --ephem-file-path.",
    )
    p.add_argument(
        "--ephem-file-path",
        type=str,
        default="sotrplib/solar_system/JPL_batched_ephemerides_2023-01-01_2033-01-01.parquet",
        help="JPL Horizons ephemeris parquet file (absolute, or relative to "
        "--repo-dir). The asteroid mask uses it if SOCat is not available. "
        "Set to '' to use SOCat only.",
    )
    p.add_argument(
        "--asteroid-mask-radius",
        type=str,
        default="10 arcmin",
        help="Radius to mask around each detected asteroid position.",
    )
    p.add_argument(
        "--repo-dir",
        type=Path,
        default=REPO_DIR,
        help="sotrplib checkout to cd into and activate .venv in for each job.",
    )
    p.add_argument(
        "--coadd-type",
        type=str,
        default="depth1_streaming_coadd",
        help="coadd_type tag stored in mapcat's depth_one_coadds table.",
    )
    p.add_argument(
        "--register-coadds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Register finished coadds (and links to their constituent maps) in mapcat.",
    )
    p.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Directory to write generated config JSONs to. Default: <output-dir>/configs.",
    )
    p.add_argument(
        "--slurm-dir",
        type=Path,
        default=None,
        help="Directory to write generated SLURM scripts + logs to. Default: <output-dir>/slurm.",
    )
    p.add_argument(
        "--account", type=str, default="simonsobs", help="SLURM account (-A)."
    )
    p.add_argument("--cpus", type=int, default=4, help="CPUs per job.")
    p.add_argument("--mem-per-cpu", type=str, default="16G", help="Memory per CPU.")
    p.add_argument("--time", type=str, default="04:00:00", help="SLURM time limit.")
    p.add_argument(
        "--rerun",
        action="store_true",
        help="Process maps that have the status completed again. This "
        "option has an effect only if maps.track_processing is true.",
    )
    p.add_argument(
        "--submit",
        action="store_true",
        help="Submit the SLURM scripts with sbatch. Without this option, "
        "the script only writes the files.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    output_dir = args.output_dir
    config_dir = args.config_dir or (output_dir / "configs")
    slurm_dir = args.slurm_dir or (output_dir / "slurm")
    for d in (output_dir, config_dir, slurm_dir):
        d.mkdir(parents=True, exist_ok=True)

    if args.start_time and args.end_time:
        start = (
            datetime.strptime(args.start_time, "%Y-%m-%dT%H:%M:%SZ")
            .replace(tzinfo=timezone.utc)
            .timestamp()
        )
        stop = (
            datetime.strptime(args.end_time, "%Y-%m-%dT%H:%M:%SZ")
            .replace(tzinfo=timezone.utc)
            .timestamp()
        )
    else:
        start, stop = get_time_range(args.database_name)

    windows = window_bounds(start, stop, args.window_days)
    print(f"{len(windows)} window(s) from {iso(start)} to {iso(stop)}")

    submitted = []
    for w, (w_start, w_stop) in enumerate(windows):
        for frequency in args.frequencies:
            tag = f"week{w:02d}_{frequency}"
            window_output_dir = output_dir / window_dirname(w_start)
            window_output_dir.mkdir(exist_ok=True)

            beam1d = None
            if args.beam1d_template:
                candidate = args.repo_dir / args.beam1d_template.format(
                    frequency=frequency
                )
                if candidate.exists():
                    beam1d = args.beam1d_template.format(frequency=frequency)

            config = build_config(
                start_time=w_start,
                end_time=w_stop,
                frequency=frequency,
                array=args.array,
                instrument=args.instrument,
                output_dir=window_output_dir,
                beam1d=beam1d,
                fields=args.fields,
                rerun=args.rerun,
                coadd_name=f"{frequency}_{int(w_start)}_{int(w_stop)}",
                coadd_type=args.coadd_type,
                register=args.register_coadds,
                use_socat=args.use_socat,
                ephem_file_path=args.ephem_file_path if args.ephem_file_path else None,
                asteroid_mask_radius=args.asteroid_mask_radius,
            )
            config_file = config_dir / f"{tag}.json"
            config_file.write_text(json.dumps(config, indent=2))

            socat_env = ""
            if args.use_socat:
                socat_env = "export socat_client_client_type=db\n"
                if args.socat_db_path:
                    socat_env += (
                        f"export socat_model_database_name={args.socat_db_path}\n"
                    )

            log_file = slurm_dir / f"{tag}.log"
            slurm_text = SLURM_HEADER.format(
                jobname=tag,
                account=args.account,
                cpus=args.cpus,
                mem_per_cpu=args.mem_per_cpu,
                time=args.time,
                slurm_out_dir=slurm_dir,
                repo_dir=args.repo_dir,
                depth_one_parent=args.depth_one_parent,
                coadd_parent=args.coadd_parent or output_dir,
                database_name=args.database_name,
                socat_env=socat_env,
                config_file=config_file,
                log_file=log_file,
            )
            slurm_file = slurm_dir / f"{tag}.slurm"
            slurm_file.write_text(slurm_text)
            submitted.append(slurm_file)

    print(f"Wrote {len(submitted)} config(s) to {config_dir}")
    print(f"Wrote {len(submitted)} SLURM script(s) to {slurm_dir}")

    if args.submit:
        import subprocess

        for slurm_file in submitted:
            subprocess.run(["sbatch", str(slurm_file)], check=True, cwd=args.repo_dir)
        print(f"Submitted {len(submitted)} job(s).")
    else:
        print("Dry run (pass --submit to sbatch these scripts).")


if __name__ == "__main__":
    main()
