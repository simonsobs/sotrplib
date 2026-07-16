#!/usr/bin/env python3
"""
Submit one SLURM job per time window (default: 7 days) that streams depth-1
intensity/inverse-variance maps for that window from a mapcat database
through sotrp-coadd: each map is built, preprocessed (matched filter etc.)
and merged into a running coadd one at a time, discarded, then the next map
is loaded -- so RAM usage stays bounded regardless of how many maps go into
one coadd. Per-map preprocessing (rather than coadding raw maps and
filtering once) matters here because moving sources smear across pixels if
many days are summed before any per-observation handling, and matched
filtering needs each observation's own noise properties. See
sotrplib.maps.streaming_coadd for details.

The finished coadd is saved to FITS and, by default, registered (along with
links to every depth-1 map that went into it) in mapcat's
depth_one_coadds / link_depth_one_map_to_coadd tables.

Generates a sotrp-coadd config JSON + a SLURM script per (window,
frequency), and by default only writes them; pass --submit to actually
sbatch them.
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
    """Min start_time / max stop_time over all depth-1 maps in the database."""
    con = sqlite3.connect(str(database_name))
    try:
        cur = con.cursor()
        cur.execute("SELECT MIN(start_time), MAX(stop_time) FROM depth_one_maps")
        start, stop = cur.fetchone()
    finally:
        con.close()
    if start is None or stop is None:
        raise ValueError(f"No depth_one_maps rows found in {database_name}")
    return start, stop


def iso(unix_time: float) -> str:
    return datetime.fromtimestamp(unix_time, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def window_bounds(
    start: float, stop: float, window_days: float
) -> list[tuple[float, float]]:
    """Rolling windows of `window_days`, anchored to `start`, covering [start, stop)."""
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
            "bucket_by_start_time": True,
        },
        "preprocessors": [
            {"preprocessor_type": "planet_mask", "mask_radius": "15 arcmin"},
            matched_filter,
            {"preprocessor_type": "kappa_rho"},
            {
                "preprocessor_type": "edge_mask",
                "mask_on": "kappa",
                "edge_width": "10.0 arcmin",
            },
        ],
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
export MAPCAT_DATABASE_NAME={database_name}

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
        help="Directory to write coadded map FITS outputs to (one subdir per window is not "
        "created automatically; include it in this path if desired).",
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
        help="Restrict to a single array/tube_slot when reading maps. Default: all arrays, "
        "summed together into one coadd per frequency (depth_one_coadds has no array "
        "column, so a registered coadd is always frequency-only).",
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
        help="Coadded map fields to save to FITS via the 'maps' output. sotrp-coadd writes "
        "these in two passes (rho/kappa before finalize, flux/snr after), so both kinds "
        "can be requested together.",
    )
    p.add_argument(
        "--beam1d-template",
        type=str,
        default="profile_{frequency}_1756699200_20000000000.txt",
        help="Path template (relative to --repo-dir) for the matched-filter 1D beam profile "
        "per frequency. Only used if the resolved file exists; set to '' to disable.",
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
        "--rerun", action="store_true", help="Re-process already-completed maps."
    )
    p.add_argument(
        "--submit",
        action="store_true",
        help="Actually sbatch the generated scripts. Without this, only writes files.",
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
                output_dir=output_dir,
                beam1d=beam1d,
                fields=args.fields,
                rerun=args.rerun,
                coadd_name=f"{frequency}_{int(w_start)}_{int(w_stop)}",
                coadd_type=args.coadd_type,
                register=args.register_coadds,
            )
            config_file = config_dir / f"{tag}.json"
            config_file.write_text(json.dumps(config, indent=2))

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
                database_name=args.database_name,
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
