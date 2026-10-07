"""
Write one sotrp config and one SLURM job for each observation in mapcat, so
that the map matcher can compare the maps of all the tubes and bands of an
observation.

An observation is a set of depth-1 maps whose time ranges overlap (the same
rule as the map matcher). Each config selects the maps of one observation by
their map_id. Thus, an observation that crosses a day boundary is analyzed
once, in one job.

Each job runs the basic runner. The jobs run in parallel. Submit them with
slurm_submitter.py:

    python scripts/depth1_map_analysis/slurm_wrapper_map_matching_by_observation.py \\
        --out-dir /path/to/outputs/ --code-dir /path/to/sotrplib/checkout \\
        --venv /path/to/sotrplib/.venv --env-setup-file /path/to/env_setup
    python scripts/depth1_map_analysis/slurm_submitter.py -d /path/to/outputs/slurm_job_scripts/
"""

import argparse as ap
import json
import os
import sqlite3
import uuid
from datetime import datetime
from getpass import getuser
from pathlib import Path

USER = getuser()

P = ap.ArgumentParser(
    description=__doc__.split("\n\n")[0],
    formatter_class=ap.ArgumentDefaultsHelpFormatter,
)
P.add_argument(
    "--mapcat",
    default=os.environ.get("MAPCAT_DATABASE_NAME", ""),
    help="mapcat SQLite file to read the maps from. Default: $MAPCAT_DATABASE_NAME.",
)
P.add_argument(
    "--start-time",
    default=None,
    help="Only maps that start at or after this time (ISO).",
)
P.add_argument(
    "--end-time", default=None, help="Only maps that start before this time (ISO)."
)
P.add_argument(
    "--bands",
    nargs="+",
    default=["f090", "f150", "f220", "f280"],
    help="Bands to analyze.",
)
P.add_argument(
    "--optics-tubes",
    nargs="+",
    default=["i1", "i3", "i4", "i6", "c1", "i5"],
    help="Optics tubes to analyze.",
)
P.add_argument(
    "--out-dir",
    required=True,
    help="Output directory for the pickles, logs and map-match summaries.",
)
P.add_argument(
    "--code-dir",
    default=os.getcwd(),
    help="sotrplib checkout (or worktree) whose code the jobs run. The jobs run from this directory.",
)
P.add_argument(
    "--venv",
    default=None,
    help="Python virtual environment. Default: <code-dir>/.venv.",
)
P.add_argument(
    "--env-setup-file",
    default="env_setup",
    help="File that sets the mapcat and socat variables.",
)
P.add_argument(
    "--beam-profile",
    default="profile_{band}_1756699200_20000000000.txt",
    help="Beam profile file for each band; {band} is replaced. Relative paths are relative to --code-dir.",
)
P.add_argument(
    "--blind-snr",
    type=float,
    default=3.0,
    help="Blind search threshold and sifter snr cut.",
)
P.add_argument(
    "--high-sig",
    type=float,
    default=5.0,
    help="Map matcher high_sig: one detection must reach this SNR.",
)
P.add_argument(
    "--low-sig",
    type=float,
    default=3.0,
    help="Map matcher low_sig: lowest SNR that is matched.",
)
P.add_argument("--min-arrays", type=int, default=2, help="Map matcher min_arrays.")
P.add_argument("--match-radius", default="1.5 arcmin", help="Map matcher radius.")
P.add_argument(
    "--pointing-flux-threshold",
    default="0.3 Jy",
    help="Minimum flux of the pointing sources.",
)
P.add_argument(
    "--flux-threshold",
    default="10 mJy",
    help="Minimum catalog flux for forced photometry.",
)
P.add_argument(
    "--pointing-snr",
    type=float,
    default=5.0,
    help="Minimum SNR of the pointing sources.",
)
P.add_argument(
    "--thumbnail-radius", default="0.1 deg", help="Pointing thumbnail half-width."
)
P.add_argument("--group-name", default="simonsobs", help="SLURM account.")
P.add_argument("--ncores", type=int, default=4, help="CPU cores for each job.")
P.add_argument("--mem", default="32G", help="Memory for each job.")
P.add_argument("--time", default="02:00:00", help="Time limit for each job.")
args = P.parse_args()


def observations(db: str) -> list[list[dict]]:
    """Read the depth-1 maps and group them by overlapping time ranges."""
    con = sqlite3.connect(db)
    rows = con.execute(
        "SELECT map_id, map_name, tube_slot, frequency, start_time, stop_time FROM depth_one_maps"
    ).fetchall()
    maps = []
    for map_id, name, tube, band, start, stop in rows:
        start, stop = (
            datetime.fromisoformat(str(start)),
            datetime.fromisoformat(str(stop)),
        )
        if tube not in args.optics_tubes or band not in args.bands:
            continue
        if args.start_time and start < datetime.fromisoformat(args.start_time):
            continue
        if args.end_time and start >= datetime.fromisoformat(args.end_time):
            continue
        maps.append(
            dict(
                map_id=str(uuid.UUID(str(map_id))),
                name=name,
                tube=tube,
                band=band,
                start=start,
                stop=stop,
            )
        )

    groups: list[list[dict]] = []
    end = None
    for m in sorted(maps, key=lambda m: m["start"]):
        if end is not None and m["start"] <= end:
            groups[-1].append(m)
            end = max(end, m["stop"])
        else:
            groups.append([m])
            end = m["stop"]
    return groups


def config(obs: list[dict], out_dir: str, summary_dir: str) -> dict:
    bands = sorted({m["band"] for m in obs})
    beam = {
        b: str(Path(args.code_dir) / args.beam_profile.format(band=b))
        if not os.path.isabs(args.beam_profile)
        else args.beam_profile.format(band=b)
        for b in bands
    }
    return {
        "maps": {
            "map_generator_type": "mapcat_database",
            "number_to_read": len(obs),
            "map_ids": [m["map_id"] for m in obs],
            "rerun": True,
        },
        "source_catalogs": [
            {"catalog_type": "socat", "flux_lower_limit": args.flux_threshold}
        ],
        "pointing_provider": {
            "photometry_type": "lmfit_pointing",
            "thumbnail_half_width": args.thumbnail_radius,
            "min_flux": args.pointing_flux_threshold,
            "reproject_thumbnails": True,
            "allowable_centroid_offset": "3.0 arcmin",
        },
        "pointing_residual": {
            "pointing_residual_type": "median",
            "min_snr": args.pointing_snr,
            "min_sources": 10,
        },
        "preprocessors": [
            {"preprocessor_type": "planet_mask", "mask_radius": "15 arcmin"},
            {
                "preprocessor_type": "matched_filter",
                "beam1d": beam,
                "band_height": "1 deg",
                "shrink_holes": "5 arcmin",
                "noisemask_lim": 0.1,
                "noisemask_radius": "10 arcmin",
                "apod_holes": "10 arcmin",
            },
            {"preprocessor_type": "kappa_rho"},
            {
                "preprocessor_type": "edge_mask",
                "mask_on": "kappa",
                "edge_width": "10.0 arcmin",
            },
        ],
        "postprocessors": [
            {
                "postprocessor_type": "flatfield",
                "sigma_val": 5.0,
                "tile_size": "0.5 deg",
            }
        ],
        "source_subtractor": {"subtractor_type": "photutils"},
        "blind_search": {
            "search_type": "photutils",
            "parameters": {"sigma_threshold": args.blind_snr},
        },
        "forced_photometry": {
            "photometry_type": "lmfit",
            "reproject_thumbnails": True,
            "flux_limit_centroid": "0.1 Jy",
            "thumbnail_half_width": "4 arcmin",
            "allowable_center_offset": "1.0 arcmin",
            "near_source_rel_flux_limit": 1.0,
        },
        "sifter": {
            "sifter_type": "default",
            "min_match_radius": "5.0 arcmin",
            "cuts": {"snr": [args.blind_snr, "inf"]},
        },
        "map_matcher": {
            "matcher_type": "multi_array",
            "radius": args.match_radius,
            "min_arrays": args.min_arrays,
            "high_sig": args.high_sig,
            "low_sig": args.low_sig,
            "summary_directory": summary_dir,
        },
        "source_outputs": [{"output_type": "pickle", "directory": out_dir}],
        "runner": "basic",
    }


def slurm(name: str, config_file: str, log_file: str, slurm_out_dir: str) -> str:
    venv = args.venv or str(Path(args.code_dir) / ".venv")
    return f"""#!/bin/bash
#SBATCH --job-name={name}
#SBATCH -A {args.group_name}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={args.ncores}
#SBATCH --mem={args.mem}
#SBATCH --time={args.time}
#SBATCH --output={slurm_out_dir}%x.out

cd {args.code_dir}
source {venv}/bin/activate
source {args.env_setup_file}
export sotrp_runner=basic

# Run the code in --code-dir, not the code of the installed sotrp command.
python -c "import sotrplib; print('sotrplib from', sotrplib.__file__)"
python -c "from sotrplib.cli import main; main()" -c {config_file} > {log_file} 2>&1
echo "sotrp exit status $?"
"""


if not args.mapcat:
    P.error("give --mapcat or set MAPCAT_DATABASE_NAME")

out_dir = str(Path(args.out_dir).resolve()) + "/"
summary_dir = out_dir + "map_match/"
script_dir = out_dir + "slurm_job_scripts/"
slurm_out_dir = out_dir + "slurm_output_files/"
for d in (out_dir, summary_dir, script_dir, slurm_out_dir):
    os.makedirs(d, exist_ok=True)

groups = observations(args.mapcat)
for obs in groups:
    label = min(m["start"] for m in obs).strftime("%Y%m%d-%H%M%S")
    config_file = f"{script_dir}obs_{label}_config.json"
    with open(config_file, "w") as f:
        json.dump(config(obs, out_dir, summary_dir), f, indent=4)
    with open(f"{script_dir}obs_{label}_sub.slurm", "w") as f:
        f.write(
            slurm(
                f"sotrp_obs_{label}",
                config_file,
                f"{out_dir}obs_{label}_sotrp.log",
                slurm_out_dir,
            )
        )

n_maps = sum(len(o) for o in groups)
print("#" * 50)
print(
    f"{len(groups)} observations, {n_maps} maps (maps per observation: {sorted({len(o) for o in groups})})"
)
print("Slurm scripts and configs saved to:", script_dir)
print("Map-match summaries saved to:", summary_dir)
print("Output files saved to:", out_dir)
print(
    f"Submit with: python scripts/depth1_map_analysis/slurm_submitter.py -d {script_dir}"
)
print("#" * 50)
