#!/usr/bin/env python3
"""
Submit one SLURM job per depth-1 coadd (mapcat's depth_one_coadds table)
that runs the full sotrp pipeline (pointing calibration, forced photometry,
blind search, sifting) against it via CoaddRhoKappaMapReader
(map_generator_type "mapcat_coadd_database").

Unlike the depth-1-map pipeline, no matched_filter preprocessor is included
-- coadds are already matched-filtered rho/kappa maps (built by
sotrp-coadd/submit_week_coadds.py), so re-filtering would double-filter.

Generates a sotrp config JSON + a SLURM script per coadd, and by default
only writes them; pass --submit to actually sbatch them.
"""

import argparse
import json
import sqlite3
from getpass import getuser
from pathlib import Path

USER = getuser()

REPO_DIR = Path(__file__).resolve().parents[2]


def get_coadds(
    database_name: Path, frequencies: list[str] | None
) -> list[tuple[str, str, str]]:
    """(coadd_id, coadd_name, frequency) for every matching coadd."""
    con = sqlite3.connect(str(database_name))
    try:
        cur = con.cursor()
        query = "SELECT coadd_id, coadd_name, frequency FROM depth_one_coadds"
        params: list[str] = []
        if frequencies:
            query += f" WHERE frequency IN ({','.join('?' * len(frequencies))})"
            params = frequencies
        query += " ORDER BY coadd_name"
        cur.execute(query, params)
        rows = cur.fetchall()
    finally:
        con.close()
    if not rows:
        raise ValueError(f"No depth_one_coadds rows found in {database_name}")
    return rows


def build_config(
    coadd_id: str,
    frequency: str,
    output_dir: Path,
    socat_db_path: Path,
    rerun: bool,
) -> dict:
    return {
        "maps": {
            "map_generator_type": "mapcat_coadd_database",
            "coadd_ids": [coadd_id],
            "frequency": frequency,
            "rerun": rerun,
        },
        "source_catalogs": [
            {"catalog_type": "socat"},
        ],
        "pointing_provider": {
            "photometry_type": "lmfit_pointing",
            "thumbnail_half_width": "6 arcmin",
            "min_flux": "0.3 Jy",
            "reproject_thumbnails": "True",
            "allowable_center_offset": "3.0 arcmin",
            "near_source_rel_flux_limit": 0.3,
        },
        "pointing_residual": {
            "pointing_residual_type": "mean",
            "min_snr": 5.0,
            "min_sources": 5,
            "sigma_clip_level": 3.0,
            "polynomial_order": 3,
        },
        "preprocessors": [
            {"preprocessor_type": "planet_mask", "mask_radius": "15 arcmin"},
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
                "tile_size": ".5 deg",
            }
        ],
        "source_subtractor": {"subtractor_type": "photutils"},
        "blind_search": {"search_type": "photutils"},
        "forced_photometry": {
            "photometry_type": "lmfit",
            "reproject_thumbnails": "True",
            "flux_limit_centroid": "0.1 Jy",
            "min_flux": "0.01 Jy",
            "thumbnail_half_width": "4 arcmin",
            "allowable_center_offset": "1.0 arcmin",
            "near_source_rel_flux_limit": 1.0,
        },
        "sifter": {"sifter_type": "default"},
        "source_outputs": [
            {"output_type": "pickle", "directory": str(output_dir)},
        ],
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
export socat_client_client_type=db
export socat_model_database_name={socat_db_path}

srun --overlap sotrp -c {config_file} > {log_file} 2>&1
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
        help="Parent directory that depth-1 map/coadd paths in the database are relative to.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write source_outputs (pickled forced-photometry/blind-search "
        "results) to. Shared across all jobs -- filenames are prefixed by coadd_id.",
    )
    p.add_argument(
        "--socat-db-path",
        type=Path,
        required=True,
        help="Path to the socat sqlite database (source catalog) to use for forced "
        "photometry and pointing calibration.",
    )
    p.add_argument(
        "--frequencies",
        nargs="+",
        default=None,
        help="Restrict to these frequency bands. Default: all coadds in the database.",
    )
    p.add_argument(
        "--repo-dir",
        type=Path,
        default=REPO_DIR,
        help="sotrplib checkout to cd into and activate .venv in for each job.",
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
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reprocess coadds even if already marked processing/completed in mapcat.",
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

    coadds = get_coadds(args.database_name, args.frequencies)
    print(f"{len(coadds)} coadd(s) found")

    submitted = []
    for coadd_id, coadd_name, frequency in coadds:
        tag = coadd_name

        config = build_config(
            coadd_id=coadd_id,
            frequency=frequency,
            output_dir=output_dir,
            socat_db_path=args.socat_db_path,
            rerun=args.rerun,
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
            socat_db_path=args.socat_db_path,
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
