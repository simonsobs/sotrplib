"""
Write one sotrp config and one SLURM job for each source, to measure the
source in all the mapcat depth-1 maps that cover it (forced photometry).

The script selects the maps of each source in mapcat by their sky coverage,
with the time window (--start-time, --end-time), the bands and the optics
tubes. Each config gives these maps by map_id. Thus, each job reads only the
maps that cover its source, in all bands and tubes. The reader crops each
map to a box around the source (--box-half-width).

The jobs do not read or write the processing status of the maps
(track_processing false). Thus, they do not change the status that the
daily pipeline uses.

There are two ways to give the sources:

- --socat-id: sources that are already in a socat database
  (--copy-from-socat-db). The script copies them, with the same socat
  source_id, to the socat database of the jobs. The lightcurvedb output
  finds a lightcurvedb source by its socat source_id, so the measurements
  go to the lightcurve of the source.
- --ra, --dec, --source-id (and optionally --flux): new sources. The script
  adds them to the socat database of the jobs.

The socat database of the jobs (--socat-db-path) contains only these
sources, so the jobs measure only these sources.

    python scripts/historical_lightcurve_extractor/slurm_wrapper_mapcat_historical_extractor.py \\
        --mapcat /path/to/mapcat.sqlite --code-dir /path/to/sotrplib/checkout \\
        --env-setup-file /path/to/env_setup --out-dir /path/to/outputs/ \\
        --copy-from-socat-db /path/to/socat.db --socat-id <uuid> \\
        --end-time 2025-09-11T00:00:00
    python scripts/depth1_map_analysis/slurm_submitter.py -d /path/to/outputs/slurm_job_scripts/
"""

import argparse as ap
import json
import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class Source:
    name: str
    ra: float
    dec: float
    flux_jy: float | None = None
    socat_id: str | None = None


def parser() -> ap.ArgumentParser:
    p = ap.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=ap.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--socat-id",
        nargs="+",
        default=[],
        help="socat source_ids of sources to copy from --copy-from-socat-db.",
    )
    p.add_argument(
        "--copy-from-socat-db",
        default=None,
        help="socat SQLite database that contains the --socat-id sources.",
    )
    p.add_argument("--ra", nargs="+", default=[], help="RA (deg) of each new source.")
    p.add_argument("--dec", nargs="+", default=[], help="Dec (deg) of each new source.")
    p.add_argument(
        "--flux", nargs="+", default=[], help="Flux (Jy) of each new source."
    )
    p.add_argument(
        "--source-id", nargs="+", default=[], help="Name of each new source."
    )
    p.add_argument(
        "--mapcat",
        default=os.environ.get("MAPCAT_DATABASE_NAME"),
        help="mapcat SQLite file to select the maps from. Default: $MAPCAT_DATABASE_NAME.",
    )
    p.add_argument(
        "--start-time",
        default=None,
        help="Only maps that start at or after this time (ISO).",
    )
    p.add_argument(
        "--end-time",
        default=None,
        help="Only maps that start before this time (ISO). Use it to not read maps after a given time.",
    )
    p.add_argument(
        "--bands",
        nargs="+",
        default=["f090", "f150", "f220", "f280"],
        help="Bands to measure.",
    )
    p.add_argument(
        "--optics-tubes",
        nargs="+",
        default=["i1", "i3", "i4", "i6", "c1", "i5"],
        help="Optics tubes to measure.",
    )
    p.add_argument(
        "--socat-db-path",
        default="",
        help="socat SQLite database of the jobs. The script adds the sources to it "
        "(and makes it if necessary). Default: <out-dir>/extractor_socat.db.",
    )
    p.add_argument(
        "--out-dir", required=True, help="Output directory for pickles and logs."
    )
    p.add_argument(
        "--code-dir",
        default=os.getcwd(),
        help="sotrplib checkout (or worktree) whose code the jobs run.",
    )
    p.add_argument(
        "--venv",
        default=None,
        help="Python virtual environment. Default: <code-dir>/.venv.",
    )
    p.add_argument(
        "--env-setup-file",
        default=None,
        help="File that the jobs source to set the mapcat (and lightcurvedb) variables.",
    )
    p.add_argument(
        "--beam-profile",
        default="profile_{band}_1756699200_20000000000.txt",
        help="Beam profile file for each band; {band} is replaced. Relative paths are relative to --code-dir.",
    )
    p.add_argument(
        "--thumbnail-radius",
        default="0.1 deg",
        help="Thumbnail half-width, json compatible quantity.",
    )
    p.add_argument(
        "--box-half-width",
        type=float,
        default=0.5,
        help="Half-width (deg) of the box that the reader cuts from each map.",
    )
    p.add_argument(
        "--lightcurvedb",
        action="store_true",
        help="Also write the measurements to lightcurvedb (settings from the LIGHTCURVEDB_ variables).",
    )
    p.add_argument(
        "--no-pickle", action="store_true", help="Do not write pickle outputs."
    )
    p.add_argument("--group-name", default="simonsobs", help="SLURM account.")
    p.add_argument("--ncores", type=int, default=4, help="CPU cores for each job.")
    p.add_argument("--mem", default="32G", help="Memory for each job.")
    p.add_argument("--time", default="04:00:00", help="Time limit for each job.")
    return p


def coverage_index(ra: float, dec: float) -> tuple[int, int]:
    from mapcat.toolkit.update_sky_coverage import dec_to_index, ra_to_index

    return ra_to_index(ra % 360.0), dec_to_index(dec)


def maps_covering(
    mapcat: str,
    ra: float,
    dec: float,
    bands: list[str],
    tubes: list[str],
    start_time: str | None = None,
    end_time: str | None = None,
) -> list[dict]:
    """
    Return the depth-1 maps whose sky coverage contains (ra, dec), in the
    bands, tubes and time window. A map is in the window if it starts at or
    after start_time and before end_time.

    If the sky coverage table of mapcat is empty, return all the maps in the
    bands, tubes and time window, and print a warning. To fill the table,
    run the mapcat command `updatesky`.
    """
    x, y = coverage_index(ra, dec)
    columns = "m.map_id, m.map_name, m.tube_slot, m.frequency, m.start_time"
    con = sqlite3.connect(f"file:{mapcat}?mode=ro", uri=True)
    try:
        (n_coverage,) = con.execute(
            "SELECT COUNT(*) FROM depth_one_sky_coverage"
        ).fetchone()
        if n_coverage == 0:
            print(
                "WARNING: the depth_one_sky_coverage table is empty. Selecting maps "
                "by band, tube and time only. Run `updatesky` to fill the table."
            )
            rows = con.execute(f"SELECT {columns} FROM depth_one_maps m").fetchall()
        else:
            rows = con.execute(
                f"SELECT {columns} FROM depth_one_maps m "
                "JOIN depth_one_sky_coverage c ON c.map_id = m.map_id "
                # A map that crosses RA 0 has tiles in the [-180, 180)
                # convention (x - 36). See sotrplib.maps.database.sky_coverage_tiles.
                "WHERE c.x IN (?, ?) AND c.y = ?",
                (x, x - 36, y),
            ).fetchall()
    finally:
        con.close()

    start = datetime.fromisoformat(start_time) if start_time else None
    end = datetime.fromisoformat(end_time) if end_time else None
    maps = []
    for map_id, name, tube, band, map_start in rows:
        if tube not in tubes or band not in bands:
            continue
        map_start = datetime.fromisoformat(str(map_start))
        if start is not None and map_start < start:
            continue
        if end is not None and map_start >= end:
            continue
        maps.append(
            dict(
                map_id=str(uuid.UUID(str(map_id))),
                name=name,
                tube=tube,
                band=band,
                start=map_start,
            )
        )
    return sorted(maps, key=lambda m: m["start"])


def socat_client(db_path: str):
    """A socat client for a SQLite file. Makes the tables if necessary."""
    import socat.database  # noqa: F401  (registers the tables)
    from socat.client.db import Client
    from sqlmodel import SQLModel, create_engine

    url = f"sqlite:///{db_path}"
    engine = create_engine(url)
    SQLModel.metadata.create_all(engine)
    engine.dispose()
    return Client(db_url=url)


def copy_socat_sources(
    source_db: str, socat_ids: list[str], target_db: str
) -> list[Source]:
    """
    Copy sources from source_db to target_db with the same source_id, and
    set them to monitored.
    """
    from socat.database import RegisteredFixedSourceTable

    source_client = socat_client(source_db)
    target_client = socat_client(target_db)
    sources = []
    for socat_id in socat_ids:
        found = source_client.get_source(source_id=uuid.UUID(socat_id))
        ra = found.position.ra.to_value("deg")
        dec = found.position.dec.to_value("deg")
        flux_mjy = found.flux.to_value("mJy") if found.flux is not None else None
        with target_client._get_session() as session:
            if session.get(RegisteredFixedSourceTable, found.source_id) is None:
                session.add(
                    RegisteredFixedSourceTable(
                        source_id=found.source_id,
                        ra_deg=ra,
                        dec_deg=dec,
                        flux_mJy=flux_mjy,
                        name=found.name,
                        monitored=True,
                    )
                )
                session.commit()
        sources.append(
            Source(
                name=found.name or str(found.source_id),
                ra=ra,
                dec=dec,
                flux_jy=flux_mjy / 1000 if flux_mjy is not None else None,
                socat_id=str(found.source_id),
            )
        )
    return sources


def add_new_sources(
    names: list[str],
    ras: list[str],
    decs: list[str],
    fluxes: list[str],
    target_db: str,
) -> list[Source]:
    from astropy import units as u
    from astropy.coordinates import ICRS

    client = socat_client(target_db)
    sources = []
    for i, name in enumerate(names):
        ra, dec = float(ras[i]), float(decs[i])
        flux = float(fluxes[i]) if fluxes else None
        created = client.create_source(
            position=ICRS(ra=ra * u.deg, dec=dec * u.deg),
            flux=flux * u.Jy if flux is not None else None,
            name=name,
            flags={"monitored": True},
        )
        sources.append(
            Source(
                name=name, ra=ra, dec=dec, flux_jy=flux, socat_id=str(created.source_id)
            )
        )
    return sources


def config(
    source: Source,
    maps: list[dict],
    out_dir: str,
    args: ap.Namespace,
) -> dict:
    bands = sorted({m["band"] for m in maps})
    beam = {
        b: (
            args.beam_profile.format(band=b)
            if os.path.isabs(args.beam_profile)
            else str(Path(args.code_dir) / args.beam_profile.format(band=b))
        )
        for b in bands
    }
    half = args.box_half_width
    outputs = []
    if not args.no_pickle:
        outputs.append({"output_type": "pickle", "directory": out_dir})
    if args.lightcurvedb:
        outputs.append({"output_type": "lightcurvedb", "upsert_sources": True})
    return {
        "maps": {
            "map_generator_type": "mapcat_database",
            "number_to_read": len(maps),
            "map_ids": [m["map_id"] for m in maps],
            "track_processing": False,
            "sky_box": [
                {
                    "ra": {"value": source.ra - half, "unit": "deg"},
                    "dec": {"value": source.dec - half, "unit": "deg"},
                },
                {
                    "ra": {"value": source.ra + half, "unit": "deg"},
                    "dec": {"value": source.dec + half, "unit": "deg"},
                },
            ],
        },
        "source_catalogs": [{"catalog_type": "socat"}],
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
        "forced_photometry": {
            "photometry_type": "lmfit",
            "reproject_thumbnails": True,
            "flux_limit_centroid": "0.1 Jy",
            "thumbnail_half_width": args.thumbnail_radius,
            "allowable_center_offset": "1.0 arcmin",
            "near_source_rel_flux_limit": 1.0,
        },
        "source_outputs": outputs,
        "runner": "basic",
    }


def slurm(
    name: str,
    config_file: str,
    log_file: str,
    slurm_out_dir: str,
    socat_db_path: str,
    args: ap.Namespace,
) -> str:
    venv = args.venv or str(Path(args.code_dir) / ".venv")
    env_setup = f"source {args.env_setup_file}\n" if args.env_setup_file else ""
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
{env_setup}export MAPCAT_DATABASE_NAME={args.mapcat}
export socat_client_client_type=db
export socat_model_database_name={socat_db_path}
export sotrp_runner=basic

# Run the code in --code-dir, not the code of the installed sotrp command.
python -c "import sotrplib; print('sotrplib from', sotrplib.__file__)"
python -c "from sotrplib.cli import main; main()" -c {config_file} > {log_file} 2>&1
echo "sotrp exit status $?"
"""


def safe_label(name: str) -> str:
    return "".join(c if c.isalnum() or c in "+-._" else "_" for c in name)


def main(argv: list[str] | None = None) -> list[str]:
    """Write the configs and jobs. Returns the paths of the SLURM scripts."""
    p = parser()
    args = p.parse_args(argv)

    if not args.mapcat:
        p.error("give --mapcat or set MAPCAT_DATABASE_NAME")
    if args.socat_id and not args.copy_from_socat_db:
        p.error("--socat-id needs --copy-from-socat-db")
    if not (len(args.ra) == len(args.dec) == len(args.source_id)):
        p.error("give one --ra, --dec and --source-id for each new source")
    if args.flux and len(args.flux) != len(args.source_id):
        p.error("give one --flux for each new source, or no --flux")
    if not args.socat_id and not args.source_id:
        p.error("give --socat-id or --ra/--dec/--source-id")

    args.mapcat = str(Path(args.mapcat).resolve())
    args.code_dir = str(Path(args.code_dir).resolve())
    out_dir = str(Path(args.out_dir).resolve()) + "/"
    script_dir = out_dir + "slurm_job_scripts/"
    slurm_out_dir = out_dir + "slurm_output_files/"
    for d in (out_dir, script_dir, slurm_out_dir):
        os.makedirs(d, exist_ok=True)
    socat_db_path = os.path.abspath(
        args.socat_db_path or os.path.join(out_dir, "extractor_socat.db")
    )

    sources = []
    if args.socat_id:
        sources += copy_socat_sources(
            args.copy_from_socat_db, args.socat_id, socat_db_path
        )
    if args.source_id:
        sources += add_new_sources(
            args.source_id, args.ra, args.dec, args.flux, socat_db_path
        )

    written = []
    for source in sources:
        maps = maps_covering(
            args.mapcat,
            source.ra,
            source.dec,
            args.bands,
            args.optics_tubes,
            args.start_time,
            args.end_time,
        )
        if not maps:
            print(f"{source.name}: no maps cover the source, no job written.")
            continue
        label = safe_label(source.name)
        config_file = f"{script_dir}{label}_config.json"
        with open(config_file, "w") as f:
            json.dump(config(source, maps, out_dir, args), f, indent=4)
        slurm_file = f"{script_dir}{label}_sub.slurm"
        with open(slurm_file, "w") as f:
            f.write(
                slurm(
                    f"historical_extractor_{label}",
                    config_file,
                    f"{out_dir}{label}_sotrp.log",
                    slurm_out_dir,
                    socat_db_path,
                    args,
                )
            )
        written.append(slurm_file)
        print(
            f"{source.name} (socat {source.socat_id}): {len(maps)} maps, "
            f"{maps[0]['start']:%Y-%m-%d} .. {maps[-1]['start']:%Y-%m-%d}"
        )

    print("#" * 50)
    print("socat database of the jobs:", socat_db_path)
    print("Slurm scripts and configs saved to:", script_dir)
    print("Output files saved to:", out_dir)
    print(
        f"Submit with: python scripts/depth1_map_analysis/slurm_submitter.py -d {script_dir}"
    )
    print("#" * 50)
    return written


if __name__ == "__main__":
    main()
