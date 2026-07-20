import argparse as ap
import os
from getpass import getuser

USER = getuser()

P = ap.ArgumentParser(
    description="Run sotrp historical forced-photometry over every mapcat-"
    "registered depth-1 map, per (frequency, optics tube). Registers the "
    "given --ra/--dec/--source-id sources as monitored in --socat-db-path "
    "(a fresh, batch-scoped db by default) and generates one slurm job per "
    "band x tube that measures every registered source against every map "
    "in a single pass.",
    formatter_class=ap.ArgumentDefaultsHelpFormatter,
)
P.add_argument(
    "--ra",
    action="store",
    default=[],
    nargs="+",
    help="RA position (decimal degrees) of sources to extract.",
)
P.add_argument(
    "--dec",
    action="store",
    default=[],
    nargs="+",
    help="Dec position (decimal degrees) of sources to extract.",
)
P.add_argument(
    "--flux",
    action="store",
    default=[],
    nargs="+",
    help="Fluxes (Jy) of sources to extract.",
)
P.add_argument(
    "--source-id",
    action="store",
    default=[],
    nargs="+",
    required=True,
    help="Source ids of sources to extract.",
)

P.add_argument(
    "--flux-threshold",
    action="store",
    default="10 mJy",
    type=str,
    help="Flux threshold for source extraction, json compatible quantity.",
)

P.add_argument(
    "--socat-db-path",
    action="store",
    default="",
    help="Path to a sqlite socat database. The sources given via --ra/--dec/--flux/"
    "--source-id are added to this database (creating it if needed) so the slurm "
    "jobs can look them up as monitored sources. If empty, defaults to "
    "'tmp_socat.db' inside --out-dir.",
)

P.add_argument(
    "--scratch-dir",
    action="store",
    default=f"/scratch/gpfs/SIMONSOBS/users/{USER}/scratch/",
    help="Location of scratch directory for users. Default: /scratch/gpfs/SIMONSOBS/users/[user]/scratch/.",
)

P.add_argument(
    "--script-dir",
    action="store",
    default="",
    help="Directory in which the trp_pipeline script lives. If default, then will use os.cwd() which assumes this script lives in the same directory.",
)

P.add_argument(
    "--out-dir",
    action="store",
    default="",
    help="Output directory for output files. If empty, will create directory in scratch_dir.",
)

P.add_argument(
    "--group-name",
    action="store",
    default="simonsobs",
    help="Name of computing cluster group that you belong to, for running slurm jobs.",
)

P.add_argument(
    "--slurm-script-dir",
    action="store",
    default="slurm_job_scripts/",
    help="Directory in which the slurm job scripts lives.",
)

P.add_argument(
    "--slurm-out-dir",
    action="store",
    default="slurm_output_files/",
    help="Directory in which the slurm output files get saved. If default, appends to scratch dir. ",
)

P.add_argument(
    "--save-thumbnails",
    action="store_true",
    default=True,
    help="Save thumbnail maps around the transient sources. ",
)

P.add_argument(
    "--thumbnail-radius",
    action="store",
    default="0.1 deg",
    type=str,
    help="Thumbnail radius (half-width of square map), json compatible quantity.",
)

P.add_argument(
    "--ncores",
    action="store",
    default=12,
    type=int,
    help="Number of cores to request for each slurm job. ",
)

P.add_argument(
    "--nserial",
    action="store",
    default=1,
    type=int,
    help="Number of serial sets of `ncores` scripts to run per job. ",
)

P.add_argument(
    "--bands",
    action="store",
    nargs="+",
    default=["f090", "f150", "f220", "f280"],
    help="Bands to analyze, default is f090,f150,f220, f280 but can also include f030 or f040. ",
)

P.add_argument(
    "--optics-tubes",
    action="store",
    nargs="+",
    default=["i1", "i3", "i4", "i6", "c1", "i5"],
    help="Optics tubes to analyze, default for nominal SO is i1,i3,i4,i6,c1,i5. ",
)

P.add_argument(
    "--also-output-lightcurvedb",
    action="store_true",
    default=False,
    help="Also add a 'lightcurvedb' source_output alongside 'pickle', so this "
    "sweep's forced-photometry measurements get uploaded directly to "
    "lightcurvedb. Assumes the source(s) given via --ra/--dec/--source-id are "
    "already registered in socat and synced into lightcurvedb before this "
    "runs (upsert_sources is left False). Requires a lightcurvedb postgres "
    "reachable from the compute nodes -- an output failure fails the whole "
    "sotrp job and marks its maps 'failed' in mapcat.",
)

P.add_argument(
    "--pythonpath",
    action="store",
    default="",
    type=str,
    help="If set, exported as PYTHONPATH in each slurm job so the jobs run "
    "sotrplib from this directory (e.g. a git worktree) instead of the "
    "editable install the venv points at.",
)

args = P.parse_args()


def generate_slurm_header(
    jobname,
    groupname,
    cpu_per_task,
    script_dir,
    slurm_out_dir,
    socat_db_path,
    pythonpath="",
):
    slurm_header = f"""#!/bin/bash
#SBATCH --job-name={jobname}            # create a short name for your job
#SBATCH -A {groupname}                    # group name, by default simonsobs
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task={cpu_per_task}       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=24G         # memory per cpu-core (4G is default)
#SBATCH --time=04:59:00          # total run time limit (HH:MM:SS)
#SBATCH --output={slurm_out_dir}%x.out

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

cd {script_dir}
source .venv/bin/activate

export socat_client_client_type=db
export socat_model_database_name={socat_db_path}

export MAPCAT_DEPTH_ONE_PARENT=/scratch/gpfs/SIMONSOBS/users/amfoster/so/lat_early_maps/
export MAPCAT_DATABASE_NAME=/scratch/gpfs/SIMONSOBS/users/amfoster/so/lat_early_maps/out_deep56/mapcat.sqlite


"""
    if pythonpath:
        slurm_header += f"export PYTHONPATH={pythonpath}\n\n"
    return slurm_header


def generate_config_json(
    frequency: str | None = None,
    array: str | None = None,
    flux_low_limit="0.01 Jy",
    thumbnail_half_width="0.2 deg",
    output_dir="./",
    also_output_lightcurvedb: bool = False,
):
    """No per-source spatial box: the sources actually measured on each map
    are whichever registered sources in the (batch-scoped) socat database
    fall inside that map's own footprint, so restricting the map query to a
    box around one source only ever excluded maps, never widened the
    per-map source list -- every map in mapcat_database's array+frequency
    query already covers this field. Dropping it lets one job (per
    frequency/array) sweep every source in the batch in a single pass over
    each historical map, instead of one job per source re-reading the same
    maps."""
    source_outputs = '{"output_type": "pickle", "directory": "' + str(output_dir) + '"}'
    if also_output_lightcurvedb:
        # upsert_sources is False: the source is expected to already be
        # registered in socat (and synced into lightcurvedb) before this
        # historical sweep runs, not created on the fly from whatever this
        # sweep happens to crossmatch against.
        source_outputs += ', {"output_type": "lightcurvedb", "upsert_sources": false}'

    config_text = f"""{{
    "maps": {{
        "map_generator_type": "mapcat_database",
        "number_to_read": 1000000,
        "instrument": "SOLAT",
        "frequency": "{frequency}",
        "array": "{array}",
        "rerun": "True"
    }},
    "source_catalogs": [
        {{
            "catalog_type": "socat"
        }}
    ],
    "preprocessors": [
        {{
            "preprocessor_type": "planet_mask",
            "mask_radius": "15 arcmin"
        }},
        {{
            "preprocessor_type": "matched_filter",
            "beam1d": "profile_{frequency}_1756699200_20000000000.txt",
            "band_height": "1 deg",
            "shrink_holes": "5 arcmin",
            "noisemask_lim": 0.1,
            "noisemask_radius": "10 arcmin",
            "apod_holes": "10 arcmin"
        }},
        {{
            "preprocessor_type": "kappa_rho"
        }},
        {{
            "preprocessor_type": "edge_mask",
            "mask_on": "kappa",
            "edge_width": "10.0 arcmin"
        }}
    ],
    "postprocessors": [
       {{
            "postprocessor_type": "flatfield",
            "sigma_val": 5.0,
            "tile_size": "0.5 deg"
        }}
    ],
    "forced_photometry": {{
        "photometry_type": "lmfit",
        "reproject_thumbnails": "True",
        "flux_limit_centroid": "0.1 Jy",
        "thumbnail_half_width": "{thumbnail_half_width}",
        "allowable_center_offset": "1.0 arcmin",
        "near_source_rel_flux_limit": 1.0
    }},
    "sifter": {{
        "sifter_type": "default",
        "min_match_radius": "5.0 arcmin"
    }},
    "source_outputs": [{source_outputs}]
}}
"""
    return config_text


nserial = args.ncores * args.nserial

## make scratch output directory
user_scratch = args.scratch_dir
if not os.path.exists(user_scratch):
    os.mkdir(user_scratch)

if not args.out_dir:
    ## get random tmp dir
    import random
    import string

    letters = string.ascii_lowercase
    random_string = "".join(random.choice(letters) for i in range(6))

    args.out_dir = user_scratch + f"tmp_sotrpdir_{random_string}/"
    print("Creating temp output directory: ", args.out_dir)

## set slurm output dir to live in the temp directory
if args.slurm_out_dir == P.get_default("slurm_out_dir"):
    args.slurm_out_dir = args.out_dir + args.slurm_out_dir
if args.slurm_script_dir == P.get_default("slurm_script_dir"):
    args.slurm_script_dir = args.out_dir + args.slurm_script_dir

if not os.path.exists(args.out_dir):
    os.mkdir(args.out_dir)
if not os.path.exists(args.slurm_script_dir):
    os.mkdir(args.slurm_script_dir)

if not args.socat_db_path:
    args.socat_db_path = os.path.join(args.out_dir, "tmp_socat.db")
args.socat_db_path = os.path.abspath(args.socat_db_path)

## Build a temporary socat sqlite database and register the requested
## sources in it as monitored sources, so the slurm jobs can look them up.
if args.source_id:
    os.environ["socat_client_client_type"] = "db"
    os.environ["socat_model_database_name"] = args.socat_db_path

    from astropy import units as u
    from astropy.coordinates import ICRS
    from socat.client.settings import SOCatClientSettings

    socat_client = SOCatClientSettings().client
    ## dispatch_historical_extraction.py registers its candidates in the
    ## production socat.db before invoking this wrapper -- re-creating them
    ## here would duplicate rows (there is no name-uniqueness constraint),
    ## so skip any name already present. Queried straight from sqlite: the
    ## client offers no lookup-by-name, and get_monitored_sources builds
    ## SourceGenerators for every monitored SSO (slow, and crashes on
    ## ephemerides with duplicate timestamps).
    already_registered = set()
    if os.path.exists(args.socat_db_path):
        import sqlite3

        try:
            with sqlite3.connect(args.socat_db_path) as connection:
                already_registered = {
                    row[0]
                    for row in connection.execute("SELECT name FROM fixed_sources")
                }
        except sqlite3.Error:
            already_registered = set()
    n_created = 0
    for s, source_id in enumerate(args.source_id):
        if source_id in already_registered:
            continue
        socat_client.create_source(
            position=ICRS(ra=float(args.ra[s]) * u.deg, dec=float(args.dec[s]) * u.deg),
            flux=float(args.flux[s]) * u.Jy if args.flux else None,
            name=source_id,
            flags={"monitored": True},
        )
        n_created += 1
    print(
        f"Registered {n_created} new source(s) in socat database at "
        f"{args.socat_db_path} "
        f"({len(args.source_id) - n_created} already present)."
    )

## One job per (band, tube): every registered source in this batch's socat
## database is measured against every historical map in a single pass,
## instead of re-reading the same maps once per source (see
## generate_config_json's docstring for why the per-source box was
## removed).
n = 0

for band in args.bands:
    for tube in args.optics_tubes:
        config_file = args.slurm_script_dir + f"{tube}_{band}_config.json"
        with open(config_file, "w") as f:
            f.write(
                generate_config_json(
                    frequency=band,
                    array=tube,
                    flux_low_limit=args.flux_threshold,
                    thumbnail_half_width=args.thumbnail_radius,
                    output_dir=args.out_dir,
                    also_output_lightcurvedb=args.also_output_lightcurvedb,
                )
            )
        slurm_text = generate_slurm_header(
            f"historical_extractor_{tube}_{band}",
            args.group_name,
            str(args.ncores),
            args.script_dir if args.script_dir else os.getcwd(),
            args.slurm_out_dir,
            args.socat_db_path,
            pythonpath=args.pythonpath,
        )
        slurm_text += f"srun --overlap sotrp -c {config_file} > {args.out_dir}{tube}_{band}_sotrp.log "

        with open(f"{args.slurm_script_dir}/{tube}_{band}_sub.slurm", "w") as f:
            f.write(slurm_text)
        n += 1

print(f"Generated {n} slurm job(s) for {len(args.source_id)} source(s).")


print("#" * 50)
print("Slurm scripts saved to :", args.slurm_script_dir)
print("Slurm output saved to :", args.slurm_out_dir)

print("Output files saved to :", args.out_dir)

print(
    "To run the slurm jobs, point slurm_submitter.py to the slurm script location stated above."
)

print("#" * 50)
