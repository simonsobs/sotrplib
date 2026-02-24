import argparse as ap
import os
from getpass import getuser

USER = getuser()

P = ap.ArgumentParser(
    description="Run sotrp on depth1 map in slurm job. Search maps in directory and make a .json config file for each observation.",
    formatter_class=ap.ArgumentDefaultsHelpFormatter,
)

P.add_argument(
    "--maps",
    action="store",
    default=[],
    nargs="+",
    help="Depth1 maps to analyze. Will null anything in date-dirs and only analyze these maps.",
)

P.add_argument(
    "--date-dirs",
    action="store",
    default=[],
    nargs="+",
    help="5 digit date directories to glob for maps to analyze. can use wildcard; i.e. 15* will glob all date dirs starting with 15",
)

P.add_argument(
    "-s",
    "--snr-threshold",
    help="SNR threshold for source finding",
    default=5.0,
    type=float,
)


P.add_argument(
    "--sim-transients",
    help="Inject transients into the maps.",
    action="store_true",
)

P.add_argument(
    "--pointing-flux-threshold",
    action="store",
    default="0.3 Jy",
    type=str,
    help="Flux threshold for pointing sources, json compatible quantity.",
)

P.add_argument(
    "--flux-threshold",
    action="store",
    default="10 mJy",
    type=str,
    help="Flux threshold for source extraction, json compatible quantity.",
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
    default=4,
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

args = P.parse_args()


def generate_slurm_header(jobname, groupname, cpu_per_task, script_dir, slurm_out_dir):
    slurm_header = f"""#!/bin/bash
#SBATCH --job-name={jobname}            # create a short name for your job
#SBATCH -A {groupname}                    # group name, by default simonsobs
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task={cpu_per_task}       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=24G         # memory per cpu-core (4G is default)
#SBATCH --time=04:59:00          # total run time limit (HH:MM:SS)
#SBATCH --output={slurm_out_dir}%x.out

module load soconda/3.11/v0.6.3

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
cd {script_dir}

source .venv/bin/activate

export socat_client_client_type=pickle
export socat_client_pickle_path=catmaker_090_3pass_socat.pickle

export MAPCAT_DEPTH_ONE_PARENT=/scratch/gpfs/SIMONSOBS/users/amfoster/so/lat_early_maps/
export MAPCAT_DATABASE_NAME=/scratch/gpfs/SIMONSOBS/users/amfoster/so/lat_early_maps/out_deep56/mapcat.sqlite

if [ ! -e "socat.pickle" ]; then
    socat-act-fits -f /scratch/gpfs/SIMONSOBS/users/amfoster/depth1_act_maps/inputs/catmaker_090_3pass_clean.fits  -o catmaker_090_3pass_socat.pickle
fi

"""
    return slurm_header


def generate_config_json(
    frequency: str | None = None,
    array: str | None = None,
    pointing_flux_threshold="0.3 Jy",
    flux_low_limit="0.01 Jy",
    thumbnail_half_width="0.2 deg",
    min_snr=5.0,
    min_pointing_sources=10,
    output_dir="./",
):
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
            "catalog_type": "socat",
            "flux_lower_limit": "{flux_low_limit}"
        }}
    ],
    "sso_catalogs": [
        {{
            "catalog_type": "sso",
            "db_path": "./sotrplib/solar_system/mpc_orbital_params_bright_asteroids.csv"
        }}
    ],
    "pointing_provider": {{
        "photometry_type": "scipy_pointing",
        "thumbnail_half_width": "{thumbnail_half_width}",
        "min_flux": "{pointing_flux_threshold}",
        "reproject_thumbnails": "True",
        "allowable_centroid_offset": "3.0 arcmin"
    }},
    "pointing_residual": {{
        "pointing_residual_type": "median",
        "min_snr": {min_snr},
        "min_sources": {min_pointing_sources}
    }},
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
    "source_subtractor": {{
            "subtractor_type": "photutils"
    }},
    "blind_search": {{
            "search_type": "photutils"
    }},
    "forced_photometry": {{
        "photometry_type": "lmfit",
        "reproject_thumbnails": "True",
        "flux_limit_centroid": "0.1 Jy",
        "thumbnail_half_width": "4 arcmin",
        "allowable_center_offset": "1.0 arcmin",
        "near_source_rel_flux_limit": 1.0
    }},
    "sifter": {{
        "sifter_type": "default",
        "min_match_radius": "5.0 arcmin"
    }},
    "outputs": [
        {{
            "output_type": "pickle",
            "directory": "{output_dir}"
        }}
    ]
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

n = 0


for band in args.bands:
    for tube in args.optics_tubes:
        config_file = args.slurm_script_dir + f"{tube}_{band}_config.json"
        with open(config_file, "w") as f:
            f.write(
                generate_config_json(
                    frequency=band,
                    array=tube,
                    pointing_flux_threshold=args.pointing_flux_threshold,
                    flux_low_limit=args.flux_threshold,
                    thumbnail_half_width=args.thumbnail_radius,
                    min_snr=args.snr_threshold,
                    output_dir=args.out_dir,
                )
            )
        slurm_text = generate_slurm_header(
            f"sotrp_{tube}_{band}_depth1",
            args.group_name,
            str(args.ncores),
            args.script_dir if args.script_dir else os.getcwd(),
            args.slurm_out_dir,
        )
        slurm_text += f"srun --overlap sotrp -c {config_file} > {args.out_dir}{tube}_{band}_sotrp.log "

        with open(f"{args.slurm_script_dir}/{tube}_{band}_sub.slurm", "w") as f:
            f.write(slurm_text)


print("#" * 50)
print("Slurm scripts saved to :", args.slurm_script_dir)
print("Slurm output saved to :", args.slurm_out_dir)

print("Output files saved to :", args.out_dir)

print(
    "To run the slurm jobs, point slurm_submitter.py to the slurm script location stated above."
)

print("#" * 50)
