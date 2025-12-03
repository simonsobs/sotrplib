import argparse as ap
import os
from getpass import getuser
from glob import glob

from tqdm import tqdm

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
    "--data-dir",
    action="store",
    default="/scratch/gpfs/SIMONSOBS/users/amfoster/so/lat_early_maps/out_deep56/",
    help="Data directory, where the depth1 maps live.",
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
    default="0.2 deg",
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
    default=4,
    type=int,
    help="Number of serial sets of `ncores` scripts to run per job. ",
)

P.add_argument(
    "--bands",
    action="store",
    nargs="+",
    default=["f090", "f150", "f220"],
    help="Bands to analyze, default is f090,f150,f220 but can also include f030 or f040. ",
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

"""
    return slurm_header


def generate_config_json(
    mapfile,
    pointing_flux_threshold="0.3 Jy",
    flux_low_limit="0.01 Jy",
    thumbnail_half_width="0.2 deg",
    min_snr=5.0,
    min_pointing_sources=10,
    poly_order=3,
    output_dir="./",
):
    ivar_map_file = mapfile.replace("_map.fits", "_ivar.fits")
    time_map_file = mapfile.replace("_map.fits", "_time.fits")
    info_file = mapfile.replace("_map.fits", "_info.hdf")
    freq = mapfile.split("_")[-2]
    arr = mapfile.split("_")[-3]
    config_text = f"""{{
    "maps": [
        {{
            "map_type": "inverse_variance",
            "intensity_map_path": "{mapfile}",
            "weights_map_path": "{ivar_map_file}",
            "time_map_path": "{time_map_file}",
            "info_path": "{info_file}",
            "frequency": "{freq}",
            "array": "{arr}",
            "intensity_units": "uK"
        }}
    ],
    "source_catalogs": [
        {{
            "catalog_type": "fits",
            "path": "/scratch/gpfs/SIMONSOBS/users/amfoster/depth1_act_maps/inputs/PS_S19_f090_2pass_optimalCatalog.fits",
            "flux_lower_limit": "{flux_low_limit}"
        }}
    ],
    "pointing_provider": {{
        "photometry_type": "scipy_pointing",
        "thumbnail_half_width": "{thumbnail_half_width}",
        "min_flux": "{pointing_flux_threshold}",
        "reproject_thumbnails": "True"
    }},
    "pointing_residual": {{
        "pointing_residual_type": "polynomial",
        "min_snr": {min_snr},
        "min_sources": {min_pointing_sources},
        "sigma_clip_level": 3.0,
        "polynomial_order": {poly_order}
    }},
    "preprocessors": [
        {{
            "preprocessor_type": "matched_filter"
        }},
        {{
            "preprocessor_type": "kappa_rho"
        }},
        {{
            "preprocessor_type": "edge_mask",
            "mask_on": "inverse_variance",
            "edge_width": "10.0 arcmin"
        }}
    ],
    "postprocessors": [
       {{
            "postprocessor_type": "flatfield",
            "sigma_val": 5.0,
            "tile_size": "1.0 deg"
        }}
    ],
    "source_subtractor": {{
            "subtractor_type": "photutils"
    }},
    "blind_search": {{
            "search_type": "photutils"
    }},
    "forced_photometry": {{
        "photometry_type": "scipy",
        "reproject_thumbnails": "True",
         "flux_limit_centroid": "0.1 Jy"
    }},
    "sifter": {{
        "sifter_type": "default"
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

datelist = []
if not args.maps:
    if len(args.date_dirs) == 0:
        datelist = sorted(glob(args.data_dir + "*"))
    elif len(args.date_dirs) == 1:
        datelist = sorted(glob(args.data_dir + args.date_dirs[0]))
    else:
        for o in args.date_dirs:
            datelist += sorted(glob(args.data_dir + o))

print("There are ", len(datelist), " date directories total")

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
slurm_text = generate_slurm_header(
    str(n).zfill(4),
    args.group_name,
    str(args.ncores),
    args.script_dir if args.script_dir else os.getcwd(),
    args.slurm_out_dir,
)

nmaps = 0
for mapfile in args.maps:
    obsid = mapfile.split("/")[-1].split("_")[1]
    dateid = mapfile.split("/")[-2]
    mapname = mapfile.split("/")[-1].replace(".fits", "")
    config_file = args.slurm_script_dir + f"{mapname}_config.json"
    with open(config_file, "w") as f:
        f.write(
            generate_config_json(
                mapfile,
                pointing_flux_threshold=args.pointing_flux_threshold,
                flux_low_limit=args.flux_threshold,
                thumbnail_half_width=args.thumbnail_radius,
                min_snr=args.snr_threshold,
                output_dir=args.out_dir,
            )
        )
    slurm_text += f"srun --overlap sotrp -c {config_file}  &\n sleep 1 \n"
    nmaps += 1
    if nmaps % args.ncores == 0 and nmaps > 0:
        slurm_text += "wait\n"

    if nmaps % nserial == 0 and nmaps > 0:
        with open(
            "%s/%s_sub.slurm" % (args.slurm_script_dir, str(n).zfill(4)), "w"
        ) as f:
            f.write(slurm_text)
            f.write("wait")
        n += 1
        slurm_text = generate_slurm_header(
            str(n).zfill(4),
            args.group_name,
            str(args.ncores),
            args.script_dir if args.script_dir else os.getcwd(),
            args.slurm_out_dir,
        )

for i in tqdm(range(len(datelist))):
    date = datelist[i].split("/")[-1]
    for band in args.bands:
        globstr = args.data_dir + date + f"/depth1*{band}*_map.fits"
        mapfiles = sorted(glob(globstr))
        for mapfile in sorted(mapfiles):
            mapname = mapfile.split("/")[-1].replace(".fits", "")
            config_file = args.slurm_script_dir + f"{mapname}_config.json"
            with open(config_file, "w") as f:
                f.write(
                    generate_config_json(
                        mapfile,
                        pointing_flux_threshold=args.pointing_flux_threshold,
                        flux_low_limit=args.flux_threshold,
                        thumbnail_half_width=args.thumbnail_radius,
                        min_snr=args.snr_threshold,
                        output_dir=args.out_dir,
                    )
                )
            slurm_text += (
                f"srun --overlap sotrp -c {config_file} > {args.out_dir}{mapname}_sotrp.log "
                f" &\n sleep 1 \n"
            )
            nmaps += 1
            if nmaps % args.ncores == 0 and nmaps > 0:
                slurm_text += "wait\n"
            if nmaps % nserial == 0 and nmaps > 0:
                with open(
                    "%s/%s_sub.slurm" % (args.slurm_script_dir, str(n).zfill(4)), "w"
                ) as f:
                    f.write(slurm_text)
                    f.write("wait")
                n += 1
                slurm_text = generate_slurm_header(
                    str(n).zfill(4),
                    args.group_name,
                    str(args.ncores),
                    args.script_dir if args.script_dir else os.getcwd(),
                    args.slurm_out_dir,
                )


with open("%s/%s_sub.slurm" % (args.slurm_script_dir, str(n).zfill(4)), "w") as f:
    f.write(slurm_text)
    f.write("wait")

print("#" * 50)
print("Slurm scripts saved to :", args.slurm_script_dir)
print("Slurm output saved to :", args.slurm_out_dir)

print("Output files saved to :", args.out_dir)

print(
    "To run the slurm jobs, point slurm_submitter.py to the slurm script location stated above."
)

print("#" * 50)
