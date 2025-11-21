import argparse as ap
import datetime
import os
from getpass import getuser
from glob import glob

from tqdm import tqdm

USER = getuser()

P = ap.ArgumentParser(
    description="Testing functionality of a depth1 historical lightcurve extractor.",
    formatter_class=ap.ArgumentDefaultsHelpFormatter,
)

P.add_argument(
    "--obsids",
    action="store",
    default=[],
    nargs="+",
    help="Observations to analyze, i.e. 1573251992.1573261318.ar5 ... not implemented yet... ",
)

P.add_argument(
    "--time-window",
    action="store",
    default=5.0,
    type=float,
    help="Time around conjunction in which to return data, days. If utc_time in ps_csv this is +- around that time. If obsids given, this is ignored.",
)
P.add_argument("--ra", nargs="+", default=[], help="Source RA (deg).")

P.add_argument("--dec", nargs="+", default=[], help="Source DEC (deg).")

P.add_argument(
    "--ps-csv",
    action="store",
    default=None,
    help="Path to the point source CSV file containing columns ra,dec which are in decimal degrees.",
)

P.add_argument(
    "--data-dir",
    action="store",
    default="/scratch/gpfs/SIMONSOBS/so/maps/actpol/depth1/",
    help="Data directory, where the depth1 maps live.",
)

P.add_argument(
    "--scratch-dir",
    action="store",
    default=f"/scratch/gpfs/SIMONSOBS/users/{USER}/scratch/",
    help="Location of scratch directory for users. Default: /scratch/gpfs/SIMONSOBS/users/[user]/scratch/.",
)

P.add_argument(
    "--run-dir",
    action="store",
    default="",
    help="Directory in which the lightcurve extractor script lives. If default, then will use os.cwd() which assumes this script lives in the same directory.",
)

P.add_argument(
    "--script-name",
    action="store",
    default="scripts/historical_lightcurve_extractor/act_depth1_lightcurve_extractor.py",
    help="Filename of the lightcurve extractor script.",
)

P.add_argument(
    "--out-dir",
    action="store",
    default="",
    help="Output directory for lightcurves. If empty, will create directory in scratch_dir.",
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
    default=False,
    help="Save thumbnail maps around the source(s). ",
)

P.add_argument(
    "--thumbnail-radius",
    action="store",
    default=0.2,
    type=float,
    help="Thumbnail radius (half-width of square map), in deg. ",
)

P.add_argument(
    "--ncores",
    action="store",
    default=24,
    type=int,
    help="Number of cores to request for each slurm job. ",
)

P.add_argument(
    "--nserial",
    action="store",
    default=3,
    help="Number of serial sets of `ncores` scripts to run per job. ",
)

P.add_argument(
    "--soconda-version",
    action="store",
    default=None,
    help="Version of soconda to load. If None uses running version.",
)

args = P.parse_args()


def get_soconda_module():
    import subprocess

    cmd = "module list 2>&1 | grep soconda | grep -o 'soconda[^ ]*'"
    SOCONDA_MODULE = subprocess.check_output(cmd, shell=True, text=True).strip()
    return SOCONDA_MODULE


def generate_slurm_header(
    jobname, groupname, cpu_per_task, run_dir, slurm_out_dir, soconda_version
):
    slurm_header = f"""#!/bin/bash
#SBATCH --job-name={jobname}            # create a short name for your job
#SBATCH -A {groupname}                    # group name, by default simonsobs
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task={cpu_per_task}       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time=04:59:00          # total run time limit (HH:MM:SS)
#SBATCH --output={slurm_out_dir}%x.out
module load {soconda_version}

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
cd {run_dir}

"""
    return slurm_header


nserial = args.ncores * args.nserial

datelist = []
if len(args.obsids) == 0:
    datelist = sorted(glob(args.data_dir + "*"))
elif len(args.obsids) == 1:
    datelist = sorted(glob(args.data_dir + args.obsids[0]))
else:
    for o in args.obsids:
        datelist += sorted(glob(args.data_dir + o))

print("There are ", len(datelist), " date directories total")

## make scratch output directory
user_scratch = args.scratch_dir
if not os.path.exists(user_scratch):
    os.mkdir(user_scratch)

if not args.soconda_version:
    args.soconda_version = get_soconda_module()

if not args.out_dir:
    ## get random tmp dir
    import random
    import string

    letters = string.ascii_lowercase
    random_string = "".join(random.choice(letters) for i in range(6))

    args.out_dir = user_scratch + f"tmp_lightcurve_{random_string}/"
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
    args.run_dir if args.run_dir else os.getcwd(),
    args.slurm_out_dir,
    args.soconda_version,
)
event_times = []
if args.ps_csv is not None:
    import pandas as pd

    ps_df = pd.read_csv(args.ps_csv).rename(columns=str.lower)
    args.ra += [str(r) for r in ps_df["ra"].values]
    args.dec += [str(d) for d in ps_df["dec"].values]
    if "utc_time" in ps_df.columns:
        event_times += [str(t) for t in ps_df["utc_time"].values]
    print(f"Loaded {len(ps_df['ra'].values)} sources from {args.ps_csv}")

for i in tqdm(range(len(datelist))):
    date = datelist[i].split("/")[-1]
    globstr = args.data_dir + date + "/depth1*rho.fits"
    lc_outfile = date + "_tmp_lightcurve.txt"
    thumb_outfile = date + "_tmp_thumbnail.hdf5"
    ras_in_daterange = []
    decs_in_daterange = []
    if event_times:
        # date is first 5 digits of unix time
        mid_unix_date = date + "50000"
        for j in range(len(event_times)):
            if (
                abs(
                    datetime.datetime.strptime(
                        event_times[j], "%Y-%m-%d %H:%M:%S.%f"
                    ).timestamp()
                    - float(mid_unix_date)
                )
                < args.time_window * 86400
            ):
                ras_in_daterange.append(args.ra[j])
                decs_in_daterange.append(args.dec[j])
    else:
        ras_in_daterange = args.ra
        decs_in_daterange = args.dec
    if len(ras_in_daterange) == 0:
        continue
    slurm_text += f"""srun --overlap python {args.script_name} --ra {" ".join(ras_in_daterange)} --dec {" ".join(decs_in_daterange)} --rho-maps {globstr} --odir {args.out_dir} --scratch-dir {user_scratch} -o {lc_outfile} --output-thumbnail-fname {thumb_outfile} --thumbnail-radius {args.thumbnail_radius}{" --save-thumbnails" if args.save_thumbnails else ""} &\n sleep 1 \n"""

    if i % args.ncores == 0 and i > 0:
        slurm_text += "wait\n"

    if i % nserial == 0 and i > 0:
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
            args.run_dir if args.run_dir else os.getcwd(),
            args.slurm_out_dir,
            args.soconda_version,
        )

with open("%s/%s_sub.slurm" % (args.slurm_script_dir, str(n).zfill(4)), "w") as f:
    f.write(slurm_text)
    f.write("wait")

print("#" * 50)
print("Slurm scripts saved to :", args.slurm_script_dir)
print("Slurm output saved to :", args.slurm_out_dir)

print("Lightcurve files saved to :", args.out_dir)

print(
    "To run the slurm jobs, point slurm_submitter.py to the slurm script location stated above."
)

print(
    "Once run, the lightcurve collator can be pointed to the lightcurve output directory."
)
print("#" * 50)
