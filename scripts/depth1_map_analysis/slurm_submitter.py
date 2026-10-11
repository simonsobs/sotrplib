import argparse as ap
import subprocess as sp
import time
from glob import glob

from tqdm import tqdm

P = ap.ArgumentParser(
    description="Submit all slurm scripts within a directory.",
    formatter_class=ap.ArgumentDefaultsHelpFormatter,
)

P.add_argument(
    "-d",
    "--dir",
    action="store",
    default="slurm_job_scripts/",
    help="Directory where the slurm job scripts live. ",
)

P.add_argument(
    "--stagger-minutes",
    action="store",
    default=0.0,
    type=float,
    help="Delay the earliest start of job i by i * stagger-minutes (sbatch --begin), "
    "so that jobs that write to the same database do not all start together. 0: no delay.",
)

P.add_argument(
    "--skip",
    action="store",
    nargs="+",
    default=[],
    help="Do not submit the slurm scripts whose file name contains one of these strings.",
)

args = P.parse_args()


slurm_files = sorted(
    f for f in glob(args.dir + "*.slurm") if not any(s in f for s in args.skip)
)

sleeptime = 0.1
for i in tqdm(range(len(slurm_files)), desc="Submitting slurm jobs"):
    cmd = ["sbatch"]
    if args.stagger_minutes > 0:
        cmd.append(f"--begin=now+{round(i * args.stagger_minutes * 60)}")
    sp.run(cmd + [slurm_files[i]])
    time.sleep(sleeptime)
