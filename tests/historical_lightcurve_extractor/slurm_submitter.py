from glob import glob
import subprocess as sp
import time
from tqdm import tqdm

import argparse as ap
P = ap.ArgumentParser(description="Submit all slurm scripts within a directory.",
                      formatter_class=ap.ArgumentDefaultsHelpFormatter,
                     )

P.add_argument("-d",
               "--dir",
               action="store",
               default='slurm_job_scripts/',
               help="Directory where the slurm job scripts live. "
              )

args = P.parse_args()


slurm_files = glob(args.dir+'*.slurm')

sleeptime = 0.1
j0=0
for i in tqdm(range(len(slurm_files)),desc='Submitting slurm jobs'):
    sp.run(['sbatch',slurm_files[i]])
    time.sleep(sleeptime)
