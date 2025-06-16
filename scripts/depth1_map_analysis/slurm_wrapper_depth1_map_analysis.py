from glob import glob
from tqdm import tqdm
from getpass import getuser
import os 


USER = getuser()

import argparse as ap
P = ap.ArgumentParser(description="Testing functionality of a depth1 source finding.",
                      formatter_class=ap.ArgumentDefaultsHelpFormatter,
                     )

P.add_argument("--maps",
               action="store",
               default=[],
               nargs='+',
               help="Depth1 maps to analyze. Will null anything in date-dirs and only analyze these maps."
              )

P.add_argument("--date-dirs",
               action="store",
               default=[],
               nargs='+',
               help="5 digit date directories to glob for maps to analyze. can use wildcard; i.e. 15* will glob all date dirs starting with 15"
              )

P.add_argument("-s", 
                "--snr-threshold", 
                help="SNR threshold for source finding", 
                default=5.0,
                type=float
                )

P.add_argument("--data-dir",
               action="store",
               default='/scratch/gpfs/SIMONSOBS/so/maps/actpol/depth1/',
               help="Data directory, where the depth1 maps live."
              )

P.add_argument("--simulated-transient-database", 
                help="Path to transient event database, generated using the sims module.", 
                action='store',
                default='/scratch/gpfs/SIMONSOBS/users/amfoster/scratch/act_depth1_transient_sim_db.db',
                )

P.add_argument("--sim-transients", 
                help="Inject transients into the maps.", 
                action='store_true',
              )

P.add_argument("--flux-threshold",
               action="store",
               default=0.03,
               type=float,
               help="Flux threshold for source extraction, in Jy. Default: 0.03"
              ) 

P.add_argument("--scratch-dir",
               action="store",
               default=f'/scratch/gpfs/SIMONSOBS/users/{USER}/scratch/',
               help="Location of scratch directory for users. Default: /scratch/gpfs/SIMONSOBS/users/[user]/scratch/."
              )

P.add_argument("--script-dir",
               action="store",
               default='',
               help="Directory in which the trp_pipeline script lives. If default, then will use os.cwd() which assumes this script lives in the same directory."
              )

P.add_argument("--script-name",
               action="store",
               default='trp_pipeline.py',
               help="Filename of the analysis script."
              )

P.add_argument("--out-dir",
               action="store",
               default='',
               help="Output directory for source json files. If empty, will create directory in scratch_dir."
              )

P.add_argument("--group-name",
               action="store",
               default='simonsobs',
               help="Name of computing cluster group that you belong to, for running slurm jobs."
              )

P.add_argument("--slurm-script-dir",
               action="store",
               default='slurm_job_scripts/',
               help="Directory in which the slurm job scripts lives."
              )

P.add_argument("--slurm-out-dir",
               action="store",
               default='slurm_output_files/',
               help="Directory in which the slurm output files get saved. If default, appends to scratch dir. "
              )

P.add_argument("--plot-thumbnails",
               action="store_true",
               default=False,
               help="Plot thumbnail maps around the transient sources. "
              )

P.add_argument("--thumbnail-radius",
               action="store",
               default=0.5,
               type=float,
               help="Thumbnail radius (half-width of square map), in deg. "
              )
P.add_argument("--coadd-n-days",
               action="store",
               default=0,
               type=int,
               help="Number of days to coadd maps over."
              )

P.add_argument("--ncores",
               action="store",
               default=12,
               type=int,
               help="Number of cores to request for each slurm job. "
              )

P.add_argument("--nserial",
               action="store",
               default=12,
               help="Number of serial sets of `ncores` scripts to run per job. "
              )

P.add_argument("--bands",
               action="store",
               nargs='+',
               default=['f090','f150','f220'],
               help="Bands to analyze, default is f090,f150,f220 but can also includ f030 or f040. "
              )

args = P.parse_args()

def generate_slurm_header(jobname,
                          groupname,
                          cpu_per_task,
                          script_dir,
                          slurm_out_dir
                          ):
    slurm_header = f'''#!/bin/bash
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

'''
    return slurm_header


nserial=args.ncores*args.nserial

datelist=[]
if not args.maps:
    if len(args.date_dirs)==0:
        datelist=sorted(glob(args.data_dir+'*'))
    elif len(args.date_dirs)==1:
        datelist=sorted(glob(args.data_dir+args.date_dirs[0]))
    else:
        for o in args.date_dirs:
            datelist += sorted(glob(args.data_dir+o))

print('There are ',len(datelist),' date directories total')

## make scratch output directory
user_scratch = args.scratch_dir
if not os.path.exists(user_scratch):
    os.mkdir(user_scratch)

if not args.out_dir:
    ## get random tmp dir
    import string
    import random
    letters = string.ascii_lowercase
    random_string = ''.join(random.choice(letters) for i in range(6))

    args.out_dir = user_scratch+f'tmp_sourcedir_{random_string}/'
    print('Creating temp output directory: ',args.out_dir)

## set slurm output dir to live in the temp directory
if args.slurm_out_dir == P.get_default('slurm_out_dir'):
    args.slurm_out_dir = args.out_dir+args.slurm_out_dir
if args.slurm_script_dir == P.get_default('slurm_script_dir'):
    args.slurm_script_dir = args.out_dir+args.slurm_script_dir

if not os.path.exists(args.out_dir):
    os.mkdir(args.out_dir)
if not os.path.exists(args.slurm_script_dir):
    os.mkdir(args.slurm_script_dir)

n=0
slurm_text = generate_slurm_header(str(n).zfill(4),
                                   args.group_name,
                                   str(args.ncores),
                                   args.script_dir if args.script_dir else os.getcwd(),
                                   args.slurm_out_dir
                                  )

nmaps = 0
for mapfile in args.maps:
    obsid = mapfile.split('/')[-1].split('_')[1]
    dateid = mapfile.split('/')[-2]
    slurm_text+=(
                f"srun --overlap python {args.script_name} --maps {mapfile} "
                f"--output-dir {args.out_dir} -s {args.snr_threshold}"
                f"{' --plot-thumbnails' if args.plot_thumbnails else ''}"
                f" --flux-threshold {args.flux_threshold}"
                f" {f'--simulated-transient-database {args.simulated_transient_database}' if args.sim_transients else ''}"
                f"{f' --inject-transients' if args.sim_transients else ''}"
                f" --coadd-n-days {args.coadd_n_days}"
                f" &\n sleep 1 \n"
                )
    nmaps+=1
    if nmaps%args.ncores == 0 and nmaps>0:
        slurm_text+='wait\n'

    if nmaps%nserial == 0 and nmaps>0:
        with open('%s/%s_sub.slurm'%(args.slurm_script_dir,str(n).zfill(4)),'w') as f:
            f.write(slurm_text)
            f.write('wait')
        n+=1
        slurm_text = generate_slurm_header(str(n).zfill(4),
                                           args.group_name,
                                           str(args.ncores),
                                           args.script_dir if args.script_dir else os.getcwd(),
                                           args.slurm_out_dir
                                          )

for i in tqdm(range(len(datelist))):
    date=datelist[i].split('/')[-1]
    for band in args.bands:
        globstr=args.data_dir+date+f'/depth1*{band}*_rho.fits'
        slurm_text+=(
                    f"srun --overlap python {args.script_name} --maps {globstr} "
                    f"--output-dir {args.out_dir} -s {args.snr_threshold}"
                    f"{' --plot-thumbnails' if args.plot_thumbnails else ''}"
                    f" --flux-threshold {args.flux_threshold}"
                    f" --coadd-n-days {args.coadd_n_days}"
                    f" {f'--simulated-transient-database {args.simulated_transient_database}' if args.sim_transients else ''}"
                    f"{f' --inject-transients' if args.sim_transients else ''}"
                    f" &\n sleep 1 \n"
                    )
        if i%args.ncores == 0 and i>0:
            slurm_text+='wait\n'

        if i%nserial == 0 and i>0:
            with open('%s/%s_sub.slurm'%(args.slurm_script_dir,str(n).zfill(4)),'w') as f:
                f.write(slurm_text)
                f.write('wait')
            n+=1
            slurm_text = generate_slurm_header(str(n).zfill(4),
                                            args.group_name,
                                            str(args.ncores),
                                            args.script_dir if args.script_dir else os.getcwd(),
                                            args.slurm_out_dir
                                            )
        
with open('%s/%s_sub.slurm'%(args.slurm_script_dir,str(n).zfill(4)),'w') as f:
    f.write(slurm_text)
    f.write('wait')

print('#'*50)
print('Slurm scripts saved to :',args.slurm_script_dir)
print('Slurm output saved to :',args.slurm_out_dir)

print('Photometry files saved to :',args.out_dir)

print('To run the slurm jobs, point slurm_submitter.py to the slurm script location stated above.')

print('#'*50)