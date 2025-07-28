def generate_slurm_header(
    jobname,
    groupname: str = "simonsobs",
    nodes: int = 1,
    ntasks: int = 1,
    cpu_per_task: int = 1,
    mem_per_cpu: int = 4,  # GB
    time: str = "01:00:00",  # "hh:mm:ss"
    script_dir: str = "./",
    slurm_out_dir: str = "./",
):
    """
    All the usual slurm header info
    mem_per_cpu is in GB

    output file is saved to slurm_out_dir with the name [jobname].out

    """
    slurm_text = f"""#!/bin/bash
#SBATCH --job-name={jobname}
#SBATCH -A {groupname}
#SBATCH --nodes={nodes}
#SBATCH --ntasks={ntasks}
#SBATCH --cpus-per-task={cpu_per_task}
#SBATCH --mem-per-cpu={mem_per_cpu}G
#SBATCH --time={time}
#SBATCH --output={slurm_out_dir}%x.out
module load soconda/3.10/20241017

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
cd {script_dir}
"""
    return slurm_text
