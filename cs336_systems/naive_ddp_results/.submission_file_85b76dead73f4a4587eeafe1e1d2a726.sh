#!/bin/bash

# Parameters
#SBATCH --error=/home/c-cye/assignment2-systems/cs336_systems/naive_ddp_results/%j_0_log.err
#SBATCH --gpus-per-task=2
#SBATCH --job-name=naive_ddp_benchmark
#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --output=/home/c-cye/assignment2-systems/cs336_systems/naive_ddp_results/%j_0_log.out
#SBATCH --partition=a2
#SBATCH --qos=a2-qos
#SBATCH --signal=USR2@90
#SBATCH --time=10
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /home/c-cye/assignment2-systems/cs336_systems/naive_ddp_results/%j_%t_log.out --error /home/c-cye/assignment2-systems/cs336_systems/naive_ddp_results/%j_%t_log.err /home/c-cye/assignment2-systems/.venv/bin/python3 -u -m submitit.core._submit /home/c-cye/assignment2-systems/cs336_systems/naive_ddp_results
