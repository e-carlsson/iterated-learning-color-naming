#!/bin/bash -l
#SBATCH -A SLURM PROJECT NUMBER GOES HERE
#SBATCH -n 1
#SBATCH --output=exp.out
#SBATCH -t 00:50:00

module load Anaconda/2021.05-nsc1
conda activate myenv
srun --unbuffered python3 $1 $2 $3 $4
