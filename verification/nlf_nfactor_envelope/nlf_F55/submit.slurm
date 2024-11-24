#!/bin/bash

#SBATCH -J nlfpse # Job Name
#SBATCH -o log.out          # name of stdout output  file (%j expands to job Id)
#SBATCH -e log.error
#SBATCH -N 1
#SBATCH -n 2
##SBATCH -p mem
#SBATCH -t 24:00:00          # run time (hh:mm:ss)
#SBATCH --mail-type=NONE    # mail alert
#SBATCH --mail-user=cagonzal@stanford.edu

# echo "Running glimPSE with: $cmd on $SLURM_JOB_NODELIST in directory "`pwd`

mpirun -n 2 python -u ~/glimPSE/glimPSE/src/main.py 

echo "Finished glimPSE run"
date
