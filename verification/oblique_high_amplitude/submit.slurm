#!/bin/bash

#SBATCH -J oblique # Job Name
#SBATCH -o log.out.%j          # name of stdout output  file (%j expands to job Id)
#SBATCH -e log.error
#SBATCH -N 2
#SBATCH -n 25
##SBATCH -p mem
#SBATCH -t 24:00:00          # run time (hh:mm:ss)
#SBATCH --mail-type=NONE    # mail alert
#SBATCH --mail-user=cagonzal@stanford.edu

# echo "Running glimPSE with: $cmd on $SLURM_JOB_NODELIST in directory "`pwd`

mpirun -n 25 python -u ~/glimPSE/glimPSE/src/main.py 

echo "Finished glimPSE run"
date
