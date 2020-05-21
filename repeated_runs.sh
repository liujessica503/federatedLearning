#!/bin/bash

#SBATCH --job-name=repeatedRunsfedBinAdamCv2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1200mb
#SBATCH --time=1:00:00
#SBATCH --account=stats_dept1
#SBATCH --partition=standard
#SBATCH --mail-user=liujess@umich.edu
#SBATCH --mail-type=FAIL
#SBATCH --output=repeatedRunsfedBinAdamCv2.%j_%a.out

#SBATCH --array=0-14

# each job uses a different seed
PARRAY=(360 361 362 363 364 365 366 367 368 369 370 371 372 373 374)
# get a unique name for the output
output_file=`expr $SLURM_ARRAY_TASK_ID`
p1=${PARRAY[`expr $SLURM_ARRAY_TASK_ID % ${#PARRAY[@]}`]}

# The application(s) to execute along with its input arguments and options:
# already installed keras and tensorflow-1.14.0
module load python3.6-anaconda
# virtual environment with the correct installed versions of libraries
source activate fedModelSetup
PYTHONHASHSEED=123456 python3 single_experiment_jobArray.py fed_initaBinAdamCv2.json -p $p1 cv2_fed_binary_adam_narrowGrid_newLocalUpdates $output_file
