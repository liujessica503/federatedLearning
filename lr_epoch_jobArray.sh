#!/bin/bash

#SBATCH --job-name=cv_jobArray
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1gb
#SBATCH --time=20:00:00
#SBATCH --account=stats_dept1
#SBATCH --partition=standard
#SBATCH --mail-user=liujess@umich.edu
#SBATCH --mail-type=FAIL
#SBATCH --output=lr_epochs_tune.%j_%a.out
#SBATCH --error=lr_epochs_tune.%j_%a.err

#SBATCH --array=0-9

# can make this go through one epoch, various learning rates
PARRAY=(0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19)
# get a unique name for the output
output_file=`expr $SLURM_ARRAY_TASK_ID`
p1=${PARRAY[`expr $SLURM_ARRAY_TASK_ID % ${#PARRAY[@]}`]}

# The application(s) to execute along with its input arguments and options:
# already installed keras and tensorflow-1.14.0
module load python3.6-anaconda
PYTHONHASHSEED=0 python3 cv_byLearnRate_jobArray.py fed_init_1cv2.json -p $p1 "[50]" cv2_fed_binary_narrowGrid_pt3_ $output_file
