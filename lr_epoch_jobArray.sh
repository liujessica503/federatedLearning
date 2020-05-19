#!/bin/bash

#SBATCH --job-name=cv_jobArray
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2gb
#SBATCH --time=10:00:00
#SBATCH --account=stats_dept1
#SBATCH --partition=standard
#SBATCH --mail-user=liujess@umich.edu
#SBATCH --mail-type=FAIL
#SBATCH --output=lr_epochs_tune.%j_%a.out
#SBATCH --error=lr_epochs_tune.%j_%a.err

#SBATCH --array=0-20

# can make this go through one epoch, various learning rates
#PARRAY=(0.00005 0.00006 0.00007 0.00008 0.00009 0.0001 0.0002 0.0003 0.0004 0.0005 0.0006 0.0007 0.0008 0.0009 0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.009 0.01 0.011 0.012 0.013 0.014 0.015 0.016 0.017 0.018 0.019 0.02)    
PARRAY=(0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25)
# get a unique name for the output
output_file=`expr $SLURM_ARRAY_TASK_ID`
p1=${PARRAY[`expr $SLURM_ARRAY_TASK_ID % ${#PARRAY[@]}`]}

# The application(s) to execute along with its input arguments and options:
# already installed keras and tensorflow-1.14.0
module load python3.6-anaconda
# global_init_1cv2 is regression
# global_init_1cv2_binary is binary
PYTHONHASHSEED=0 python3 cv_byLearnRate_jobArray.py fed_init_adam_1cv2.json -p $p1 "[5, 10, 15, 20, 25]" cv2_fed_regression_adam_narrowGrid_ $output_file
