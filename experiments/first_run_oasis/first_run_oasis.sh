#!/bin/bash
#$ -wd /users/win-fmrib-analysis/lhw539/TransferLearning_Neuroimaging/experiments/first_run_oasis/
#$ -P win.prjc
#$ -q gpu8.q
#$ -pe shmem 2
#$ -l gpu=2
#$ -l gputype=p100
# Log locations which are relative to the current                                                                                                                                                                  # working directory of the submission
###$ -o output.log
###$ -e error.log   
echo "------------------------------------------------"
echo "Run on host: "`hostname`
echo "Operating system: "`uname -s`
echo "Username: "`whoami`
echo "Started at: "`date`
echo "------------------------------------------------"

#Load Python module:
module load Python/3.7.4-GCCcore-8.3.0

#Activate the correct python environment:
source ~/python/ccpu_py_tlneuro
pip --version

python ~/TransferLearning_Neuroimaging/train.py -epochs 3 	

#qsub -l gpu=2 -l gputype=p100 -pe shmem 2 ~/TransferLearning_Neuroimaging/experiments/f
