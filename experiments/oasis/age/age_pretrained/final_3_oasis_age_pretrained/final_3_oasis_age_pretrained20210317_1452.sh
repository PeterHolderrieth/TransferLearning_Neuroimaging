#!/bin/bash
#$ -wd /users/win-fmrib-analysis/lhw539/TransferLearning_Neuroimaging/
#$ -P win.prjc
#$ -q gpu8.q
#$ -j y #Error and output file are merged to output file
#$ -l gpu=2
#$ -pe shmem 2 #Should be the same as the number of GPUs 
#$ -l gputype=p100
#Save file to:
# Log locations which are relative to the current                                                                                                                                                                  # working directory of the submission
#$ -o experiments/oasis/age/age_pretrained/final_3_oasis_age_pretrained/final_3_oasis_age_pretrained20210317_1452.log

echo "------------------------------------------------"
echo "Job ID: $JOB_ID"
echo "Run on host: "`hostname`
echo "Operating system: "`uname -s`
echo "Username: "`whoami`
echo "Started at: "`date`
echo "------------------------------------------------"

#Load Python module:
module load Python/3.7.4-GCCcore-8.3.0

#Activate the correct python environment:
source ~/python/ccpu_py_tlneuro

python ~/TransferLearning_Neuroimaging/main.py -deb full -con experiments/oasis/age/age_pretrained/final_3_oasis_age_pretrained/final_3_oasis_age_pretrained20210317_1452.json


echo "------------------------------------------------"
echo "Finished at: "`date`
echo "------------------------------------------------" 
