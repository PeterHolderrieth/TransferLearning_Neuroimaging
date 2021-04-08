#!/bin/bash
#$ -wd /users/win-fmrib-analysis/lhw539/TransferLearning_Neuroimaging/
#$ -P win.prjc
#$ -q win000
#$ -j y #Error and output file are merged to output file
#$ -l gpu=0
#$ -pe shmem 0 #Should be the same as the number of GPUs 
#$ -l gputype=p100
#Save file to:
# Log locations which are relative to the current                                                                                                                                                                  # working directory of the submission
#$ -o experiments/oasis/progmci/elastic/run_1_final_oasis_progmci_elastic/run_1_final_oasis_progmci_elastic20210407_1648.log

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

python ~/TransferLearning_Neuroimaging/main.py -deb full -con experiments/oasis/progmci/elastic/run_1_final_oasis_progmci_elastic/run_1_final_oasis_progmci_elastic20210407_1648.json


echo "------------------------------------------------"
echo "Finished at: "`date`
echo "------------------------------------------------" 

