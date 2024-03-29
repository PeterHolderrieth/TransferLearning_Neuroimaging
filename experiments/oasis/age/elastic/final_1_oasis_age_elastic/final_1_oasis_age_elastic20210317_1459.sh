#!/bin/bash
#$ -wd /users/win-fmrib-analysis/lhw539/TransferLearning_Neuroimaging/
#$ -P win.prjc
#$ -q win000
#$ -j y #Error and output file are merged to output file
#Save file to:
# Log locations which are relative to the current                                                                                                                                                                  # working directory of the submission
#$ -o experiments/oasis/age/elastic/final_1_oasis_age_elastic/final_1_oasis_age_elastic20210317_1459.log

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

python ~/TransferLearning_Neuroimaging/main.py -deb full -con experiments/oasis/age/elastic/final_1_oasis_age_elastic/final_1_oasis_age_elastic20210317_1459.json


echo "------------------------------------------------"
echo "Finished at: "`date`
echo "------------------------------------------------" 

