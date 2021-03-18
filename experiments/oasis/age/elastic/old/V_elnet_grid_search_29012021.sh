#!/bin/bash
#$ -wd /users/win-fmrib-analysis/lhw539/TransferLearning_Neuroimaging/experiments/elnet_grid_search/results/
#$ -P win.prjc
#$ -q short.qe
#$ -j y #Error and output file are merged to output file
#Save file to:
# Log locations which are relative to the current                                                                                                                                                                  # working directory of the submission
#$ -o V_elnet_grid_search.log

echo "------------------------------------------------"
echo "Job ID: $JOB_ID"
echo "Run on host: "`hostname`
echo "Operating system: "`uname -s`
echo "Username: "`whoami`
echo "Started at: "`date`
echo "------------------------------------------------"

#Load Python module:
#module load Python/3.7.4-GCCcore-8.3.0

#Activate the correct python environment:
#source ~/python/ccpu_py_tlneuro

#Change to U: increase number of features.


python ~/TransferLearning_Neuroimaging/elastic.py \
-deb full \
-batch 500 \
-reg 1. \
-l1rat 0.5 \
-feat 120	


echo "------------------------------------------------"
echo "Finished at: "`date`
echo "------------------------------------------------"
