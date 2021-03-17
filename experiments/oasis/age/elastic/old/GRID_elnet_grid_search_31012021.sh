#!/bin/bash
#$ -wd /users/win-fmrib-analysis/lhw539/TransferLearning_Neuroimaging/experiments/elnet_grid_search/results/
#$ -P win.prjc
#$ -q short.qe
#$ -j y #Error and output file are merged to output file
#Save file to:
# Log locations which are relative to the current                                                                                                                                                                  # working directory of the submission
#$ -o T_elnet_grid_search.log

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

for reg in 0.1 0.2 0.4 1. 2. 4.
do 
    for l1rat in 0.1 0.2 0.5 0.7 0.9 
    do
        for n_feat in 60 80 100 120 140 160 200 300
        do
            echo "------------------------------------------------"
            python ~/TransferLearning_Neuroimaging/elastic.py \
                -deb full \
                -batch 500 \
                -reg $reg \
                -l1rat $l1rat \
                -feat $n_feat	
            echo "------------------------------------------------"
        done
    done

done 



echo "------------------------------------------------"
echo "Finished at: "`date`
echo "------------------------------------------------"

