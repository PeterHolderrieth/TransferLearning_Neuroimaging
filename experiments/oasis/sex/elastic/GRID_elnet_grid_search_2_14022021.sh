#!/bin/bash
#$ -wd /users/win-fmrib-analysis/lhw539/TransferLearning_Neuroimaging/experiments/sex/elnet/
#$ -P win.prjc
#$ -q short.qe
#$ -j y #Error and output file are merged to output file
#Save file to:
# Log locations which are relative to the current                                                                                                                                                                  # working directory of the submission
#$ -o results/Grid_search_sex_2.log

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

for reg in 0.01 0.05 0.1 0.15 
do 
    for l1rat in 0.3 0.4 0.5 0.6  
    do
        for n_feat in 5 10 20 30 50 60 70 
        do
            echo "------------------------------------------------"
            python ~/TransferLearning_Neuroimaging/elastic.py \
                -deb full \
                -batch 500 \
                -reg $reg \
                -l1rat $l1rat \
                -feat $n_feat  \
                -task sex	
            echo "------------------------------------------------"
        done
    done

done 



echo "------------------------------------------------"
echo "Finished at: "`date`
echo "------------------------------------------------"

