#!/bin/bash
#$ -wd /users/win-fmrib-analysis/lhw539/TransferLearning_Neuroimaging/experiments/sex/age_pretrained/
#$ -P win.prjc
#$ -q gpu8.q
#$ -j y #Error and output file are merged to output file
#$ -l gpu=2
#$ -pe shmem 2 #Should be the same as the number of GPUs 
#$ -l gputype=p100
#Save file to:
#$ -o results/L_sex_age_pretrained.log

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

debug=full

#Change to F: re-train more final layers. 

python ~/TransferLearning_Neuroimaging/train.py \
-deb $debug \
-train pre_full \
-batch 4 \
-n_work 4 \
-path ../../ \
-lr 5e-3 \
-gamma 0.2 \
-epochs 150 \
-pat 15 \
-wdec 5e-2 \
-mom 0.8 \
-pl none \
-task sex \
-pre age \
-loss ent \
-retr 2 \
-run 0 

echo "------------------------------------------------"
echo "Finished at: "`date`
echo "------------------------------------------------"


