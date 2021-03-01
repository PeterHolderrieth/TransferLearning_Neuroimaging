#!/bin/bash
#$ -wd /users/win-fmrib-analysis/lhw539/TransferLearning_Neuroimaging/experiments/sex/sex_pretrained/
#$ -P win.prjc
#$ -q gpu8.q
#$ -j y #Error and output file are merged to output file
#$ -l gpu=2
#$ -pe shmem 2 #Should be the same as the number of GPUs 
#$ -l gputype=p100
#Save file to:
#$ -o results/E_sex_sex_pretrained.log

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

#Change to C: regularize even more (increase weight decay, increase momentum)
#Increase patience.

python ~/TransferLearning_Neuroimaging/train.py \
-deb $debug \
-train pre_full \
-batch 8 \
-n_work 4 \
-lr_ll 1e-1 \
-mom_ll 0.9 \
-wdec_ll 1e-3 \
-pl_ll none \
-pat_ll 8 \
-gamma_ll 0.1 \
-path ../../ \
-lr 5e-4 \
-gamma 0.2 \
-epochs 40 \
-pat 10 \
-wdec 1e-2 \
-mom 0.9 \
-pl none \
-epochs_ll 25 \
-task sex \
-pre sex \
-loss ent \
-run 0 

echo "------------------------------------------------"
echo "Finished at: "`date`
echo "------------------------------------------------"

