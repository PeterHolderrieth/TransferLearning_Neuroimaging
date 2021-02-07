#!/bin/bash
#$ -wd /users/win-fmrib-analysis/lhw539/TransferLearning_Neuroimaging/experiments/replicating_han/
#$ -P win.prjc
#$ -q gpu8.q
#$ -j y #Error and output file are merged to output file
#$ -l gpu=2
#$ -pe shmem 2 #Should be the same as the number of GPUs 
#$ -l gputype=p100
#Save file to:
# Log locations which are relative to the current                                                                                                                                                                  # working directory of the submission
#$ -o results/U_Run_0_replicating_han_22012021.log

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

#CHANGE TO C_replicating_han_22012021.sh: batch size reduced.

python ~/TransferLearning_Neuroimaging/train.py \
-deb $debug \
-train pre_step \
-loss kl \
-batch 4 \
-n_work 4 \
-lr_ll 1e-1 \
-mom_ll 0.9 \
-wdec_ll 1e-3 \
-wdec 1e-4 \
-pl_ll none \
-pat_ll 8 \
-gamma_ll 0.1 \
-path ../../ \
-lr 5e-4 \
-gamma 0.2 \
-epochs 30 \
-pat 5 \
-wdec 5e-4 \
-mom 0.8 \
-pl none \
-epochs_ll 25 \
-run 0

echo "------------------------------------------------"
echo "Finished at: "`date`
echo "------------------------------------------------"

