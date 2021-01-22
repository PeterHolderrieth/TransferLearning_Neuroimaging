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
#$ -o results/A_replicating_han_22012021.log

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

epochs=150
lr=1e-2
gamma=0.75
loss=kl
pat=10
drop=True 
debug=full 
batch=4

learning_rate = 0.01
weight_decay = 0.001
decay_epoch = 30
decay_gamma = 0.3
momentum = 0.9
total_epoch = 130
log_interval = 1
batch_size = 8
seed_number = 3517
preprocessing = ('pixel_shift', 'average', 'mirror_1')
optimizer = sgd
scheduler_type = step
num_workers = 6

python ~/TransferLearning_Neuroimaging/train.py -epochs $epochs -debug $debug -lr $lr -gamma 0.75 -init fresh -drop $drop -pat $pat -loss $loss


echo "------------------------------------------------"
echo "Finished at: "`date`
echo "------------------------------------------------"

