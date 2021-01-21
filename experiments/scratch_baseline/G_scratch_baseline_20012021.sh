#!/bin/bash
#$ -wd /users/win-fmrib-analysis/lhw539/TransferLearning_Neuroimaging/experiments/scratch_baseline/
#$ -P win.prjc
#$ -q gpu8.q
#$ -j y #Error and output file are merged to output file
#$ -l gpu=1
# -pe shmem 2
# -l gputype=p100
#Save file to:
# Log locations which are relative to the current                                                                                                                                                                  # working directory of the submission
#$ -o results/G_scratch_baseline_mae.log

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
loss=mae
pat=10
drop=drop 
debug=full 
pl=none 

python ~/TransferLearning_Neuroimaging/train.py -epochs $epochs -deb $debug -lr $lr -gamma 0.75 -init fresh -drop $drop -pat $pat -loss $loss -pl $pl 


echo "------------------------------------------------"
echo "Finished at: "`date`
echo "------------------------------------------------"

