#!/bin/bash
#$ -wd $HOME/TransferLearning_Neuroimaging/
#$ -P win.prjc
echo "------------------------------------------------"
echo "Run on host: "`hostname`
echo "Operating system: "`uname -s`
echo "Username: "`whoami`
echo "Started at: "`date`
echo "------------------------------------------------"

module load Python/3.7.2-GCCcore-8.2.0

#qsub -l gpu=2 -l gputype=p100 -pe shmem 2 ~/deep_medicine/submit_sh/run_20191206_00.sh

sleep 10s
python -c 'print("Hello")'



