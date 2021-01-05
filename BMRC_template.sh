#!/bin/bash
#$ -wd $HOME/TransferLearning_Neuroimaging/
#$ -N  template_job #specifies a name
#$ -q short.qc #specifies queue

echo "------------------------------------------------"
echo "Run on host: "`hostname`
echo "Operating system: "`uname -s`
echo "Username: "`whoami`
echo "Started at: "`date`
echo "------------------------------------------------"

module load Python/3.7.2-GCCcore-8.2.0

sleep 10s
python -c 'print("Hello")'



