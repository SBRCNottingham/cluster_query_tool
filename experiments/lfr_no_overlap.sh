#!/bin/bash
#PBS -k oe
#PBS -l select=1:ncpus=16:mem=16gb
#PBS -l walltime=00:30:00
#PBS -P HPCA-02856-EFR

# setting MU parameter
MU=$0
N=$1

WORK_DIR=$HOME/repos/cluster_query_tool
RESULTS_DIR=$WORK_DIR/hpc_results

JOB=$PBS_ARRAY_INDEX

mkdir -p $RESULTS_DIR
cd $WORK_DIR
if [ "$0" != "" ]; then
    echo "RUNNING JOB SETTING MU=$MU, $N=$N"
    python experiments/lfr_nooverlap.py auc_compute $N $MU $JOB $RESULTS_DIR/no_overlap_$N.json
else
    echo "CANNOT RUN WITHOUT SETTING MU"
fi