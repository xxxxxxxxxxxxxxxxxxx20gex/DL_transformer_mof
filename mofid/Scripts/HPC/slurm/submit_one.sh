#!/bin/bash
# Call as Scripts/HPC/slurm/submit_one.sh path_to_cifs short_friendly_job_name

dir=$1
echo "Submitting $dir"
export CIFDIR=$dir
export CIFOUTLABEL=$2
export OUTPUT_DIR="Output_$2"
export MOFSTDOUT="out_$2.smi"
sbatch -J id-$2 -o pbs_out_$2.txt -e err_$2.txt Scripts/HPC/slurm/child_mofid_slurm.job

