#!/bin/bash

#SBATCH --job-name=hdrnet_pre
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=72:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=awd275@nyu.edu
#SBATCH --output=slurm_%j.out

# Refer to https://sites.google.com/a/nyu.edu/nyu-hpc/documentation/prince/batch/submitting-jobs-with-sbatch
# for more information about the above options

# Remove all unused system modules
module purge

source ~/.bashrc
conda activate hdrnet

cd /home/awd275/hdrnet-pytorch/

export PYTHONPATH="./"
echo pretrained
# Execute the script
py3=/scratch/awd275/miniconda3/envs/hdrnet/bin/python3
pyscript=/home/awd275/hdrnet-pytorch/scripts/train.py
logging_path=/home/awd275/hdrnet-pytorch/fivek_logs/
gpus=1
dataset_flag=1
mode=2
#checkpoint_path=/home/awd275/hdrnet-pytorch/checkpoints/default1.chkpt 
batch_size=8
$py3 $pyscript -l $logging_path -g $gpus -d $dataset_flag -m $mode -b $batch_size




