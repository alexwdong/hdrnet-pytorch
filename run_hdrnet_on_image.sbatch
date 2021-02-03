#!/bin/bash

#SBATCH --job-name=hdrnet_one_im
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=1:00
#SBATCH --output=slurm_%j.out

# Refer to https://sites.google.com/a/nyu.edu/nyu-hpc/documentation/prince/batch/submitting-jobs-with-sbatch
# for more information about the above options

# Remove all unused system modules
module purge

source ~/.bashrc
conda activate hdrnet

cd /home/awd275/hdrnet-pytorch/

export PYTHONPATH="src/"

# Execute the script
py3=python3
pyscript=scripts/run_hdrnet_on_image.py

mode=1
image_path=/home/awd275/hdrnet-pytorch/data/tiny_original/DSC_4329.jpg
checkpoint_path=/home/awd275/hdrnet-pytorch/fivek_logs/lightning_logs/version_1924577/checkpoints/epoch=25-step=12749.ckpt
out_size=1024
$py3 $pyscript -m $mode -i $image_path -c $checkpoint_path -o $out_size




