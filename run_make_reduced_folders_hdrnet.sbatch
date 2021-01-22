#!/bin/bash

#SBATCH --job-name=hdrnet
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64gb
#SBATCH --time=6:00:00
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

# Execute the script
py3=/scratch/awd275/miniconda3/envs/hdrnet/bin/python3
pyscript=/home/awd275/hdrnet-pytorch/scripts/make_reduced_folder.py 
size=1024

input=/scratch/awd275/StyleTransfer/data/fivek_dataset/raw/fivek_original 
output=/scratch/awd275/StyleTransfer/data/fivek_dataset/processed/fivek_original_reduced/
echo 'run1'
$py3 $pyscript -i $input -o $output -s $size &> output_make_reduced.txt

input=/scratch/awd275/StyleTransfer/data/fivek_dataset/raw/fivek_editor_A
output=/scratch/awd275/StyleTransfer/data/fivek_dataset/processed/fivek_editor_A_reduced/
echo 'run2'
$py3 $pyscript -i $input -o $output -s $size  >> output_make_reduced.txt

input=/scratch/awd275/StyleTransfer/data/fivek_dataset/raw/fivek_editor_B
output=/scratch/awd275/StyleTransfer/data/fivek_dataset/processed/fivek_editor_B_reduced/
echo 'run3'
$py3 $pyscript -i $input -o $output -s $size >> output_make_reduced.txt

input=/scratch/awd275/StyleTransfer/data/fivek_dataset/raw/fivek_editor_C
output=/scratch/awd275/StyleTransfer/data/fivek_dataset/processed/fivek_editor_C_reduced/
echo 'run4'
$py3 $pyscript -i $input -o $output -s $size >> output_make_reduced.txt

input=/scratch/awd275/StyleTransfer/data/fivek_dataset/raw/fivek_editor_D
output=/scratch/awd275/StyleTransfer/data/fivek_dataset/processed/fivek_editor_D_reduced/
echo 'run5'
$py3 $pyscript -i $input -o $output -s $size >> output_make_reduced.txt

input=/scratch/awd275/StyleTransfer/data/fivek_dataset/raw/fivek_editor_E
output=/scratch/awd275/StyleTransfer/data/fivek_dataset/processed/fivek_editor_E_reduced/
echo 'run6'
$py3 $pyscript -i $input -o $output -s $size  >> output_make_reduced.txt






