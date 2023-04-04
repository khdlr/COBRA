#!/bin/bash

#SBATCH --account=hai_gmb_dl
#SBATCH --nodes=1
#SBATCH --output=slurm/slurm_%j.out
#SBATCH --error=slurm/slurm_%j.out
#SBATCH --time=20:00:00
#SBATCH --job-name=semi
#SBATCH --gres=gpu:4
#SBATCH --partition=booster
#SBATCH --mail-type=ALL

# go to the project directory
CWD="$PROJECT_hai_gmb_dl/konrad/cobra"
echo "CWD = $CWD"
cd $CWD

source ~/.bashrc
# activate the conda environment
conda activate

CUDA_VISIBLE_DEVICES=0 python -m train -f -s 1092830 &
CUDA_VISIBLE_DEVICES=1 python -m train -f -s 9182791 &
CUDA_VISIBLE_DEVICES=2 python -m train -f -s 2938790 &
CUDA_VISIBLE_DEVICES=3 python -m train -f -s 1928708 &

wait
