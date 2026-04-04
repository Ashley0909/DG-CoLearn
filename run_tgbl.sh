#!/bin/bash
#SBATCH -c 8
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --job-name=dgcolearn-tgbl
#SBATCH --tasks-per-node=1
#SBATCH --output=%x-%j.out
#SBATCH --time=11:59:00

source /nfs-share/ahta3/miniforge3/etc/profile.d/conda.sh
conda activate dgcolearn311

python3 -u main.py tgbl-coin --mode ctdg --patch_size 100000 --incremental_learning False