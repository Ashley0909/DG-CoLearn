#!/bin/bash
#SBATCH -c 8
#SBATCH -p hopper
#SBATCH -w ruapehu
#SBATCH --gres=gpu:1
#SBATCH --job-name=dgcolearn-tgbn
#SBATCH --tasks-per-node=1
#SBATCH --output=%x-%j.out
#SBATCH --time=11:59:00

source /nfs-share/ahta3/miniforge3/etc/profile.d/conda.sh
conda activate dgcolearn311

python3 -u main.py tgbn-reddit --mode ctdg --patch_size 100000 --incremental_learning True
# python3 main.py tgbn-reddit --mode ctdg --patch_size 100000 --fl_strategy feddgl --incremental_learning False
