#!/bin/bash
#SBATCH -p performance
#SBATCH -t 2-00:00:00
#SBATCH --gpus=4
#SBATCH --job-name=model_training
#SBATCH --output=outerr.log
#SBATCH --error=outerr.log

module load python/3.10.12
module load cuda
module load nccl
pipenv run python functions/main.py
