#!/bin/bash
#SBATCH --job-name=skin_cnn
#SBATCH --output=output/slurm/transfer_%j.out
#SBATCH --error=output/slurm/transfer_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu_partition
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu

cd /mnt/home/users/fqm_235_genyo/davidnunez/skin-disease-classification
mkdir -p output/slurm

module purge
module load pytorch/2.10.0

export NUM_WORKERS=6
export ARCH=${ARCH:-resnet18}
export TORCH_HOME=/mnt/home/users/fqm_235_genyo/davidnunez/.cache/torch

python3 -u scripts/03_transfer_learning.py
