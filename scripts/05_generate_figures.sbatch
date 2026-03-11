#!/bin/bash
#SBATCH --job-name=skin_figs
#SBATCH --output=output/slurm/figures_%j.out
#SBATCH --error=output/slurm/figures_%j.err
#SBATCH --time=00:15:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu_partition
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu

cd /mnt/home/users/fqm_235_genyo/davidnunez/skin-disease-classification
mkdir -p output/slurm

module purge
module load pytorch/2.10.0

python3 -u scripts/05_generate_figures.py
