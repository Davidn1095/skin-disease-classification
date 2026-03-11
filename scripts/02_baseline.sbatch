#!/bin/bash
#SBATCH --job-name=skin_baseline
#SBATCH --output=output/slurm/baseline_%j.out
#SBATCH --error=output/slurm/baseline_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=batch

cd /mnt/home/users/fqm_235_genyo/davidnunez/skin-disease-classification
mkdir -p output/slurm

module load python/3.11.4 2>/dev/null || true

python3 -u scripts/02_baseline.py
