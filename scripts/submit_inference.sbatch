#!/bin/bash
#SBATCH --job-name=fno_infer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm_fno_inference_%j.out

echo "=========================================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Compute Node: $(hostname)"
echo "Submit Time:  $(date)"
echo "=========================================================="

echo "Allocated GPU Hardware:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

module purge
module load anaconda3
source activate fno-inference

python scripts/run_inference.py

echo "=========================================================="
echo "Job Completion Time: $(date)"
echo "=========================================================="
