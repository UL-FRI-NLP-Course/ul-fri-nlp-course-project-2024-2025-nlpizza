#!/bin/bash
#SBATCH --job-name=posix_eval
#SBATCH --output=logs/posix_%A_%a.log
#SBATCH --error=logs/posix_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --array=0-45                     # 46 jobs (each handles 100 prompts)

#SBATCH --mail-type=END                 # Notify when job ends
#SBATCH --mail-user=gk40784@student.uni-lj.si

source ~/.bashrc
conda activate posixenv

start=$((SLURM_ARRAY_TASK_ID * 100))
end=$((start + 100))

python run_posix.py $start $end
