#!/bin/bash
#SBATCH --job-name=posix_eval
#SBATCH --output=logs/posix_%A_%a.log
#SBATCH --error=logs/posix_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --array=0-199               # 200 jobs (each handles 600 prompts)

#SBATCH --mail-type=END                 # Notify when job ends
#SBATCH --mail-user=gr88022@student.uni-lj.si

# Load environment (modify as needed)
source ~/.bashrc
conda activate GC

# Set technique and model
#export TECHNIQUE=chain_of_thought    # can be: None, chain_of_thought, self_refinement, self_consistency, or an int
export MODEL_ID=tiiuae/falcon-7b-instruct
export FINETUNE_FLAG = 0 # works as a bollean 0 will be False, 1 will be True
export FINETUNE_PATH = None
export INPUT_FILE = 
export BATCH_SIZE = 50
# Run script (automatically uses $SLURM_ARRAY_TASK_ID)
python Posix_Script_General.py
