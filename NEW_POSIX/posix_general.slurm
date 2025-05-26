#!/bin/bash
#SBATCH --job-name=posix_eval
#SBATCH --output=logs/posix_%A_%a.log
#SBATCH --error=logs/posix_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --array=0-99              # 50 jobs

#SBATCH --mail-type=END                 # Notify when job ends
#SBATCH --mail-user=gr88022@student.uni-lj.si

# Load environment (modify as needed)
module load Python/3.10.8-GCCcore-12.2.0
source GC/bin/activate

# Set technique and model
#export TECHNIQUE="chain_of_tought"   # can be: None, chain_of_thought, self_refinement, self_consistency, or 'iterative_refinement'
export MODEL_ID="mistralai/Mistral-7B-v0.1"
export FINETUNE_FLAG="0" # works as a bollean 0 will be False, 1 will be True
export FINETUNE_PATH="None"
export INPUT_FILE="sft_test.jsonl"
export BATCH_SIZE="5"
# Run script (automatically uses $SLURM_ARRAY_TASK_ID)
python Posix_script.py

deactivate
