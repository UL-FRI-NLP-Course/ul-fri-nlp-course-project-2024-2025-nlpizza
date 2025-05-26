#!/bin/bash
#SBATCH --job-name=lora-llm
#SBATCH --output=logs/lora-%j.out
#SBATCH --error=logs/lora-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

source ~/.bashrc
# Conda enviroment according to the requirements.txt file
conda activate posixenv

cd $SLURM_SUBMIT_DIR

# Example command to run the finetuning script with LoRA on Falcon-7B-Instruct model
python finetune.py \
    --model_name "tiiuae/Falcon-7B-Instruct" \
    --dataset_path data/sft_train.jsonl \
    --output_dir falcon7b_lora_output \
    --quant4bit \
    --lora_r 16 \
    --lora_alpha 32 \
    --target_modules "query_key_value,dense,dense_h_to_4h,dense_4h_to_h" \
    --batch_size 4 \
    --gradient_accumulation 4
