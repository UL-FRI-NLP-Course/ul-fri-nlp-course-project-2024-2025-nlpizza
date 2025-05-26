# Natural language processing course: `Improving Prompt Sensitivity in Large Language Models`

This repository contains the codebase, data, models, and report for our project on improving **prompt sensitivity** in Large Language Models (LLMs). We explore both inference-time prompting strategies and training-time fine-tuning methods to reduce variability in outputs across semantically equivalent prompt variants. This is Project 3 which is part of the Natural Language Processing course for the academic year 2024/2025. 

## Table of Contents
- [Project Overview](#project-overview)
  - [Inference-Time Prompting Techniques](#inference-time-prompting-techniques)
  - [Training-Time Fine-Tuning](#training-time-fine-tuning)
  - [Evaluation Methods](#evaluation-methods)
- [Models Used](#models-used)
- [Repository Structure](#repository-structure)
- [Reproducibility](#reproducibility)
  - [1. Set up the environment](#1-set-up-the-environment)
  - [2. Prepare the data](#2-prepare-the-data)
  - [3. Fine-tune the model with LoRA](#3-fine-tune-the-model-with-lora)
  - [4. Evaluation with POSIX](#4-evaluation-with-posix)
  - [5. Evaluation with AlpacaEval (LLM-as-a-Judge)](#5-evaluation-with-alpacaeval-llm-as-a-judge)



## Team 
Gonçalo Cardoso, Gopika Krishnan, Ali Muhammad

## Project Overview

**Prompt sensitivity** refers to the variability in model responses when given different prompts with the same underlying meaning. This project investigates the issue through:

### Inference-Time Prompting Techniques

- Vanilla prompting
- Chain of Thought
- Self-Refinement
- Self-Consistency
- Iterative Refinement

### Training-Time Fine-Tuning

- Fine-tuning LLMs using Low-Rank Adaptation (LoRA) with Parameter-Efficient Fine-Tuning (PEFT) on prompt variant groups that share the same intended output.

## Models Used

We experimented with multiple open-weight Large Language Models (LLMs) at the 7B scale for both inference and fine-tuning:

### Inference-Time Evaluation
- Mistral-7B
- Falcon-7B Instruct
- LLaMA-2 7B

### Fine-Tuning (PEFT with LoRA)
- Falcon-7B
- Falcon-RW-1B
- Mistral-7B

### Evaluation Methods

- Prompt Sensitivity Index (POSIX): Quantifies output variability across prompt perturbations.
- AlpacaEval: Uses a Language Model (LLM) as a judge to assess output consistency and quality.



## Repository Structure
```graphql
improving-prompt-sensitivity/
│
├── data/
│   ├── raw/                     # Original POSIX and Alpaca-style prompt variant data
│   ├── processed/               # Annotated and LLM-extended variants with perturbation types and targets
│   └── sample/                  # Minimal working examples for testing and reproducibility
│
├── models/                      # Fine-tuned LoRA weights and saved checkpoints
│
├── notebooks/                   # Jupyter notebooks for analysis and visualization
│
├── reports/
│   ├── fig/                     # Generated figures for the final report
│   ├── code/                    # Highlighted code snippets for inclusion in LaTeX
│   ├── report.tex               # LaTeX source for the final report
│   ├── report.bib               # BibLaTeX bibliography
│   ├── ds_report.cls            # Custom LaTeX document class
│   └── report.pdf               # Compiled PDF report
│
├── src/
│   ├── data/                    # Scripts for preprocessing, grouping, and augmenting prompt variants
│
│   ├── finetune/                # Training-time methods (LoRA, PEFT)
│   │   ├── train_lora.py        # Fine-tuning on grouped prompt variants
│   │   └── submit_lora.slurm    # SLURM script for running LoRA fine-tuning on HPC
│
│   ├── posix_eval/              # Inference-time prompting + POSIX scoring
│   │   ├── run_prompting.py     # Runs inference with CoT, Self-Consistency, etc.
│   │   ├── compute_posix.py     # Computes Prompt Sensitivity Index
│   │   └── submit_posix_eval.slurm  # SLURM job script for POSIX evaluation
│
│   ├── alpaca_eval/             # LLM-as-a-judge evaluation framework
│   │   ├── eval_judge.py        # Uses an LLM to judge consistency/quality of outputs
│   │   └── submit_alpaca_eval.slurm  # SLURM script for LLM-as-judge evaluation
│
├── README.md                    # Project overview, reproducibility guide
├── requirements.txt             # Pinned dependencies for environment setup


```
## Reproducibility
This project was designed and tested on a High-Performance Computing (HPC) environment using the ARNES SLURM-based cluster. While local execution is possible for smaller models or testing, we recommend running on an HPC system with SLURM for full-scale fine-tuning and evaluation.

To reproduce our results from scratch:

### 1. Set up the environment

We recommend Python 3.10+. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2. Prepare the data

The data/processed/ directory already contains annotated and LLM-extended prompt variant groups with named perturbation types and target outputs.

If you want to reproduce the step of annotating and extending the Alpaca dataset from POSIX, use: 
```bash
python src/data/group_variants.py
```
This step is optional — all evaluations and fine-tuning can be done directly using the data in data/processed/sft_train.jsonl.

## 3. Fine-tune the model with LoRA
Option A: Local run (quick example with Falcon-RW-1B)
```bash
python src/finetune/finetune.py \
  --model_name "tiiuae/falcon-rw-1b" \
  --dataset_path data/processed/sft_train.jsonl \
  --output_dir models/falcon1b_lora_output \
  --sample_groups 100 \
  --epochs 1
```
Option B: Full SLURM run (on ARNES or other cluster)
```bash
sbatch src/finetune/run_finetune.slurm
```
## 4. Evaluation with POSIX
Evaluate output stability across prompt variants using the Prompt Sensitivity Index.  Supports both base and fine-tuned models, and allows switching between prompting techniques. 

### Option A: Run locally (for a few groups)
```bash
export TASK_ID=0
export BATCH_SIZE=5
export TECHNIQUE=None
export MODEL_ID="tiiuae/falcon-rw-1b"
python src/posix_eval/compute_posix.py
```
To use a prompting strategy like Chain of Thought:
```bash
export TECHNIQUE=chain_of_thought
```
To use a fine-tuned model:
```bash
export FINETUNE_FLAG=1
export FINETUNE_PATH=models/falcon7b_lora_output/
```
### Option B: Run on SLURM HPC (recommended for full evaluation)
The script is SLURM-array-ready for parallel POSIX computation:
```bash
sbatch src/posix_eval/submit_posix_eval.slurm
```
Each job generates a .csv file with generated responses, formatted prompts, and POSIX scores (overall, per type). The script will also print the average POSIX score for that batch.

## 5. Evaluation with AlpacaEval (LLM-as-a-Judge)
Evaluate output quality and alignment using a language model judge. You can use it with base or fine-tuned models, and with outputs from any prompting technique.
```bash
python src/alpaca_eval/eval_judge.py \
  --predictions data/processed/generated_outputs/ \
  --model mistral-7b
```
