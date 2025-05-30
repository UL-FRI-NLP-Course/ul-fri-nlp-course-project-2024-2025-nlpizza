#!/usr/bin/env python3
"""
judge_model_eval.py

This script evaluates target model responses using a specified judge model.
It computes similarity scores between gold references and candidate responses.
It generates CSVs and plots summarizing performance by perturbation and technique.

Usage Example:
    python judge_model_eval.py --base_dir /path/to/project \
                               --judge_model mistralai/Mistral-7B-Instruct-v0.1 \
                               --target_model tiiuae/falcon-7b-instruct

Expected Directory Structure:
    base_dir/
        ├── responses_all_models/
        │      ├── posix_{technique}_{target_model_short}_merged.csv
        │      ├── posix_{technique}_{judge_model_short}_merged.csv
        │      └── ...
        └── outputs/

Naming Conventions:
    - Gold references (Mistral or any judge model) should follow:
        posix_{technique}_{judge_model_short}_merged.csv
    - Target responses should follow:
        posix_{technique}_{target_model_short}_merged.csv
    - Example:
        posix_chain_of_thought_mistralai_Mistral-7B-Instruct-v0.2_merged.csv
        posix_chain_of_thought_tiiuae_falcon-7b-instruct_merged.csv

Tips:
    - Replace underscores in technique names with spaces in final plots (optional).
    - Models are identified by the substrings after the last '/' (e.g., mistralai/Mistral-7B-Instruct-v0.1 → Mistral-7B-Instruct-v0.1).
"""

import os
import sys
import re
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, logging as hf_logging

# ─── ARGUMENT PARSING ───
parser = argparse.ArgumentParser(description="Judge Model Evaluation Script")
parser.add_argument("--base_dir", type=str, required=True,
                    help="Root directory containing responses and outputs")
parser.add_argument("--judge_model", type=str, required=True,
                    help="Judge model name or path (e.g. mistralai/Mistral-7B-Instruct-v0.1)")
parser.add_argument("--target_model", type=str, required=True,
                    help="Target model name or path (e.g. tiiuae/falcon-7b-instruct)")
args = parser.parse_args()

BASE_DIR = args.base_dir
RESP_DIR = os.path.join(BASE_DIR, "responses_all_models")
os.makedirs(RESP_DIR, exist_ok=True)

# ─── LOGGING SETUP ───
log_file_path = os.path.join(BASE_DIR, "judge_model_eval.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("judge_eval")
def L(msg): logger.info(msg)

# ─── SUPPRESS TRANSFORMERS WARNINGS ───
hf_logging.set_verbosity_error()

# ─── MODEL LOADING ───
def load_judge_model(model_id):
    """
    Loads the judge model pipeline from HuggingFace or local cache.
    """
    L(f"🔍 Loading judge model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=BASE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=BASE_DIR
    )
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=64,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

# ─── SIMILARITY SCORING ───
def sim_score(judge, gold, cand):
    """
    Computes the similarity score between a gold and candidate response using the judge model.
    """
    prompt = (
        "You are an expert in evaluating how semantically similar two answers are.\n"
        "Given a reference and a candidate, output only a similarity score between 0.0 and 1.0.\n\n"
        f"Reference:\n{gold}\n\n"
        f"Candidate:\n{cand}\n\n"
        "Similarity Score:"
    )
    try:
        raw_out = judge(prompt)[0]["generated_text"]
        cleaned = raw_out.strip().split("Similarity Score:")[-1].strip()
        m = re.search(r"([01](?:\.\d+)?)(?!\d)", cleaned)
        if not m:
            m = re.search(r"\b0?\.\d+\b|\b1\.0\b", cleaned)
        score = float(m.group(0)) if m else 0.0
    except Exception as e:
        L(f"❌ Error scoring: {e}")
        score = 0.0
    return round(max(0.0, min(1.0, score)), 3)

# ─── MAIN EVALUATION ───
if __name__ == "__main__":
    judge_pipeline = load_judge_model(args.judge_model)
    judge_short = args.judge_model.split("/")[-1]
    target_short = args.target_model.split("/")[-1]
    all_scores = []

    # Process each technique
    for file in os.listdir(RESP_DIR):
        if not file.endswith("merged.csv"):
            continue

        match = re.match(r"posix_(.*?)_.*_(.*)_merged.csv", file)
        if not match:
            continue

        technique = match.group(1)

        # Check if this is a target response file
        if target_short in file:
            gold_file = f"posix_{technique}_{judge_short}_merged.csv"
            gold_path = os.path.join(RESP_DIR, gold_file)
            if not os.path.exists(gold_path):
                L(f"⚠️ Missing gold file: {gold_file} for technique: {technique}")
                continue

            df_gold = pd.read_csv(gold_path)
            gold_map = dict(zip(df_gold["formatted_prompt"], df_gold["generated_response"]))

            L(f"📄 Judging file: {file}")
            df_target = pd.read_csv(os.path.join(RESP_DIR, file))
            technique_scores = []

            for _, row in df_target.iterrows():
                prompt = row["formatted_prompt"]
                gold = gold_map.get(prompt, "")
                if not gold:
                    continue
                cand = row["generated_response"]
                score = sim_score(judge_pipeline, gold, cand)
                L(f"🧪 {technique} | {row['group_id']} | {target_short} | pert={row['perturbation']} → score={score:.3f}")
                row_dict = {
                    "group_id": row["group_id"],
                    "perturbation": row["perturbation"],
                    "technique": technique.replace("_", " ").title(),
                    "model": target_short,
                    "score": score,
                    "prompt": prompt,
                    "response": cand
                }
                all_scores.append(row_dict)
                technique_scores.append(row_dict)

            # Save per-technique CSV
            df_tech = pd.DataFrame(technique_scores)
            judged_file_path = os.path.join(BASE_DIR, f"judged_scores_{technique}_{target_short}.csv")
            df_tech.to_csv(judged_file_path, index=False)
            L(f"✅ Saved judged_scores_{technique}_{target_short}.csv")

    # Save combined results
    df_all = pd.DataFrame(all_scores)
    df_all.to_csv(os.path.join(BASE_DIR, "judged_scores_all.csv"), index=False)
    L("✅ Saved judged_scores_all.csv")

    # ─── VISUALIZATIONS ───
    sns.set_theme(style="whitegrid")

    # Boxplot: Perturbations per model
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=df_all, x="perturbation", y="score", hue="model")
    plt.title("Model Similarity Scores by Perturbation")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "boxplot_perturbation_models.png"))
    L("📊 Saved perturbation boxplot")

    # Boxplot: Techniques per model
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=df_all, x="technique", y="score", hue="model")
    plt.title("Model Similarity Scores by Technique")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "boxplot_technique_models.png"))
    L("📊 Saved technique boxplot")

    # Mean Score Table
    summary = df_all.groupby(["technique", "model", "perturbation"]).score.agg(['mean', 'std']).reset_index()
    summary.to_csv(os.path.join(BASE_DIR, "summary_scores_table.csv"), index=False)
    L("✅ Saved summary_scores_table.csv")

    overall_summary = df_all.groupby(["model"]).score.mean().reset_index()
    L(f"🔎 Overall Mean Scores:\n{overall_summary}")

    L("✅ All done! This script is fully reproducible and ready for sharing!")

