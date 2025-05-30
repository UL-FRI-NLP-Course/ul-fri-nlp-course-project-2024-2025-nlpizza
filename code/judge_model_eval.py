#!/usr/bin/env python3
"""
judge_model_eval.py

This script evaluates target model responses using a chosen judge model.
It computes similarity scores between gold (Mistral) references and candidate responses.
It works for any judge model (Mistral, Llama, etc.) and any target models (Falcon, Llama, etc.).
It also generates CSVs and plots summarizing performance by perturbation and technique.

Usage:
    python judge_model_eval.py --base_dir /path/to/project --judge_model mistralai/Mistral-7B-Instruct-v0.1
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
parser.add_argument("--base_dir", type=str, required=True, help="Root directory containing CSV files and outputs")
parser.add_argument("--judge_model", type=str, default="mistralai/Mistral-7B-Instruct-v0.1", help="Judge model name or path")
args = parser.parse_args()
BASE_DIR = args.base_dir
RESP_DIR = os.path.join(BASE_DIR, "responses_all_models")

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
    Loads the judge model pipeline.
    """
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
    judge = load_judge_model(args.judge_model)
    all_scores = []

    # Process each response file
    for file in os.listdir(RESP_DIR):
        if not file.endswith("merged.csv") or "mistralai_Mistral-7B-Instruct-v0.2" in file:
            continue  # skip gold files or unrelated files

        match = re.match(r"posix_(.*?)_(meta|tiiuae)_.*_merged.csv", file)
        if not match:
            continue  # skip files without valid naming

        technique = match.group(1)
        model_type = "Llama-2-7b" if "llama" in file.lower() else "Falcon-7b"

        # Load gold references for the technique
        gold_file = f"posix_{technique}_mistralai_Mistral-7B-Instruct-v0.2_merged.csv"
        gold_path = os.path.join(RESP_DIR, gold_file)
        if not os.path.exists(gold_path):
            L(f"⚠️ Missing gold file: {gold_file} for technique: {technique}")
            continue

        df_gold = pd.read_csv(gold_path)
        gold_map = dict(zip(df_gold["formatted_prompt"], df_gold["generated_response"]))

        L(f"📄 Judging file: {file}")
        df_target = pd.read_csv(os.path.join(RESP_DIR, file))
        technique_scores = []

        # Evaluate all group_ids
        for _, row in df_target.iterrows():
            prompt = row["formatted_prompt"]
            gold = gold_map.get(prompt, "")
            if not gold:
                continue
            cand = row["generated_response"]
            score = sim_score(judge, gold, cand)
            L(f"🧪 {technique} | {row['group_id']} | {model_type} | pert={row['perturbation']} → score={score:.3f}")
            row_dict = {
                "group_id": row["group_id"],
                "perturbation": row["perturbation"],
                "technique": technique.replace("_", " ").title(),
                "model": model_type,
                "score": score,
                "prompt": prompt,
                "response": cand
            }
            all_scores.append(row_dict)
            technique_scores.append(row_dict)

        # Save per-technique CSV
        df_tech = pd.DataFrame(technique_scores)
        judged_file_path = os.path.join(BASE_DIR, f"judged_scores_{technique}_{model_type}.csv")
        df_tech.to_csv(judged_file_path, index=False)
        L(f"✅ Saved judged_scores_{technique}_{model_type}.csv")

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
