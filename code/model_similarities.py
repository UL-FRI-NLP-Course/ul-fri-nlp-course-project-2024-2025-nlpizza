#!/usr/bin/env python3
import os, sys, json, gc, logging, re, argparse
import torch, numpy as np, pandas as pd, matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, logging as hf_logging

# ─── ARGUMENTS ───
parser = argparse.ArgumentParser()
parser.add_argument("--job_idx", type=int, required=True, help="Job index (1 to 4)")
parser.add_argument("--samples_per_job", type=int, default=10, help="Number of data samples per job")
args = parser.parse_args()
JOB_IDX = args.job_idx
SAMPLES_PER_JOB = args.samples_per_job

# ─── PATHS & ENV ───
BASE_DIR = "/d/hpc/projects/FRI/ma76193"
MODEL_CACHE = os.path.join(BASE_DIR, "model_cache")
MODEL_CACHE_JOB = os.path.join(MODEL_CACHE, f"job_{os.environ.get('SLURM_JOB_ID', f'LOCAL_{JOB_IDX}')}")
LOG_FILE = os.path.join(BASE_DIR, "logs", f"fast_run_job_{JOB_IDX}.log")
SCORES_FILE = os.path.join(BASE_DIR, f"scores_job_{JOB_IDX}.csv")

os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
os.makedirs(MODEL_CACHE_JOB, exist_ok=True)

os.environ.update({
    "HF_HOME": BASE_DIR,
    "HUGGINGFACE_HUB_CACHE": MODEL_CACHE_JOB,
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "TRANSFORMERS_OFFLINE": "0",
    "TRANSFORMERS_CACHE": MODEL_CACHE_JOB,
})

# ─── LOGGING SETUP ───
logger = logging.getLogger("fast_run")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(LOG_FILE, mode="w")
fh.setFormatter(logging.Formatter("%(asctime)s ─ %(levelname)s ─ %(message)s"))
logger.addHandler(fh)
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(logging.Formatter("%(asctime)s ─ %(levelname)s ─ %(message)s"))
logger.addHandler(sh)
hf_logging.set_verbosity_info()
hf_logging.enable_default_handler()
hf_logging.enable_explicit_format()

def L(msg):
    logger.info(msg)
    for h in logger.handlers:
        h.flush()

# ─── EARLY EXIT ───
if os.path.exists(SCORES_FILE):
    L(f"✅ Job {JOB_IDX} already completed. Found: {SCORES_FILE}")
    sys.exit(0)

if "HUGGINGFACE_HUB_TOKEN" not in os.environ or not os.environ["HUGGINGFACE_HUB_TOKEN"]:
    sys.exit("❌ HUGGINGFACE_HUB_TOKEN is empty or not set.")

# ─── MODEL LOAD HELPERS ───
def load_pipeline(repo, tokenizer_repo=None, max_new_tokens=128, do_sample=False):
    L(f"🔄 Loading model: {repo}")
    if tokenizer_repo is None:
        tokenizer_repo = repo
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_repo,
        cache_dir=MODEL_CACHE_JOB,
        use_auth_token=os.environ["HUGGINGFACE_HUB_TOKEN"],
        trust_remote_code=True,
        padding_side="left",
        resume_download=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        repo,
        cache_dir=MODEL_CACHE_JOB,
        use_auth_token=os.environ["HUGGINGFACE_HUB_TOKEN"],
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        resume_download=True
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=0.7)

# ─── DATASET & OUTPUT MANAGEMENT ───
def load_or_generate_gold(judge, data):
    gold_path = os.path.join(BASE_DIR, f"job_{JOB_IDX}_Gold_outputs.json")
    if os.path.exists(gold_path):
        with open(gold_path) as f:
            return json.load(f)
    L("🖍️ Generating Gold outputs...")
    gold = {}
    for ex in data:
        pid = str(ex["id"])
        gold[pid] = []
        for i, p in enumerate(ex["prompts"]):
            L(f"🔁 JOB {JOB_IDX} | Prompt {i+1}/10 for ID {pid}")
            try:
                result = judge(p)[0]["generated_text"][len(p):].strip()
            except Exception as e:
                L(f"⚠️ Error during generation: {e}")
                result = ""
            gold[pid].append(result)
    with open(gold_path, "w") as f:
        json.dump(gold, f, indent=2)
    return gold

def load_model_outputs(name):
    path = os.path.join(BASE_DIR, f"job_{JOB_IDX}_{name}_outputs.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

# ─── SIM SCORE ───
def sim_score(judge, gold, cand):
    prompt = (
        "Compare the following two texts and rate their similarity on a scale from 0.0 to 1.0.\n"
        "Return only the number, as a decimal with at least one digit after the point.\n\n"
        f"Reference:\n{gold}\n\nCandidate:\n{cand}\n\nSimilarity Score:"
    )
    out = judge(prompt)[0]["generated_text"]
    L(f"🔎 Raw Judge Output: {out!r}")
    m = re.search(r"Similarity Score:\s*([01](?:\.\d+)?)(?!\d)", out)
    score = float(m.group(1)) if m else 0.0
    score = max(0.0, min(1.0, score))
    L(f"    ↳ sim_score = {score}")
    return score

# ─── MAIN RUN ───
if __name__ == "__main__":
    L(f"🚀 Fast run starting for JOB {JOB_IDX}")
    with open(os.path.join(BASE_DIR, "alpaca_prompts.json")) as f:
        alpaca = json.load(f)

    start = (JOB_IDX - 1) * SAMPLES_PER_JOB
    end = start + SAMPLES_PER_JOB
    data = alpaca[start:end]
    for ex in data:
        ex["prompts"] = ex["prompts"][:10]

    judge = load_pipeline("mistralai/Mistral-7B-Instruct-v0.1", max_new_tokens=64)
    Gold = load_or_generate_gold(judge, data)
    del judge; gc.collect(); torch.cuda.empty_cache()

    models = {
        "GPT2-medium": "openai-community/gpt2-medium",
        "EleutherAI-pythia-410m": "EleutherAI/pythia-410m",
        "Falcon-7b-instruct": "tiiuae/falcon-7b-instruct",
    }

    AllOut = {}
    for name, repo in models.items():
        outputs = load_model_outputs(name)
        if outputs:
            L(f"✅ Loaded existing outputs for {name}")
            AllOut[name] = outputs
            continue

        pipe = load_pipeline(repo, do_sample=True)
        out_d = {}
        for ex in data:
            pid = str(ex["id"])
            outs = [pipe(p)[0]["generated_text"][len(p):].strip() for p in ex["prompts"]]
            outs = [o for o in outs if o.strip() and len(set(o.split())) > 5]
            out_d[pid] = outs
            L(f"    ↳ {name}[{pid}] → {len(outs)} gens")

        with open(os.path.join(BASE_DIR, f"job_{JOB_IDX}_{name}_outputs.json"), "w") as f:
            json.dump(out_d, f, indent=2)
        AllOut[name] = out_d
        del pipe; gc.collect(); torch.cuda.empty_cache()

    judge = load_pipeline("mistralai/Mistral-7B-Instruct-v0.1", max_new_tokens=64)
    results = []
    for pid, gold_list in Gold.items():
        for name in AllOut:
            if pid not in AllOut[name]:
                L(f"⚠️ Missing output for {name} on ID {pid}")
                continue
            scores = [
                sim_score(judge, gold_list[i], AllOut[name][pid][i])
                for i in range(min(len(gold_list), len(AllOut[name][pid])))
            ]
            arr = np.array(scores, dtype=float)
            results.append({
                "model": name, "id": pid,
                "orig": float(arr[0]) if len(arr) > 0 else 0.0,
                "min": float(arr.min()) if len(arr) > 0 else 0.0,
                "max": float(arr.max()) if len(arr) > 0 else 0.0,
                "mean": float(arr.mean()) if len(arr) > 0 else 0.0,
                "std": float(arr.std()) if len(arr) > 0 else 0.0
            })

    df = pd.DataFrame(results)
    df.to_csv(SCORES_FILE, index=False)
    L(f"✅ Wrote scores → {SCORES_FILE}")
    L("🏁 Job complete.")
