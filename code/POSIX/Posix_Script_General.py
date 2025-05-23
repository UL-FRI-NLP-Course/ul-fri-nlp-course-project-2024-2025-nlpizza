import os
import re
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict

# ==== Config and Environment Variables ====
TASK_ID = int(os.getenv('SLURM_ARRAY_TASK_ID', '0'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '50'))
TECH_ARG = os.getenv('TECHNIQUE', 'None')
FINETUNE_FLAG = int(os.getenv('FINETUNE_FLAG', '0')) == '1'
FINETUNE_PATH = os.getenv('FINETUNE_PATH', '')
MODEL_ID = os.getenv('MODEL_ID', 'tiiuae/falcon-7b-instruct')
INPUT_FILE = os.getenv('INPUT_FILE', 'prompts.json')  # expects JSON list of prompt entries

# ==== Utility Functions ==== 
def normalize_whitespace(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([?.!,:])', r'\1', text)
    return text.strip()

def clean_prompt(text):
    #text = re.sub(r"^(question|q|Q|Q::|q:::)[\s:：]*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[\n\r]?(answer|a|a:::|A:|A::)[\s:：]*$", "", text, flags=re.IGNORECASE)
    return normalize_whitespace(text)

def extract_clean_response(generated_text, formatted_prompt, cleaned_prompt):
    gen = normalize_whitespace(generated_text)
    prompt = normalize_whitespace(formatted_prompt)
    cleaned = normalize_whitespace(cleaned_prompt).rstrip(" ?!." )

    if gen.startswith(prompt):
        gen = gen[len(prompt):].strip()
    gen = re.sub(rf"^(?:{re.escape(cleaned)})([\s\?\.!:\-]*)", "", gen, flags=re.IGNORECASE)
    gen = re.sub(r"^(assistant|a|answer)[\s:：\-]*", "", gen, flags=re.IGNORECASE)
    return gen.strip()

def compute_log_prob(prompt, response, tokenizer, model):
    formatted = f"You're a helpful assistant. Please answer clearly:\n\n{prompt.strip()}"
    formatted = normalize_whitespace(formatted)
    response = normalize_whitespace(response)

    input_text = formatted + ' ' + response
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids.to(model.device)
    resp_ids = tokenizer(response, return_tensors='pt').input_ids.to(model.device)
    resp_len = resp_ids.shape[1]

    labels = -100 * torch.ones_like(input_ids)
    prompt_len = input_ids.shape[1] - resp_len
    labels[0, prompt_len:] = input_ids[0, prompt_len:]

    with torch.no_grad():
        outputs = model(input_ids, labels=labels)
        log_prob = -outputs.loss.item() * resp_len
    return log_prob, resp_len

def _generate_response(formatted, cleaned, tokenizer, model, do_sample=False, top_p=None):
    inputs = tokenizer(formatted, return_tensors='pt').to(model.device)
    gen_kwargs = {
        'max_new_tokens': 30,
        'pad_token_id': tokenizer.eos_token_id,
        'do_sample': do_sample
    }
    if top_p is not None:
        gen_kwargs['top_p'] = top_p
    output = model.generate(inputs.input_ids, **gen_kwargs)
    return extract_clean_response(tokenizer.decode(output[0], skip_special_tokens=True), formatted, cleaned)

def compute_posix(prompts, responses, tokenizer, model):
    N = len(prompts)
    log_probs = np.zeros((N, N))
    lengths = np.zeros(N)
    for i in range(N):
        for j in range(N):
            lp, L = compute_log_prob(prompts[i], responses[j], tokenizer, model)
            log_probs[i][j] = lp
            if i == j:
                lengths[i] = L
    psi = sum(
        abs((log_probs[i][j] - log_probs[i][i]) / lengths[i])
        for i in range(N) for j in range(N) if i != j
    ) / (N * (N - 1))
    return psi

# ==== Main Processing ==== 
def process_group(entries, tokenizer, model, prompting_technique=None):
    responses, perturbs = [], []
    for entry in entries:
        cleaned = clean_prompt(entry['instruction'])
        # choose formatting based on technique
        if prompting_technique == "chain_of_thought":
            formatted = f"You're a helpful assistant. Please answer clearly:\n\n{cleaned} Let's think step by step."
            resp = _generate_response(formatted, cleaned, tokenizer, model)
        elif prompting_technique == "self_refinement":
            rewrite = f"Rewrite this question for better clarity:\n\n{cleaned}"
            rewritten = _generate_response(rewrite, cleaned, tokenizer, model)
            formatted = f"You're a helpful assistant. Please answer clearly:\n\n{rewritten}"
            resp = _generate_response(formatted, cleaned, tokenizer, model)
        elif prompting_technique == "self_consistency":
            # generate multiple samples
            base_prompt = f"You're a helpful assistant. Please answer clearly:\n\n{cleaned} Let's think step by step."
            gens = [ _generate_response(base_prompt, cleaned, tokenizer, model, do_sample=True, top_p=0.9)
                     for _ in range(3) ]
            # assemble meta-prompt asking the model to pick the best answer
            choices = "\n".join([f"{i+1}. {ans}" for i,ans in enumerate(gens)])
            meta = (
                f"You are a meta-evaluator. Given the question and multiple candidate answers,"
                f" choose the best answer and output just the answer.\n\n"
                f"Question: {cleaned}\n\n"
                f"Candidates:\n{choices}\n\n"
                f"Preferred answer:"
            )
            resp = _generate_response(meta, cleaned, tokenizer, model)
        elif prompting_technique is not None and prompting_technique.isdigit():
            steps = int(prompting_technique)
            output = ""
            for s in range(steps):
                segment = f"PROMPT:\n{cleaned}\n"
                if s > 0:
                    segment += f"PREVIOUS OUTPUT:\n{output.strip()}\nPlease refine or continue the answer above.\n"
                formatted = f"You're a helpful assistant. Please answer clearly:\n\n{segment}"
                output = _generate_response(formatted, cleaned, tokenizer, model)
            resp = output
        else:
            formatted = f"You're a helpful assistant. Please answer clearly:\n\n{cleaned}"
            resp = _generate_response(formatted, cleaned, tokenizer, model)
        responses.append(resp)
        perturbs.append(entry['perturbation'])

    # Compute POSIX scores
    prompts = [e['instruction'] for e in entries]
    posix_all = compute_posix(prompts, responses, tokenizer, model)
    types = sorted({p for p in perturbs if p != 'original'})
    per_type = {}
    for ptype in types:
        idxs = [i for i,p in enumerate(perturbs) if p in ['original', ptype]]
        if len(idxs) >= 2:
            ps = [prompts[i] for i in idxs]
            rs = [responses[i] for i in idxs]
            per_type[ptype] = compute_posix(ps, rs, tokenizer, model)
        else:
            per_type[ptype] = None
    non_noise = [v for k,v in per_type.items() if k.lower() != 'noise_injection' and v is not None]
    posix_all_but_noise = float(np.mean(non_noise)) if non_noise else None

    # Build records
    records = []
    for e,r in zip(entries, responses):
        rec = {
            'group_id': e['group_id'],
            'original_prompt': e.get('original_prompt',''),
            'instruction': e['instruction'],
            'perturbation': e['perturbation'],
            'response': r,
            'posix_all': posix_all,
            **{f'posix_{k}': per_type.get(k) for k in per_type},
            'posix_all_but_noise': posix_all_but_noise
        }
        records.append(rec)
    return records, posix_all

if __name__ == '__main__':
    # select groups
    with open(INPUT_FILE) as f:
        data = json.load(f)
    grouped = defaultdict(list)
    for item in data:
        grouped[item['group_id']].append(item)
    ids = sorted(grouped.keys())
    batch = ids[TASK_ID*BATCH_SIZE:(TASK_ID+1)*BATCH_SIZE]

    # parse technique
    try:
        prompting_technique = None if TECH_ARG=='None' else int(TECH_ARG)
    except ValueError:
        prompting_technique = TECH_ARG

    # load model/tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        FINETUNE_PATH if FINETUNE_FLAG else MODEL_ID,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        FINETUNE_PATH if FINETUNE_FLAG else MODEL_ID,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True
    )
    model.eval()

    all_records, scores = [], []
    for gid in tqdm(batch, desc='Groups'):
        recs, psi = process_group(grouped[gid], tokenizer, model, prompting_technique)
        all_records.extend(recs)
        scores.append(psi)

    df = pd.DataFrame(all_records)
    if FINETUNE_FLAG:
        out = f"posix_summary_{TECH_ARG}_finetuned_{MODEL_ID.replace('/','_')}_{TASK_ID}.csv"
    else:
        out = f"posix_summary_{TECH_ARG}_{MODEL_ID.replace('/','_')}_{TASK_ID}.csv"
    df.to_csv(out, index=False)
    print(f"Average POSIX: {np.mean(scores):.4f}")
    print(f"Results saved to {out}")
