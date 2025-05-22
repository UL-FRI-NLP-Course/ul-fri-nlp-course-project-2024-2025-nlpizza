import torch
import numpy as np
import json
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import sys
import os
from collections import defaultdict

def normalize_whitespace(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([?.!,:])', r'\1', text)
    return text.strip()

def clean_prompt(text):
    text = re.sub(r"^(question|q|Q|Q::|q:::)[\s:：]*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[\n\r]?(answer|a|a:::|A:|A::)[\s:：]*$", "", text, flags=re.IGNORECASE)
    return normalize_whitespace(text)

def extract_clean_response(generated_text, formatted_prompt, cleaned_prompt):
    gen = normalize_whitespace(generated_text)
    prompt = normalize_whitespace(formatted_prompt)
    cleaned = normalize_whitespace(cleaned_prompt).rstrip(" ?!.")

    if gen.startswith(prompt):
        gen = gen[len(prompt):].strip()

    gen = re.sub(rf"^(?:{re.escape(cleaned)})([\s\?\.!:\-]*)", "", gen, flags=re.IGNORECASE)
    gen = re.sub(r"^(assistant|a|answer)[\s:：\-]*", "", gen, flags=re.IGNORECASE)

    return gen.strip()

def compute_log_prob(prompt, response, tokenizer, model):
    formatted_prompt = f"You're a helpful assistant. Please answer clearly:\n\n{prompt.strip()}"
    formatted_prompt = normalize_whitespace(formatted_prompt)
    response = normalize_whitespace(response)

    input_text = formatted_prompt + " " + response
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
    response_ids = tokenizer(response, return_tensors="pt").input_ids.to(model.device)
    response_length = response_ids.shape[1]

    labels = -100 * torch.ones_like(input_ids)
    prompt_length = input_ids.shape[1] - response_length
    labels[0, prompt_length:] = input_ids[0, prompt_length:]

    with torch.no_grad():
        outputs = model(input_ids, labels=labels)
        log_prob = -outputs.loss.item() * response_length

    return log_prob, response_length

def _generate_response(formatted, cleaned, tokenizer, model, do_sample=False, top_p=None):
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    gen_kwargs = {
        'max_new_tokens': 30,
        'pad_token_id': tokenizer.eos_token_id,
        'do_sample': do_sample
    }
    if top_p is not None:
        gen_kwargs['top_p'] = top_p
    output = model.generate(inputs.input_ids, **gen_kwargs)
    return extract_clean_response(tokenizer.decode(output[0], skip_special_tokens=True), formatted, cleaned)

def process_prompt_set(set_id, entries, tokenizer, model, prompting_technique=None):
    responses, techniques = [], []
    for entry in entries:
        prompt_text = entry['instruction']
        perturb = entry['perturbation']
        cleaned = clean_prompt(prompt_text)

        if prompting_technique == "chain_of_thought":
            formatted = f"You're a helpful assistant. Please answer clearly:\n\n{cleaned} Let's think step by step."
            response = _generate_response(formatted, cleaned, tokenizer, model)
        elif prompting_technique == "self_refinement":
            rewrite = f"Rewrite this question for better clarity:\n\n{cleaned}"
            rewritten = _generate_response(rewrite, cleaned, tokenizer, model)
            formatted = f"You're a helpful assistant. Please answer clearly:\n\n{rewritten}"
            response = _generate_response(formatted, cleaned, tokenizer, model)
        elif prompting_technique == "self_consistency":
            formatted = f"You're a helpful assistant. Please answer clearly:\n\n{cleaned} Let's think step by step."
            gens = [_generate_response(formatted, cleaned, tokenizer, model, do_sample=True, top_p=0.9) for _ in range(5)]
            response = max(set(gens), key=gens.count)
        elif isinstance(prompting_technique, int):
            output_text = ""
            for step in range(prompting_technique):
                segment = f"PROMPT:\n{cleaned}\n"
                if step > 0:
                    segment += f"PREVIOUS OUTPUT:\n{output_text.strip()}\nPlease refine or continue the answer above.\n"
                formatted = f"You're a helpful assistant. Please answer clearly:\n\n{segment}"
                output_text = _generate_response(formatted, cleaned, tokenizer, model)
            response = output_text
        else:
            formatted = f"You're a helpful assistant. Please answer clearly:\n\n{cleaned}"
            response = _generate_response(formatted, cleaned, tokenizer, model)

        responses.append(response)
        techniques.append(perturb)

    N = len(entries)
    log_probs = np.zeros((N, N))
    lengths = np.zeros(N)
    for i in range(N):
        for j in range(N):
            logp, L = compute_log_prob(entries[i]['instruction'], responses[j], tokenizer, model)
            log_probs[i][j], lengths[j] = logp, lengths[j] if i!=j else L
    psi = sum(abs((log_probs[i][j] - log_probs[i][i]) / lengths[i]) for i in range(N) for j in range(N) if i!=j) / (N*(N-1))

    records = []
    for entry, resp, perturb in zip(entries, responses, techniques):
        records.append({
            'group_id': entry['group_id'],
            'original_prompt': entry['original_prompt'],
            'instruction': entry['instruction'],
            'perturbation': perturb,
            'response': resp,
            'posix_score': psi
        })
    return psi, records

def main():
    # Determine batching indices: args > env
    task = int(os.getenv('SLURM_ARRAY_TASK_ID', '0'))
    start_idx, end_idx = task*50, (task + 1)*50
    tech_arg = os.getenv('TECHNIQUE', 'None')

    try:
        prompting_technique = int(tech_arg)
    except ValueError:
        prompting_technique = None if tech_arg == "None" else tech_arg

    # Load and group prompts
    with open('prompts.json') as f:
        data = json.load(f)
    grouped = defaultdict(list)
    for item in data:
        grouped[item['group_id']].append(item)
    group_ids = sorted(grouped.keys())[start_idx:end_idx]
    prompt_sets = [grouped[g] for g in group_ids]

    # Load model
    model_id = os.getenv('MODEL_ID', 'tiiuae/falcon-7b-instruct')
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map='auto')
    model.eval()

    all_records, posix_scores = [], []
    for gid, entries in zip(group_ids, tqdm(prompt_sets, desc="Groups")):
        psi, recs = process_prompt_set(gid, entries, tokenizer, model, prompting_technique)
        posix_scores.append(psi)
        all_records.extend(recs)

    df = pd.DataFrame(all_records)
    fname = f"posix_alpacaextended_{tech_arg}_{model_id.replace('/', '_')}.csv"
    df.to_csv(fname, index=False)
    print(f"Average POSIX: {np.mean(posix_scores):.4f}")
    print(f"Saved results to {fname}")

if __name__ == '__main__':
    main()
