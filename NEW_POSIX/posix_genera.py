import os
import re
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict
import torch.nn.functional as F

# ==== Config and Environment Variables ====
TASK_ID = int(os.getenv('SLURM_ARRAY_TASK_ID', '0'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '50'))
TECH_ARG = os.getenv('TECHNIQUE', 'None')
FINETUNE_FLAG = int(os.getenv('FINETUNE_FLAG', '0')) == 1
FINETUNE_PATH = os.getenv('FINETUNE_PATH', '')
MODEL_ID = os.getenv('MODEL_ID', 'tiiuae/falcon-7b-instruct')
INPUT_FILE = os.getenv('INPUT_FILE', 'alpaca_prompts_extended.json')

# ==== Utility Functions ====
def normalize_whitespace(text):
    return re.sub(r'\s+', ' ', text).strip()

def normalize_text(text):
    text = normalize_whitespace(text)
    return text.lower().strip(' .!?;:,')

def clean_prompt(text):
    # Remove any leading Q:/A: markers
    text = re.sub(r'^(q:+|q\.|question:|question\s*-)[\s]*(?:\.\.\.)?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(a:+|answer:|answer\s*-)[\s]*(?:\.\.\.)?$', '', text, flags=re.IGNORECASE)
    return normalize_whitespace(text)

def extract_clean_response(gen_text, cleaned_prompt):
    gen = normalize_whitespace(gen_text)
    cleaned = normalize_text(cleaned_prompt)
    gen_norm = normalize_text(gen)
    # Strip echoed question prefix
    if gen_norm.startswith(cleaned):
        gen = gen[len(cleaned):].lstrip(' .!?;:')
    # Remove assistant markers
    gen = re.sub(r'^(assistant|answer)[\s:：\-]*', '', gen, flags=re.IGNORECASE)
    return gen.strip()

# ==== Batched Log-Prob Computation ====
def compute_log_probs_batch(prompts, responses, tokenizer, model):
    formatted = [normalize_whitespace(f"You're a helpful assistant. Please answer clearly:\n\n{p}") for p in prompts]
    full_inputs = [f"{p} {normalize_whitespace(r)}" for p, r in zip(formatted, responses)]
    enc = tokenizer(full_inputs, return_tensors='pt', padding=True, truncation=True, max_length=4096)
    input_ids = enc.input_ids.to(model.device)
    attn_mask = enc.attention_mask.to(model.device)
    prompt_lens = [len(tokenizer(p, return_tensors='pt').input_ids[0]) for p in formatted]
    labels = input_ids.clone().fill_(-100)
    for i, pl in enumerate(prompt_lens):
        labels[i, pl:] = input_ids[i, pl:]
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attn_mask, labels=labels)
        loss_flat = F.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1), reduction='none')
        loss = loss_flat.view(labels.shape)
    log_probs, lengths = [], []
    for i, pl in enumerate(prompt_lens):
        mask = labels[i] != -100
        seq_loss = loss[i][mask].sum().item()
        log_probs.append(-seq_loss)
        lengths.append(mask.sum().item())
    return log_probs, lengths

# ==== POSIX Computation ====
def compute_posix(prompts, responses, tokenizer, model):
    N = len(prompts)
    diag_lp, lengths = compute_log_probs_batch(prompts, responses, tokenizer, model)
    cross = np.zeros((N, N))
    for j in range(N):
        # We compute log-probs of all prompts with response[j]
        lps, _ = compute_log_probs_batch(prompts, [responses[j]] * N, tokenizer, model)
        cross[:, j] = lps
    total = 0.0
    for i in range(N):
        for j in range(N):
            if i != j:
                # Ensure length is not zero to prevent division by zero
                if lengths[i] > 0:
                    total += abs((cross[i, j] - diag_lp[i]) / lengths[i])
    return total / (N * (N - 1)) if N > 1 else 0.0 # Handle N=1 case

# ==== Generation ====
def _generate_response(formatted, cleaned, tokenizer, model, do_sample=True, top_p=0.80, temperature=0.6, repetition_penalty=1.2):
    if not formatted.strip().endswith('Answer:'):
        formatted = f"{formatted.strip()}\n\nAnswer:"
    enc = tokenizer(formatted, return_tensors='pt', padding=True).to(model.device)

    gen_kwargs = {
        'max_new_tokens': 250,
        'pad_token_id': tokenizer.eos_token_id,
        'eos_token_id': tokenizer.eos_token_id,
        'do_sample': do_sample,
        'temperature': temperature,         # Added temperature
        'top_p': top_p,                     # Ensure top_p is set and do_sample is True
        'repetition_penalty': repetition_penalty # Added repetition penalty
    }

    # Ensure do_sample is True if top_p or temperature are used
    if top_p is not None or temperature is not None:
        gen_kwargs['do_sample'] = True
    else:
        # If no sampling parameters are provided, default to greedy decoding
        gen_kwargs['do_sample'] = False
        # Remove sampling specific parameters if not doing sampling
        gen_kwargs.pop('temperature', None)
        gen_kwargs.pop('top_p', None)
        gen_kwargs.pop('repetition_penalty', None)


    out = model.generate(enc.input_ids, attention_mask=enc.attention_mask, **gen_kwargs)
    resp_tokens = out[0][enc.input_ids.shape[1]:]
    text = tokenizer.decode(resp_tokens, skip_special_tokens=True)
    return extract_clean_response(text, cleaned)

# ==== Main Loop ====
def process_group(entries, tokenizer, model, technique=None):
    prompts = []
    resps = []
    pert_types = []
    formatted_prompts = []

    # Default generation parameters for main response generation
    # These can be tuned based on the model and desired output
    default_gen_params = {
        'do_sample': True,
        'top_p': 0.8, # A good balance
        'temperature': 0.6, # Not too random, not too deterministic
        'repetition_penalty': 1.1 # Key for preventing repetition
    }

    for e in entries:
        cp = clean_prompt(e['instruction'])
        fmt = "" # Initialize fmt
        
        # Determine generation parameters for technique-specific calls
        gen_params = default_gen_params.copy() # Start with defaults

        if technique == 'chain_of_thought':
            fmt = f"You're a helpful assistant. Please answer clearly:\n\n{cp} Let's think step by step."
        elif technique == 'self_refinement':
            # For rewriting, we might want less randomness for a focused rewrite
            rewrite = _generate_response(f"Rewrite for clarity:\n\n{cp}", cp, tokenizer, model, **default_gen_params)
            fmt = f"You're a helpful assistant. Please answer clearly:\n\n{rewrite}"
        elif technique == 'self_consistency':
            base = f"You're a helpful assistant. Please answer clearly:\n\n{cp} Let's think step by step."
            # For self-consistency candidates, we want more diversity, so higher temperature/top_p, potentially lower repetition_penalty
            
            cand = [_generate_response(base, cp, tokenizer, model, **default_gen_params) for _ in range(3)]
            
            # Filter out empty or extremely short candidates that might arise from bad generations
            cand = [c for c in cand if len(c.strip()) > 10]
            if not cand: # If all candidates are bad, generate one with default params
                cand = [_generate_response(base, cp, tokenizer, model, **default_gen_params)]

            choices = '\n'.join(f"{i+1}. {c}" for i,c in enumerate(cand))
            fmt = f"You are a meta-evaluator. Question: {cp}\nCandidates:\n{choices}\nPreferred answer:"
            # For the meta-evaluator, we might want less randomness to pick the best
            gen_params = {'do_sample':False, 'max_new_tokens': 100} # Meta-evaluator should be deterministic and concise
        elif technique == 'iterative_refinement':
            # Store the current conversation history
            conversation_history = []
            current_response = ""
            
            # Initial generation
            initial_prompt_for_model = f"You're a helpful assistant. Please answer clearly:\n\n{cp}"
            current_response = _generate_response(initial_prompt_for_model, cp, tokenizer, model, **default_gen_params)
            conversation_history.append(f"Q: {cp}\nA: {current_response}") # Add initial Q&A to history
            k_iterations = 2
            # Iterative refinement loop
            for i in range(k_iterations - 1): # If k=1, this loop won't run. If k=2, it runs once for 1 refinement.
                # Construct the prompt for refinement
                # Include the original question and previous answers for context
                refinement_prompt_text = f"You're a helpful assistant. Please refine the previous answer based on the following conversation:\n\n"
                refinement_prompt_text += "\n\n".join(conversation_history)
                refinement_prompt_text += f"\n\nBased on the above, please provide a refined answer to the original question."
                
                # Generate a refined response
                refined_response = _generate_response(refinement_prompt_text, cp, tokenizer, model, **default_gen_params)
                
                # Update current_response and history
                current_response = refined_response
                conversation_history.append(f"Refined Answer {i+1}: {current_response}") # Add refined answer to history
                
            # The final response for this entry after k iterations
            resp = current_response
            fmt = initial_prompt_for_model # Store the initial formatted prompt for consistency
        else:
            fmt = f"You're a helpful assistant. Please answer clearly:\n\n{cp}"
        
        # Generate the final response using the determined parameters if not iterative_refinement
        # For iterative_refinement, `resp` is already determined in its block.
        if technique!= 'iterative_refinement':
            if technique== 'self_consistency':
                resp = _generate_response(fmt, cp, tokenizer, model, **gen_params)
            else:
                resp = _generate_response(fmt, cp, tokenizer, model, **default_gen_params)
        else:
            fmt = f"You're a helpful assistant. Please answer clearly:\n\n{cp}"
        
        


        prompts.append(cp)
        resps.append(resp)
        pert_types.append(e['perturbation'])
        formatted_prompts.append(fmt)
    
    # Handle cases where N < 2 for compute_posix to avoid errors
    if len(prompts) < 2:
        psi = 0.0 # Or raise an error, depending on desired behavior
        per_type = {t: None for t in sorted(set(pert_types) - {'original'})}
        avg_no_noise = 0.0
    else:
        psi = compute_posix(prompts, resps, tokenizer, model)
        pert = pert_types
        types = sorted(set(pert) - {'original'})
        per_type = {}
        for t in types:
            idx = [i for i,p in enumerate(pert) if p in ('original', t)]
            if len(idx)>1: # Need at least 2 entries to compute POSIX
                ps = [prompts[i] for i in idx]
                rs = [resps[i] for i in idx]
                per_type[t] = compute_posix(ps, rs, tokenizer, model)
            else:
                per_type[t] = None
        
        # Calculate avg_no_noise carefully, handling potential None values
        noise_free_values = [v for k,v in per_type.items() if k!='noise_injection' and v is not None]
        avg_no_noise = np.mean(noise_free_values) if noise_free_values else 0.0

    recs = []
    for e,cp,fp,r in zip(entries, prompts, formatted_prompts, resps):
        rec = {
            'group_id':e['group_id'],
            'perturbation':e['perturbation'],
            'instruction': e['instruction'],
            'clean_prompt':cp,
            'formatted_prompt':fp,
            'response':r,
            'posix_all':psi,
            'posix_all_but_noise':avg_no_noise
        }
        rec.update({f'posix_{k}':v for k,v in per_type.items()})
        recs.append(rec)
    return recs, psi

if __name__=='__main__':
    data = [] # Initialize an empty list
    with open(INPUT_FILE, 'r') as f:
        for line in f:
            data.append(json.loads(line)) # <--- ADD THIS LOOP

    grp = defaultdict(list)
    for x in data:
        grp[x['group_id']].append(x)
    ids = sorted(grp)
    batch = ids[TASK_ID*BATCH_SIZE:(TASK_ID+1)*BATCH_SIZE]
    try:
        tech = None if TECH_ARG=='None' else TECH_ARG # Changed to string check
    except ValueError: # More robust error handling for conversion
        tech = TECH_ARG
    tokenizer = AutoTokenizer.from_pretrained(FINETUNE_PATH if FINETUNE_FLAG else MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Load model with mixed precision (float16) and device_map='auto' for efficiency
    model = AutoModelForCausalLM.from_pretrained(FINETUNE_PATH if FINETUNE_FLAG else MODEL_ID, torch_dtype=torch.float16, device_map='auto', trust_remote_code=True)
    model.eval() # Set model to evaluation mode

    all_rec, scores = [], []
    for gid in tqdm(batch, desc='Groups'):
        recs, s = process_group(grp[gid], tokenizer, model, tech)
        all_rec.extend(recs)
        scores.append(s)
    
    df = pd.DataFrame(all_rec)
    
    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)
    
    fname = f"results/posix_{TECH_ARG}_{'ft_' if FINETUNE_FLAG else ''}{MODEL_ID.replace('/','_').split('/')[-1]}_{TASK_ID}.csv"
    df.to_csv(fname, index=False)
    
    # Only print average POSIX if scores exist
    if scores:
        print('Average POSIX:', np.mean(scores))
    else:
        print('No scores to compute average POSIX for this batch.')