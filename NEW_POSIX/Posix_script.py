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
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '5'))
TECH_ARG = os.getenv('TECHNIQUE', 'None')
FINETUNE_FLAG = int(os.getenv('FINETUNE_FLAG', '0')) == 1
FINETUNE_PATH = os.getenv('FINETUNE_PATH', '')
MODEL_ID = os.getenv('MODEL_ID', 'tiiuae/falcon-7b-instruct')
INPUT_FILE = os.getenv('INPUT_FILE', 'alpaca_prompts_extended.json')

# ==== Utility Functions ====
def normalize_whitespace(text):
    return re.sub(r'\s+', ' ', text).strip()

def clean_prompt(text):
    # Only normalize whitespace based on new requirements
    return normalize_whitespace(text)

def extract_clean_response(gen_text, prompt):
    gen_norm = normalize_whitespace(gen_text)
    # Strip echoed prompt prefix if it appears
    # This remains robust in case the model echoes the prompt
    if gen_norm.startswith(prompt):
        gen_norm = gen_norm[len(prompt):].lstrip(' .!?;:')
    
    # Aggressively cut off unsolicited conversational turns or patterns
    # This handles cases where the model generates more than just the answer
    unsolicited_pattern = r'\b(?:Q(?:uestion)?[\s:：\-]*\s*(?:\d+\.)?|\bA(?:nswer)?[\s:：\-]*\s*(?:\d+\.)?|Now, please check out|The following questions|question \d+|answers):?'
    match = re.search(unsolicited_pattern, gen_norm, flags=re.IGNORECASE)
    if match:
        gen_norm = gen_norm[:match.start()].strip() # Cut the string at the start of the unwanted pattern

    return gen_norm.strip()

# ==== Batched Log-Prob Computation ====
def compute_log_probs_batch(prompts, responses, tokenizer, model):
    # Prompts here are the *formatted* prompts as given to the model for generation
    formatted_prompts = [p for p in prompts] # Assuming 'prompts' here are already formatted
    full_inputs = [f"{p} {normalize_whitespace(r)}" for p, r in zip(formatted_prompts, responses)]
    
    # Ensure consistent tokenization with max_length and truncation
    enc = tokenizer(full_inputs, return_tensors='pt', padding=True, truncation=True, max_length=tokenizer.model_max_length)
    input_ids = enc.input_ids.to(model.device)
    attn_mask = enc.attention_mask.to(model.device)
    
    # Recalculate prompt lengths based on the actual formatted prompts for log_probs
    prompt_lens = [len(tokenizer(p, return_tensors='pt', truncation=True, max_length=tokenizer.model_max_length).input_ids[0]) for p in formatted_prompts]
    
    labels = input_ids.clone().fill_(-100)
    for i, pl in enumerate(prompt_lens):
        labels[i, pl:] = input_ids[i, pl:] # Only compute loss for response tokens

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attn_mask, labels=labels)
        # Use outputs.logits to get log-probabilities
        # Make sure to flatten for cross_entropy, then reshape for per-token analysis
        loss_flat = F.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1), reduction='none')
        loss = loss_flat.view(labels.shape)
        
    log_probs, lengths = [], []
    for i, pl in enumerate(prompt_lens):
        mask = labels[i] != -100 # Mask for response tokens
        seq_loss = loss[i][mask].sum().item()
        log_probs.append(-seq_loss) # Convert negative log-likelihood (loss) to log-probability
        lengths.append(mask.sum().item()) # Number of response tokens
    return log_probs, lengths

# ==== POSIX Computation ====
def compute_posix(prompts, responses, tokenizer, model):
    # prompts here should be the *formatted* prompts used for generation
    N = len(prompts)
    if N < 2: # Handle N=1 case gracefully
        return 0.0

    diag_lp, lengths = compute_log_probs_batch(prompts, responses, tokenizer, model)
    cross = np.zeros((N, N))

    for j in range(N):
        # We compute log-probs of all formatted prompts with response[j]
        lps, _ = compute_log_probs_batch(prompts, [responses[j]] * N, tokenizer, model)
        cross[:, j] = lps

    total = 0.0
    for i in range(N):
        for j in range(N):
            if i != j:
                # Ensure length is not zero to prevent division by zero
                if lengths[i] > 0:
                    total += abs((cross[i, j] - diag_lp[i]) / lengths[i])
    return total / (N * (N - 1))


# ==== Generation ====
def _generate_response(formatted, original_cleaned_prompt, tokenizer, model, do_sample=True, top_p=0.85, temperature=0.7, repetition_penalty=1.1, max_new_tokens=300):
    # Ensure truncation and padding are handled consistently
    enc = tokenizer(formatted, return_tensors='pt', padding=True).to(model.device)

    gen_kwargs = {
        'max_new_tokens': max_new_tokens,
        'pad_token_id': tokenizer.eos_token_id,
        'eos_token_id': tokenizer.eos_token_id,
        'do_sample': do_sample,
        'temperature': temperature,
        'top_p': top_p,
        'repetition_penalty': repetition_penalty,
    }

    # Ensure do_sample is True if top_p or temperature are used
    if top_p is not None or temperature is not None:
        gen_kwargs['do_sample'] = True
    else:
        gen_kwargs['do_sample'] = False
        gen_kwargs.pop('temperature', None)
        gen_kwargs.pop('top_p', None)
        gen_kwargs.pop('repetition_penalty', None)


    out = model.generate(enc.input_ids, attention_mask=enc.attention_mask, **gen_kwargs)
    
    # Decode only the newly generated tokens
    # out[0] is the entire sequence (prompt + generated response)
    # enc.input_ids.shape[1] is the length of the prompt
    resp_tokens = out[0][enc.input_ids.shape[1]:]
    text = tokenizer.decode(resp_tokens, skip_special_tokens=True)
    
    # Pass the *original_cleaned_prompt* to extract_clean_response if you expect it to be echoed
    # If the formatted prompt itself is used for echoing, pass `formatted`
    return extract_clean_response(text, original_cleaned_prompt)


# ==== Main Loop ====
def process_group(entries, tokenizer, model, technique=None):
    original_prompts = [] # The raw instruction from the dataset
    formatted_prompts = [] # The prompt string sent to the model for generation
    generated_responses = [] # The model's response
    perturbation_types = []

    # Default generation parameters for main response generation
    # These can be tuned based on the model and desired output
    default_gen_params = {
        'do_sample': True,
        'top_p': 0.85, 
        'temperature': 0.7,
        'repetition_penalty': 1.1,
        'max_new_tokens': 300 
    }

    # Parameters for iterative refinement
    iterative_gen_params = {
        'do_sample': True,
        'top_p': 0.85, 
        'temperature': 0.7,
        'repetition_penalty': 1.1,
        'max_new_tokens': 400 
    }

    for e in entries:
        # cp is the 'instruction' from the dataset, after minimal cleaning (whitespace only)
        cp = clean_prompt(e['instruction']) 
        
        # fmt is the actual prompt string that will be sent to the model
        fmt = "" 
        # resp is the model's generated response
        resp = "" 

        if technique == 'chain_of_thought':
            # COT: Add "Let's think step by step."
            fmt = f"Your role is to meticulously execute instructions and provide concise, relevant output.\n\nTask: {cp} Let's think step by step."
            resp = _generate_response(fmt, cp, tokenizer, model, **default_gen_params)
            
        elif technique == 'self_refinement':
            # Self-Refinement: First, generate a rewrite of the prompt
            rewrite_gen_params = default_gen_params.copy()
            rewrite_gen_params['max_new_tokens'] = 150 # Shorter rewrite
            rewrite_gen_params['temperature'] = 0.5 # A bit more deterministic for rewrite
            rewrite_gen_params['top_p'] = 0.8

            # The prompt for the rewrite itself
            rewrite_prompt_for_model = f"Your role is to rewrite prompts concisely for clarity.\n\nTask: Rewrite this prompt for clarity:\n{cp}\n\nRewritten prompt:"
            rewrite = _generate_response(rewrite_prompt_for_model, cp, tokenizer, model, **rewrite_gen_params)
            
            # Then, use the rewritten prompt to get the final answer
            fmt = f"Your role is to meticulously execute instructions and provide concise, relevant output.\n\nTask: {rewrite}"
            resp = _generate_response(fmt, cp, tokenizer, model, **default_gen_params)

        elif technique == 'self_consistency':
            # Self-Consistency: Generate multiple candidates with COT prompt, then use meta-evaluator
            base_prompt_for_candidates = f"Your role is to meticulously execute instructions and provide concise, relevant output.\n\nTask: {cp} Let's think step by step."
            
            # Generate candidates (e.g., 3 candidates)
            cand = [_generate_response(base_prompt_for_candidates, cp, tokenizer, model, **default_gen_params) for _ in range(3)]
            
            # Filter out empty or extremely short candidates that might arise from bad generations
            cand = [c for c in cand if len(c.strip()) > 10]
            if not cand: # If all candidates are bad, generate one with default params
                cand = [_generate_response(base_prompt_for_candidates, cp, tokenizer, model, **default_gen_params)]

            choices = '\n'.join(f"{i+1}. {c}" for i,c in enumerate(cand))
            
            # Meta-evaluator prompt and parameters (deterministic)
            fmt = f"You are a meta-evaluator. Evaluate the following candidates and select the best answer for the question: '{cp}'\n\nCandidates:\n{choices}\n\nPreferred answer:"
            meta_eval_gen_params = {'do_sample':False, 'max_new_tokens': 300, 'temperature': 0.7, 'top_p': 0.85, 'repetition_penalty': 1.0} 
            resp = _generate_response(fmt, cp, tokenizer, model, **meta_eval_gen_params)

        elif technique == 'iterative_refinement':
            # Iterative Refinement: Generate an initial response, then refine it iteratively
            conversation_history = []
            
            # Initial generation
            initial_prompt_for_model = f"Your role is to meticulously execute instructions and provide concise, relevant output.\n\nTask: {cp}"
            current_response = _generate_response(initial_prompt_for_model, cp, tokenizer, model, **default_gen_params)
            
            # Store initial Q&A in history for refinement context
            conversation_history.append(f"User: {initial_prompt_for_model}\nAssistant: {current_response}") 
            
            k_iterations = 2 # Number of refinement turns
            
            for i in range(k_iterations - 1): 
                # Construct the prompt for refinement
                refinement_prompt_text = f"Your role is to meticulously refine previous answers based on the conversation history."
                refinement_prompt_text += "\n\n".join(conversation_history)
                refinement_prompt_text += f"\n\nRefinement Request: Please refine the last assistant answer to be more precise." # Specific refinement instruction
                refinement_prompt_text += f"\n\nRefined Answer:" # Clear delimiter for refined answer

                # Generate a refined response using iterative_gen_params
                refined_response = _generate_response(refinement_prompt_text, cp, tokenizer, model, **iterative_gen_params)
                
                # Update history with the refinement request and the refined answer
                conversation_history.append(f"User: Refinement Request: Please refine the last assistant answer to be more precise.\nAssistant: {refined_response}")
                current_response = refined_response # Update current response

            resp = current_response # The final response after k iterations
            fmt = initial_prompt_for_model # Store the initial formatted prompt (first iteration's prompt) for consistency

        else: # Default case: No specific technique, just the base prompt
            fmt = f"Your role is to meticulously execute instructions and provide concise, relevant output.\n\nTask: {cp}"
            resp = _generate_response(fmt, cp, tokenizer, model, **default_gen_params)
        
        # Collect data for CSV
        original_prompts.append(e['instruction']) # Keep original instruction
        formatted_prompts.append(fmt) # The actual prompt sent to the model
        generated_responses.append(resp) # The final response from the model
        perturbation_types.append(e['perturbation'])
    
    # POSIX calculation needs to use the *formatted_prompts* that were actually sent to the model
    # and the *generated_responses* from those prompts.
    if len(formatted_prompts) < 2:
        psi = 0.0 
        per_type = {t: None for t in sorted(set(perturbation_types) - {'original'})}
        avg_no_noise = 0.0
    else:
        psi = compute_posix(formatted_prompts, generated_responses, tokenizer, model)
        
        # Compute POSIX per perturbation type
        pert = perturbation_types
        types = sorted(set(pert) - {'original'})
        per_type = {}
        for t in types:
            # For per-type POSIX, we need formatted prompts and responses ONLY for that type + 'original'
            idx = [i for i,p in enumerate(pert) if p in ('original', t)]
            if len(idx)>1: 
                ps_subset = [formatted_prompts[i] for i in idx]
                rs_subset = [generated_responses[i] for i in idx]
                per_type[t] = compute_posix(ps_subset, rs_subset, tokenizer, model)
            else:
                per_type[t] = None
        
        noise_free_values = [v for k,v in per_type.items() if k!='noise_injection' and v is not None]
        avg_no_noise = np.mean(noise_free_values) if noise_free_values else 0.0

    recs = []
    for i, e in enumerate(entries): # Iterate through original entries to link data
        rec = {
            'group_id': e['group_id'],
            'perturbation': e['perturbation'],
            'original_instruction': original_prompts[i], # Original instruction from dataset
            'formatted_prompt': formatted_prompts[i], # The full prompt sent to model
            'generated_response': generated_responses[i], # The model's final response
            'posix_all': psi,
            'posix_all_but_noise': avg_no_noise
        }
        rec.update({f'posix_{k}':v for k,v in per_type.items()})
        recs.append(rec)
    return recs, psi

if __name__=='__main__':
    data = [] 
    with open(INPUT_FILE, 'r') as f:
        for line in f:
            data.append(json.loads(line)) 
    print('loaded data')
    grp = defaultdict(list)
    print('started grouping loop')
    for x in data:
        grp[x['group_id']].append(x)
    ids = sorted(grp)
    batch = ids[TASK_ID*BATCH_SIZE:(TASK_ID+1)*BATCH_SIZE]
    try:
        tech = None if TECH_ARG=='None' else TECH_ARG 
    except ValueError: 
        tech = TECH_ARG
    print('Loading tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(FINETUNE_PATH if FINETUNE_FLAG else MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print('Loaded tokenizer, loading model')
    model = AutoModelForCausalLM.from_pretrained(FINETUNE_PATH if FINETUNE_FLAG else MODEL_ID, torch_dtype=torch.float16, device_map='auto', trust_remote_code=True)
    model.eval() 
    print('Loaded model, starting big loop')
    all_rec, scores = [], []
    for gid in tqdm(batch, desc='Groups'):
        recs, s = process_group(grp[gid], tokenizer, model, tech)
        all_rec.extend(recs)
        scores.append(s)
    print('finished big loop')
    df = pd.DataFrame(all_rec)
    
    os.makedirs('results', exist_ok=True)
    
    fname = f"results/posix_{TECH_ARG}_{'ft_' if FINETUNE_FLAG else ''}{MODEL_ID.replace('/','_').split('/')[-1]}_{TASK_ID}.csv"
    df.to_csv(fname, index=False)
    
    if scores:
        print('Average POSIX:', np.mean(scores))
    else:
        print('No scores to compute average POSIX for this batch.')