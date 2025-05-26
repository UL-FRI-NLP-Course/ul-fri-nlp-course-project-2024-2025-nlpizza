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

# --- Configuration and Environment Variables ---
TASK_ID = int(os.getenv('SLURM_ARRAY_TASK_ID', '0'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '5'))
TECH_ARG = os.getenv('TECHNIQUE', 'None')
FINETUNE_FLAG = int(os.getenv('FINETUNE_FLAG', '0')) == 1
FINETUNE_PATH = os.getenv('FINETUNE_PATH', '')
MODEL_ID = os.getenv('MODEL_ID', 'tiiuae/falcon-7b-instruct')
INPUT_FILE = os.getenv('INPUT_FILE', 'alpaca_prompts_extended.json')

# --- Utility Functions (Minimal) ---

# --- Batched Log-Prob Computation ---
def compute_log_probs_batch(prompts, responses, tokenizer, model):
    """
    Computes log-probabilities for generated responses given prompts in a batch.
    Prompts here are the *formatted* prompts as given to the model for generation.
    """
    # Create full input strings by concatenating formatted prompt and response
    # Responses are stripped to handle decoding artifacts, but original prompts are untouched
    full_inputs = [f"{p}{r.strip()}" for p, r in zip(prompts, responses)]
    
    # Tokenize inputs, padding to the longest in the batch and truncating if necessary
    enc = tokenizer(full_inputs, return_tensors='pt', padding=True, truncation=True, max_length=model.config.max_position_embeddings)
    input_ids = enc.input_ids.to(model.device)
    attn_mask = enc.attention_mask.to(model.device)
    
    # Recalculate prompt lengths based on the actual formatted prompts
    # This ensures log_probs are only computed for the generated response part
    prompt_lens = [len(tokenizer(p, return_tensors='pt').input_ids[0]) for p in prompts]
    
    # Create labels: -100 for prompt tokens (ignored in loss), actual token IDs for response
    labels = input_ids.clone().fill_(-100)
    for i, pl in enumerate(prompt_lens):
        # Ensure the response part starts at the correct index, capped by input_ids length
        labels[i, pl:input_ids.shape[1]] = input_ids[i, pl:input_ids.shape[1]]

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attn_mask, labels=labels)
        
        # Calculate loss (negative log-likelihood) per token
        loss_flat = F.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1), reduction='none')
        
        # Reshape loss back to batch x sequence length
        loss = loss_flat.view(labels.shape)
        
    log_probs, lengths = [], []
    for i, pl in enumerate(prompt_lens):
        mask = labels[i] != -100 # Mask for response tokens
        seq_loss = loss[i][mask].sum().item()
        log_probs.append(-seq_loss) # Convert negative log-likelihood to log-probability
        lengths.append(mask.sum().item()) # Number of response tokens
        
    return log_probs, lengths

# --- POSIX Computation ---
def compute_posix(prompts, responses, tokenizer, model):
    """
    Computes the POSIX metric for a given set of prompts and responses.
    """
    N = len(prompts)
    if N < 2:
        return 0.0

    diag_lp, lengths = compute_log_probs_batch(prompts, responses, tokenizer, model)
    cross_lps = np.zeros((N, N))

    for j in range(N): # Iterate through each response
        # Compute log-probs of all formatted prompts with response[j]
        # We need N identical responses for batching
        lps, _ = compute_log_probs_batch(prompts, [responses[j]] * N, tokenizer, model)
        cross_lps[:, j] = lps

    total_diff_sum = 0.0
    num_pairs = 0
    for i in range(N):
        for j in range(N):
            if i != j:
                # Ensure length is not zero to prevent division by zero
                if lengths[i] > 0:
                    total_diff_sum += abs((cross_lps[i, j] - diag_lp[i]) / lengths[i])
                    num_pairs += 1
    
    return total_diff_sum / num_pairs if num_pairs > 0 else 0.0


# --- Generation Function (Simplified) ---
def _generate_response(formatted_prompt, tokenizer, model, **gen_kwargs):
    """
    Generates a response from the model given a formatted prompt.
    Uses default generation parameters unless specified.
    """
    # Tokenize input, padding if needed, truncating to model's max position embeddings
    # `add_special_tokens=False` might be relevant if your raw prompts already have <s>, etc.
    enc = tokenizer(formatted_prompt, return_tensors='pt', padding=True, truncation=True, 
                    max_length=model.config.max_position_embeddings).to(model.device)

    # Merge custom gen_kwargs with model's defaults for generation
    final_gen_kwargs = model.generation_config.to_dict()
    final_gen_kwargs.update(gen_kwargs)

    # Ensure pad_token_id and eos_token_id are set if not in config
    if tokenizer.pad_token_id is not None:
        final_gen_kwargs['pad_token_id'] = tokenizer.pad_token_id
    elif tokenizer.eos_token_id is not None: # Fallback if pad_token is none
        final_gen_kwargs['pad_token_id'] = tokenizer.eos_token_id
        
    if tokenizer.eos_token_id is not None:
        final_gen_kwargs['eos_token_id'] = tokenizer.eos_token_id

    out = model.generate(enc.input_ids, attention_mask=enc.attention_mask, **final_gen_kwargs)
    
    # Decode only the newly generated tokens
    # out[0] is the entire sequence (prompt + generated response)
    # enc.input_ids.shape[1] is the length of the prompt
    resp_tokens = out[0][enc.input_ids.shape[1]:]
    text = tokenizer.decode(resp_tokens, skip_special_tokens=True)
    
    return text.strip() # Still strip trailing whitespace from the *generated response*

# --- Main Logic for Processing a Group ---
def process_group(entries, tokenizer, model, technique=None):
    """
    Processes a group of related prompts (e.g., original + perturbations)
    by generating responses using a specified technique and computing POSIX.
    """
    original_instructions = []  # The raw instruction from the dataset
    formatted_prompts = []      # The prompt string actually sent to the model
    generated_responses = []    # The model's generated response
    perturbation_types = []     # Type of perturbation for the entry

    # Default generation parameters to use for main response generation
    generation_params = {'max_new_tokens' : 200} # Using empty dict to signal use model defaults

    # Parameters for iterative refinement, if different from general generation
    iterative_gen_params = {} # Also using model defaults here

    for e in entries:
        # `raw_instruction` is the instruction from the dataset, used directly
        raw_instruction = e['instruction'] 
        
        fmt = "" # The prompt that will be sent to the model
        resp = "" # The model's response

        if technique == 'chain_of_thought':
            fmt = f"Your role is to meticulously execute instructions and provide concise, relevant output.\n\n {raw_instruction} Let's think step by step."
            resp = _generate_response(fmt, tokenizer, model, **generation_params)
            
        elif technique == 'self_refinement':
            # Parameters for the rewrite generation (can be different from main generation)
            # Example specific params
            
            # First, generate a rewrite of the prompt
            rewrite_prompt_for_model = f"Your role is to rewrite prompts concisely for clarity.\n\n Rewrite this prompt for clarity:\n{raw_instruction}\n\nRewritten prompt:"
            rewrite = _generate_response(rewrite_prompt_for_model, tokenizer, model, **generation_params)
            
            # Then, use the rewritten prompt to get the final answer
            fmt = f"Your role is to meticulously execute instructions and provide concise, relevant output.\n\n {rewrite}"
            resp = _generate_response(fmt, tokenizer, model, **generation_params)

        elif technique == 'self_consistency':
            base_prompt_for_candidates = f"Your role is to meticulously execute instructions and provide concise, relevant output.\n\n {raw_instruction} Let's think step by step."
            
            # Generate multiple candidates
            num_candidates = 3 # Number of candidates to generate
            candidates = [_generate_response(base_prompt_for_candidates, tokenizer, model, **generation_params) for _ in range(num_candidates)]
            
            # Filter out potentially bad candidates (e.g., very short)
            candidates = [c for c in candidates if len(c.strip()) > 10]
            
            # If after filtering, we have no candidates or too few, try to generate at least one default
            if not candidates:
                candidates = [_generate_response(base_prompt_for_candidates, tokenizer, model, **generation_params)]
            
            # Create a numbered list of candidates for the meta-evaluator
            choices_list_str = '\n'.join(f"Candidate {i+1}: {c}" for i, c in enumerate(candidates))
            
            # Meta-evaluator prompt to choose a number
            evaluator_prompt = (
                f"You are a meta-evaluator. Evaluate the following candidates and select the best answer for the question: '{raw_instruction}'\n\n"
                f"Candidates:\n{choices_list_str}\n\n"
                f"Please choose the number of the best answer (e.g., '1', '2', etc.). Your choice:"
            )
            
            # Generate the meta-evaluator's choice (should be a number)
            chosen_number_str = _generate_response(evaluator_prompt, tokenizer, model, **generation_params)
            
            # Attempt to extract the number and select the candidate
            chosen_index = -1
            try:
                # Use regex to find the first number in the string
                match = re.search(r'\b(\d+)\b', chosen_number_str)
                if match:
                    chosen_index = int(match.group(1)) - 1 # Convert to 0-based index
                
                if 0 <= chosen_index < len(candidates):
                    resp = candidates[chosen_index]
                else:
                    # Fallback if chosen number is out of range
                    print(f"Warning: Meta-evaluator chose {chosen_index+1} (out of range). Falling back to first candidate.")
                    resp = candidates[0] 
            except ValueError:
                # Fallback if no valid number can be extracted
                print(f"Warning: Meta-evaluator did not provide a valid number '{chosen_number_str}'. Falling back to first candidate.")
                resp = candidates[0]
            
            fmt = evaluator_prompt
        elif technique == 'iterative_refinement':
            conversation_history = []
            
            # Initial generation
            initial_prompt_for_model = f"Your role is to meticulously execute instructions and provide concise, relevant output.\n\n {raw_instruction}"
            current_response = _generate_response(initial_prompt_for_model, tokenizer, model, **generation_params)
            
            conversation_history.append(f"User: {initial_prompt_for_model}\nAssistant: {current_response}") 
            
            k_iterations = 2 # Number of refinement turns
            
            for i in range(k_iterations - 1): 
                # Construct refinement prompt
                refinement_prompt_text = f"Your role is to meticulously refine previous answers based on the conversation history."
                refinement_prompt_text += "\n\n".join(conversation_history)
                refinement_prompt_text += f"\n\nRefinement Request: Please refine the last assistant answer to be more precise."
                refinement_prompt_text += f"\n\nRefined Answer:" 

                # Generate refined response
                refined_response = _generate_response(refinement_prompt_text, tokenizer, model, **generation_params)
                
                conversation_history.append(f"User: Refinement Request: Please refine the last assistant answer to be more precise.\nAssistant: {refined_response}")
                current_response = refined_response 

            resp = current_response # Final response after iterations
            fmt = initial_prompt_for_model # Store the initial formatted prompt for consistency

        else: # Default case: No specific technique
            fmt = f"Your role is to meticulously execute instructions and provide concise, relevant output.\n\n {raw_instruction}"
            resp = _generate_response(fmt, tokenizer, model, **generation_params)
        
        # Collect data for CSV
        original_instructions.append(e['instruction']) # Keep original instruction
        formatted_prompts.append(fmt) 
        generated_responses.append(resp) 
        perturbation_types.append(e['perturbation'])
    
    # POSIX calculation
    # Ensure there are enough valid prompts and responses for POSIX calculation
    valid_indices = [i for i, r in enumerate(generated_responses) if len(r.strip()) > 0 and len(formatted_prompts[i].strip()) > 0]
    
    if len(valid_indices) < 2:
        psi = 0.0 
        per_type = {t: None for t in sorted(set(perturbation_types) - {'original'})}
        avg_no_noise = 0.0
    else:
        # Filter prompts and responses to only include valid ones for POSIX
        filtered_prompts = [formatted_prompts[i] for i in valid_indices]
        filtered_responses = [generated_responses[i] for i in valid_indices]
        filtered_perturbations = [perturbation_types[i] for i in valid_indices]

        psi = compute_posix(filtered_prompts, filtered_responses, tokenizer, model)
        
        per_type = {}
        types = sorted(set(filtered_perturbations) - {'original'})
        for t in types:
            idx = [i for i,p in enumerate(filtered_perturbations) if p in ('original', t)]
            if len(idx) > 1: 
                ps_subset = [filtered_prompts[i] for i in idx]
                rs_subset = [filtered_responses[i] for i in idx]
                per_type[t] = compute_posix(ps_subset, rs_subset, tokenizer, model)
            else:
                per_type[t] = None
        
        noise_free_values = [v for k,v in per_type.items() if k!='noise_injection' and v is not None]
        avg_no_noise = np.mean(noise_free_values) if noise_free_values else 0.0

    # Prepare records for DataFrame
    recs = []
    for i, e in enumerate(entries): # Iterate through original entries to link data
        rec = {
            'group_id': e['group_id'],
            'perturbation': e['perturbation'],
            'original_instruction': original_instructions[i],
            'formatted_prompt': formatted_prompts[i],
            'generated_response': generated_responses[i],
            'posix_all': psi, # Overall POSIX for the group
            'posix_all_but_noise': avg_no_noise
        }
        rec.update({f'posix_{k}':v for k,v in per_type.items()}) # POSIX per perturbation type
        recs.append(rec)
    return recs, psi

# --- Main Execution Block ---
if __name__=='__main__':
    data = [] 
    with open(INPUT_FILE, 'r') as f:
        for line in f:
            data.append(json.loads(line)) 
    print('Loaded data.')
    
    # Group data by 'group_id' for POSIX calculation
    grp = defaultdict(list)
    print('Starting grouping loop.')
    for x in data:
        grp[x['group_id']].append(x)
    ids = sorted(grp)
    print('Finished grouping.')

    # Determine batch of group IDs for this SLURM task
    batch = ids[TASK_ID * BATCH_SIZE : (TASK_ID + 1) * BATCH_SIZE]
    
    # Determine technique from environment variable
    technique = None if TECH_ARG == 'None' else TECH_ARG 
    print(f"Processing with technique: {technique}")

    # Load Tokenizer
    print('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(FINETUNE_PATH if FINETUNE_FLAG else MODEL_ID)
    # Set pad_token if not defined, common for Llama models
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    print('Tokenizer loaded.')

    # Load Model
    print('Loading model...')
    model = AutoModelForCausalLM.from_pretrained(FINETUNE_PATH if FINETUNE_FLAG else MODEL_ID, 
                                                torch_dtype=torch.float16, 
                                                device_map='auto')
    model.eval() # Set model to evaluation mode (disables dropout, etc.)
    print('Model loaded. Starting main processing loop...')
    
    all_records, overall_scores = [], []
    for gid in tqdm(batch, desc='Processing Groups'):
        records_for_group, group_posix_score = process_group(grp[gid], tokenizer, model, technique)
        all_records.extend(records_for_group)
        overall_scores.append(group_posix_score)
    print('Finished main processing loop.')

    # Save results to CSV
    df = pd.DataFrame(all_records)
    
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Construct filename based on parameters
    model_name_for_file = MODEL_ID.replace('/', '_').split('/')[-1]
    fname = f"{results_dir}/posix_{TECH_ARG}_{'ft_' if FINETUNE_FLAG else ''}{model_name_for_file}_{TASK_ID}.csv"
    df.to_csv(fname, index=False)
    print(f"Results saved to: {fname}")
    
    # Print average POSIX score for this batch
    if overall_scores:
        print('Average POSIX for this batch:', np.mean(overall_scores))
    else:
        print('No scores to compute average POSIX for this batch.')