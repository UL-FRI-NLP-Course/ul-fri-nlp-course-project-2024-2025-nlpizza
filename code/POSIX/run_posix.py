import torch
import numpy as np
import json
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import sys


def normalize_whitespace(text):
    """Collapse extra spaces and remove space before punctuation."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([?.!,:])', r'\1', text)
    return text.strip()


def clean_prompt(text):
    """Remove prompt/answer prefixes and normalize spacing."""
    text = re.sub(r"^(question|q|Q|Q::|q:::)[\s:：]*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[\n\r]?(answer|a|a:::|A:|A::)[\s:：]*$", "", text, flags=re.IGNORECASE)
    return normalize_whitespace(text)


def extract_clean_response(generated_text, formatted_prompt, cleaned_prompt):
    """Extract the actual response from the model output."""
    gen = normalize_whitespace(generated_text)
    prompt = normalize_whitespace(formatted_prompt)
    cleaned = normalize_whitespace(cleaned_prompt).rstrip(" ?!.")

    if gen.startswith(prompt):
        gen = gen[len(prompt):].strip()

    # Remove repeated prompt or assistant markers
    gen = re.sub(rf"^(?:{re.escape(cleaned)})([\s\?\.!:\-]*)", "", gen, flags=re.IGNORECASE)
    gen = re.sub(r"^(assistant|a|answer)[\s:：\-]*", "", gen, flags=re.IGNORECASE)

    return gen.strip()


def compute_log_prob(prompt, response, tokenizer, model):
    """Compute total log probability of response given prompt."""
    formatted_prompt = f"You're a helpful assistant. Please answer clearly:\n\n{prompt.strip()}"

    #if chain-of-thought
    # cot_suffix = " Let's think step by step."
    # formatted_prompt = f"You're a helpful assistant. Please answer clearly:\n\n{prompt.strip()}{cot_suffix}"

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


def process_prompt_set(set_id, prompt_set, tokenizer, model):
    """Generate responses and compute POSIX score for a single prompt set."""
    responses = []
    formatted_prompts = []

    for prompt in prompt_set:
        cleaned = clean_prompt(prompt)
        formatted = f"You're a helpful assistant. Please answer clearly:\n\n{prompt.strip()}"

        #if chain-of-thought
        # cot_suffix = " Let's think step by step."
        # formatted = f"You're a helpful assistant. Please answer clearly:\n\n{prompt.strip()}{cot_suffix}"

        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
        output = model.generate(
            inputs.input_ids,
            max_new_tokens=30,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False
        )
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        response = extract_clean_response(generated_text, formatted, cleaned)
        responses.append(response)
        formatted_prompts.append(formatted)

    # Compute POSIX score
    N = len(prompt_set)
    log_probs = np.zeros((N, N))
    lengths = np.zeros(N)

    for i in range(N):
        for j in range(N):
            logp, L = compute_log_prob(prompt_set[i], responses[j], tokenizer, model)
            log_probs[i][j] = logp
            if i == j:
                lengths[j] = L

    psi = sum(
        abs((log_probs[i][j] - log_probs[i][i]) / lengths[i])
        for i in range(N) for j in range(N) if i != j
    ) / (N * (N - 1))

    # Collect results
    records = []
    for prompt, response in zip(prompt_set, responses):
        records.append({
            "set_id": set_id,
            "original_prompt": prompt.strip(),
            "cleaned_prompt": clean_prompt(prompt),
            "response": response.strip(),
            "posix_score": psi
        })

    return psi, records


def main():
    # Get range from SLURM script
    start_index = int(sys.argv[1])
    end_index = int(sys.argv[2])

    # Load prompts
    with open("alpaca_prompts.json", "r") as f:
        alpaca_data = json.load(f)
    intent_prompt_sets = [item["prompts"] for item in alpaca_data[start_index:end_index]]
    print(f"Processing prompt sets {start_index} to {end_index} ({len(intent_prompt_sets)} total)")


    # Load model
    model_id = "tiiuae/falcon-7b-instruct"
    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()

    # Process each prompt set
    all_records = []
    posix_scores = []

    for set_id, prompt_set in enumerate(tqdm(intent_prompt_sets, desc="Prompt Sets"), start=start_index):
        psi, records = process_prompt_set(set_id, prompt_set, tokenizer, model)
        posix_scores.append(psi)
        all_records.extend(records)

    # Save results
    df_all = pd.DataFrame(all_records)
    df_all.to_csv("posix_prompt_response_scores.csv", index=False)
    print(f"\nAverage POSIX score: {np.mean(posix_scores):.4f}")
    print(df_all.head())
    print("Saved to posix_prompt_response_scores.csv")


if __name__ == "__main__":
    main()
