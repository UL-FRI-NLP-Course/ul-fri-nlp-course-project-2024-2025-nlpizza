import torch
import numpy as np
import json
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

def normalize_whitespace(text):
    text = re.sub(r'\s+', ' ', text)                   # Collapse multiple spaces
    text = re.sub(r'\s+([?.!,:])', r'\1', text)        # Remove space before punctuation
    return text.strip()


# Load model + tokenizer
model_id = "tiiuae/falcon-rw-1b"  # Or: "mistralai/Mistral-7B-Instruct-v0.2"
print(f"Loading model: {model_id}")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

# Load prompt sets
with open("/content/alpaca_prompts.json", "r") as f:
    alpaca_data = json.load(f)

num_sets = 2  # Adjust as needed
intent_prompt_sets = [item["prompts"] for item in alpaca_data[:num_sets]]
print(f"Processing {len(intent_prompt_sets)} prompt sets")

# Prompt cleaner
def clean_prompt(text):
    text = re.sub(r"^(question|q|Q|Q::|q:::)[\s:：]*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[\n\r]?(answer|a|a:::|A:|A::)[\s:：]*$", "", text, flags=re.IGNORECASE)
    return normalize_whitespace(text)




# Log prob calculator
def compute_log_prob(prompt, response):
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


def extract_clean_response(generated_text, formatted_prompt, cleaned_prompt):
    gen = normalize_whitespace(generated_text)
    prompt = normalize_whitespace(formatted_prompt)
    cleaned = normalize_whitespace(cleaned_prompt).rstrip(" ?!.")

    if gen.startswith(prompt):
        gen = gen[len(prompt):].strip()

    # Remove repeated cleaned prompt if echoed
    pattern = re.escape(cleaned)
    gen = re.sub(rf"^(?:{pattern})([\s\?\.!:\-]*)", "", gen, flags=re.IGNORECASE)

    # Remove assistant-like prefixes again
    gen = re.sub(r"^(assistant|a|answer)[\s:：\-]*", "", gen, flags=re.IGNORECASE)

    return gen.strip()








# Main loop
records = []
posix_scores = []

for set_id, prompt_set in enumerate(tqdm(intent_prompt_sets, desc="Prompt Sets")):
    responses = []
    
    for prompt in prompt_set:
        cleaned = clean_prompt(prompt)
        formatted = f"You're a helpful assistant. Please answer clearly:\n\n{prompt.strip()}"
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
        output = model.generate(
            inputs.input_ids,
            max_new_tokens=30,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False
        )
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        response = extract_clean_response(generated_text, formatted,cleaned)
        responses.append(response)

    N = len(prompt_set)
    log_probs = np.zeros((N, N))
    lengths = np.zeros(N)

    for i in range(N):
        for j in range(N):
            logp, L = compute_log_prob(prompt_set[i], responses[j])
            log_probs[i][j] = logp
            if i == j:
                lengths[j] = L

    psi = 0.0
    for i in range(N):
        for j in range(N):
            if i != j:
                psi += abs((log_probs[i][j] - log_probs[i][i]) / lengths[i])
    psi /= (N * (N - 1))
    posix_scores.append(psi)


    for prompt, response in zip(prompt_set, responses):
      records.append({
          "set_id": set_id,
          "original_prompt": prompt.strip(),           # Save raw for traceability
          "cleaned_prompt": clean_prompt(prompt),      # What the model saw
          "response": response.strip(),
          "posix_score": psi
      })


#  Save results
df_all = pd.DataFrame(records)
print(f"\nAverage POSIX score: {np.mean(posix_scores):.4f}")
print(df_all.head())
df_all.to_csv("posix_prompt_response_scores.csv", index=False)
print("\nSaved to posix_prompt_response_scores.csv")
