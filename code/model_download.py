from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# Set your desired cache directory
cache_dir = "/d/hpc/projects/onj_fri/nlpizza/huggingface"
os.makedirs(cache_dir, exist_ok=True)

models = {
    "baseline":  "mosaicml/phi-2",
    "llama":     "meta-llama/Llama-2-7b-chat-hf",
    "mistral":   "mistralai/Mistral-7B-Instruct-v0.1",
    "reasoning": "deepseek-ai/deepseek-coder-6.7b-instruct",
}

for name, repo in models.items():
    print(f"Downloading {name} from {repo}...")
    tokenizer = AutoTokenizer.from_pretrained(repo, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(repo, cache_dir=cache_dir)
    print(f"✅ Done: {name}")


# to use we do:

'''model_id = "baseline"  # or any model downloaded

tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="/d/hpc/projects/onj_fri/nlpizza/huggingface")
model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir="/d/hpc/projects/onj_fri/nlpizza/huggingface")

# Prepare prompt
prompt = "Why is the sky blue? Think step by step."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))'''