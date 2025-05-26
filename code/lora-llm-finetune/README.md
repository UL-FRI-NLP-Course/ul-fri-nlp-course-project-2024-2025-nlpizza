# LoRA Finetuning on Prompt Variants

This folder contains one of the experimental techniques in our broader project on improving the prompt sensitivity of large language models (LLMs). Specifically, this module applies **Low-Rank Adaptation (LoRA)** for **parameter-efficient fine-tuning** using **prompt variant groupings** to reduce prompt sensitivity. This folder provides a general-purpose framework that is model-agnostic and could support most HuggingFace-compatible causal language model (e.g., Falcon, Mistral, LLaMA).


## Objective

Language models often respond inconsistently when the wording of a prompt changes, even if the underlying intent is the same. This sensitivity can reduce reliability and generalization. In this experiment, we fine-tune a causal LLM using grouped prompt variants—multiple instructions that share the same meaning and target output.

By training on this structured dataset, the model is encouraged to:
- Respond more consistently across paraphrased inputs
- Focus on underlying intent rather than surface form
- Generalize better to unseen formulations of known tasks


## Dataset Format

The expected dataset is a `.jsonl` file with the following fields per line:
- instruction: the input prompt
- output: the corresponding target response
- group_id: identifier for a group of semantically similar prompts

```json
{
  "instruction": "Rephrase this sentence: The cat sat on the mat.",
  "output": "The feline rested on the rug.",
  "group_id": 42
}
In our training, we use original and variants of the original prompt for training.


## Method Overview
- Uses HuggingFace Transformers and PEFT (Parameter-Efficient Fine-Tuning) libraries
- Supports 4-bit quantized models with bitsandbytes (optional)
- Loads any causal language model via AutoModelForCausalLM
- Applies LoRA adapters to selected modules for efficient training
- Compatible with local or SLURM-based HPC environments

## Usage

### Installation

```bash
pip install -r requirements.txt

### Running the Script
The script is configurable with arguments for model name, dataset path, LoRA settings, number of groups to sample, and more. Example of training Falcon-7B instruct model.

```bash
python finetune.py \
  --model_name "tiiuae/Falcon-7B-Instruct" \
  --dataset_path data/sft_train.jsonl \
  --output_dir falcon7b_lora_output \
  --quant4bit \
  --lora_r 16 \
  --lora_alpha 32 \
  --target_modules "query_key_value,dense,dense_h_to_4h,dense_4h_to_h"


## Output

After training, the model and LoRA adapters will be saved to the specified output directory:
```pgqsl
falcon7b_lora_output/
├── adapter_config.json
├── adapter_model.bin
├── tokenizer_config.json
├── tokenizer.json

These can be loaded at inference time using:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained("tiiuae/Falcon-7B-Instruct")
model = PeftModel.from_pretrained(model, "falcon7b_lora_output")
tokenizer = AutoTokenizer.from_pretrained("falcon7b_lora_output")



---

```markdown
## Credits

This setup builds on the following work:

- **LoRA: Low-Rank Adaptation of Large Language Models**  
  Hu et al. (2021) — [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)

- **POSIX: A Prompt Sensitivity Index For Large Language Models**  
  Chatterjee et al. (2024) — [https://arxiv.org/abs/2410.02185](https://arxiv.org/abs/2410.02185)

