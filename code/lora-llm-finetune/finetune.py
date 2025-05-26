import argparse
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model, TaskType

def format_prompt(example):
    return f"{example['instruction'].strip()} {example['output'].strip()}"

def tokenize(example, tokenizer, max_length):
    tokens = tokenizer(example["text"], truncation=True, padding="max_length", max_length=max_length)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

def main(args):
    # Load model and tokenizer
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.quant4bit,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    ) if args.quant4bit else None

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules.split(","),
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    df = pd.read_json(args.dataset_path, lines=True)
    if args.sample_groups > 0:
        group_ids = df["group_id"].drop_duplicates().sample(n=args.sample_groups, random_state=42)
        df = df[df["group_id"].isin(group_ids)].reset_index(drop=True)

    df["text"] = df.apply(format_prompt, axis=1)
    dataset = Dataset.from_pandas(df[["text"]].copy())

    dataset = dataset.map(lambda x: tokenize(x, tokenizer, args.max_length), remove_columns=["text"])

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_dir="./logs",
        save_strategy="epoch",
        fp16=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer
    )

    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("LoRA finetuning complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Base HF model name")
    parser.add_argument("--dataset_path", type=str, default="data/sft_train.jsonl")
    parser.add_argument("--output_dir", type=str, default="lora_output")
    parser.add_argument("--quant4bit", action="store_true")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--sample_groups", type=int, default=500)
    args = parser.parse_args()

    main(args)
