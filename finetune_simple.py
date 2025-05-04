#!/usr/bin/env python
# finetune_simple.py
"""
4-bit QLoRA fine-tuning script for Llama models with memory optimizations
Using standard libraries without unsloth
"""

import os
import gc
import argparse
import json
from pathlib import Path

import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_dataset
from accelerate import init_empty_weights

# Enable memory optimizations
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def main():
    parser = argparse.ArgumentParser(description="Simple 4-bit QLoRA fine-tuning")
    parser.add_argument("--base_model", required=True,
                        help="Path or HF hub name of the base model (e.g. Llama-4-Scout)")
    parser.add_argument("--dataset", required=True,
                        help="JSONL file with one {\"text\": ...} per line")
    parser.add_argument("--save_path", default="scout_4bit_lora",
                        help="Where to store the fine-tuned checkpoint")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size per GPU/TPU core")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=1024,
                        help="Maximum sequence length")
    args = parser.parse_args()

    # Sanity checks
    if not Path(args.dataset).exists():
        raise FileNotFoundError(f"Dataset file not found: {args.dataset}")

    print("[INFO] Starting 4-bit QLoRA fine-tune with memory optimizations...")
    print(f"[INFO] Device count: {torch.cuda.device_count()}")
    print(f"[INFO] Device 0 memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Clear any existing CUDA caches
    torch.cuda.empty_cache()
    gc.collect()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set up quantization configuration with maximum memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    print("[INFO] Loading model with 4-bit quantization...")
    
    # Load model with quantization and CPU offloading where needed
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        max_memory={0: "72GiB", "cpu": "64GiB"},  # Set explicit memory limits
    )
    
    # Disable KV caching during training (done after model initialization)
    model.config.use_cache = False
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    print("[INFO] Setting up LoRA adapters...")
    
    # Focus only on router and gate components (like the original script)
    target_modules = []
    for name, _ in model.named_modules():
        if "router.classifier" in name or "gate" in name:
            target_modules.append(name)  # Extract the parent module name
    
    # If no router/gate components were found, fall back to these basic modules
    if not target_modules:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj"
        ]
    
    # LoRA configuration with smaller rank for memory efficiency
    lora_config = LoraConfig(
        r=8,  # Lower rank 
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # Load dataset from JSONL
    print("[INFO] Loading and processing dataset...")
    dataset = load_dataset("json", data_files=args.dataset, split="train")
    
    # Create instructional prompts from the data
    def format_prompt(example):
        text = example["text"]
        return {"text": text}
    
    formatted_dataset = dataset.map(format_prompt)
    
    # Tokenize the dataset
    def tokenize_function(examples):
        results = tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )
        results["labels"] = results["input_ids"].copy()
        return results
    
    tokenized_dataset = formatted_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=[col for col in formatted_dataset.column_names if col != "text"],
    )
    
    # Set up training arguments with memory optimizations
    training_args = TrainingArguments(
        output_dir=args.save_path,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=16,  # Increased for lower memory usage
        max_grad_norm=0.3,  # Lower max gradient norm
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        bf16=True,  # Use bfloat16 instead of fp16 for better numerical stability
        save_strategy="epoch",
        logging_steps=10,
        optim="paged_adamw_32bit",  # 32-bit optimizer for better stability
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        gradient_checkpointing=True,
        save_total_limit=1,  # Only keep the latest checkpoint
        dataloader_drop_last=True,
        report_to="none",  # Disable wandb, etc.
    )
    
    # Custom data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False
    )
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # Start training
    print("[INFO] Starting training...")
    trainer.train()
    
    # Save model
    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)
    
    print("[INFO] Training complete. Checkpoint saved to", args.save_path)


if __name__ == "__main__":
    main()
