#!/usr/bin/env python
# finetune_qlora.py
"""
4‑bit QLoRA fine‑tuning with Unsloth ≥2025.3 (FastLanguageModel API)
- Router‑only LoRA for Llama‑4‑Scout
- TRL SFTTrainer back‑end
"""

import argparse, json, os, torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig
from peft import get_peft_model, LoraConfig, TaskType

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--dataset",    required=True)
    p.add_argument("--save_path",  default="scout_4bit_lora")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs",     type=int, default=1)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--max_seq_length", type=int, default=512)
    return p.parse_args()

def main():
    args = parse_args()

    # 1) 载入 4‑bit 模型 —— FastLanguageModel 会自动调用 bitsandbytes
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name      = args.base_model,
        max_seq_length  = args.max_seq_length,
        load_in_4bit    = True,
        dtype           = torch.float16,
        device_map      = "auto",
    )

    # Patch Llama4TextModel._update_causal_mask to ensure tuple output (workaround for None bug)
    try:
        from transformers.models.llama4.modeling_llama4 import Llama4TextModel
        if not hasattr(Llama4TextModel, "_original_update_causal_mask"):
            Llama4TextModel._original_update_causal_mask = Llama4TextModel._update_causal_mask
            def _patched_update_causal_mask(self, *args, **kwargs):
                out = self._original_update_causal_mask(*args, **kwargs)
                # Some compiled variants erroneously return None. Convert to (None, None).
                if out is None:
                    return (None, None)
                return out
            Llama4TextModel._update_causal_mask = _patched_update_causal_mask
            print("[INFO] Patched Llama4TextModel._update_causal_mask to prevent None return crash.")
    except ImportError:
        pass

    # 2) Identify LoRA target modules (router & gate)
    target = [name for name, _ in model.named_modules()
              if name.split(".")[-1] in {"router", "gate", "gate_proj"}]

    # Fallback to common projection layers if nothing matched
    if not target:
        target = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        print(f"[WARN] No router/gate modules found; using default targets: {target}")

    # Apply LoRA using get_peft_model
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target
    )
    
    # Apply PEFT
    model = get_peft_model(model, peft_config)
    
    # Enable gradient checkpointing for memory efficiency
    model.enable_input_require_grads()
    model.config.use_cache = False

    # 3) 读数据
    dataset = load_dataset("json", data_files=args.dataset, split="train")

    # 4) Trainer (TRL‑SFTTrainer)
    training_args = SFTConfig(
        output_dir = args.save_path,
        per_device_train_batch_size = args.batch_size,
        num_train_epochs = args.epochs,
        learning_rate = args.lr,
        bf16 = False,
        fp16 = True,
        max_seq_length = args.max_seq_length,
        logging_steps = 10,
        save_strategy = "epoch",
        optim = "paged_adamw_8bit",
        gradient_checkpointing = True,
    )
    trainer = SFTTrainer(
        model          = model,
        train_dataset  = dataset,
        tokenizer      = tokenizer,
        args           = training_args,
    )
    trainer.train()

    # 5) 保存：LoRA 已 merge + 4‑bit 权重
    # For newer unsloth versions, we save the adapter
    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)
    print("✅  Done! 4‑bit checkpoint in", args.save_path)

if __name__ == "__main__":
    main()