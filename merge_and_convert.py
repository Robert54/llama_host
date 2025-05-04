# merge_and_convert.py
import subprocess, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

BASE_MODEL = "/home/ubuntu/llama-hosting/Llama-4-Scout-17B-16E"   # ★换成你的原始基座
LORA_DIR   = "/home/ubuntu/llama-hosting/scout_4bit_lora"                         # ★LoRA 保存目录
MERGED_DIR = "/home/ubuntu/llama-hosting/scout_full_bf16"                       # 输出目录

os.makedirs(MERGED_DIR, exist_ok=True)

print("1) 载入基座 (BF16)…")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cpu"
)
print("2) 添加 language_model wrapper 以匹配 LoRA target modules…")
import torch.nn as nn
class _LangWrapper(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.model = inner  # expose .model
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
# only add if not already present
if not hasattr(model, "language_model"):
    model.add_module("language_model", _LangWrapper(model.model if hasattr(model, "model") else model))
print("3) 加载 LoRA 并合并…")
model = PeftModel.from_pretrained(model, LORA_DIR)
model = model.merge_and_unload()

print("4) 保存合并后模型…")
model.save_pretrained(MERGED_DIR, safe_serialization=True)
AutoTokenizer.from_pretrained(BASE_MODEL).save_pretrained(MERGED_DIR)

print("5) 转 GGUF (BF16)…")
subprocess.check_call([
    "python", "llama.cpp/convert_hf_to_gguf.py",
    MERGED_DIR,
    "--outfile", f"{MERGED_DIR}.gguf",
    "--outtype", "bf16",
])

print("6) 量化为 Q4_K_M…")
subprocess.check_call([
    "llama.cpp/build/bin/llama-quantize",
    f"{MERGED_DIR}.gguf",
    "scout.Q4_K_M.gguf",
    "q4_k_m",
    str(os.cpu_count())
])

print("全部完成 -> scout.Q4_K_M.gguf")