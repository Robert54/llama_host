#!/usr/bin/env python
# make_ft_dataset.py
"""
Build a 10 k instruction‑tuning set:
Alpaca 3 k  +  Self‑Instruct 2 k  +  Dolly 5 k
Output: data/ft_mix_10k.jsonl  (one {"text": ...} per line)
"""

import os, json
from datasets import load_dataset, concatenate_datasets, disable_caching, Dataset

disable_caching()   # 减少磁盘占用

# 1) 下载并随机抽样
alpaca_ds = load_dataset("tatsu-lab/alpaca", split="train").shuffle(seed=42)
alpaca = alpaca_ds.select(range(3000))

self_i_ds = load_dataset("yizhongw/self_instruct", "self_instruct", split="train").shuffle(seed=42)
self_i = self_i_ds.select(range(2000))

dolly_ds = load_dataset("databricks/databricks-dolly-15k", split="train").shuffle(seed=42)
dolly = dolly_ds.select(range(5000))

# 2) 统一转 Llama 模板
# Define separate conversion functions for each dataset
def to_llama_alpaca(example):
    inst = example["instruction"]
    out = example["output"]
    return {"text": f"<s>[INST] {inst.strip()} [/INST]\n{out.strip()}"}

def to_llama_self_instruct(example):
    inst = example["prompt"]
    out = example["completion"]
    return {"text": f"<s>[INST] {inst.strip()} [/INST]\n{out.strip()}"}

def to_llama_dolly(example):
    inst = example["instruction"]
    out = example["response"]
    return {"text": f"<s>[INST] {inst.strip()} [/INST]\n{out.strip()}"}

# Apply the appropriate conversion function to each dataset
alpaca = alpaca.map(to_llama_alpaca)
self_i = self_i.map(to_llama_self_instruct)
dolly = dolly.map(to_llama_dolly)

# 3) 合并 & 打乱
mix = concatenate_datasets([alpaca, self_i, dolly]).shuffle(seed=123)

# 4) 保存 JSONL
os.makedirs("data", exist_ok=True)
with open("data/ft_mix_10k.jsonl", "w", encoding="utf-8") as f:
    for ex in mix:
        json.dump(ex, f, ensure_ascii=False)
        f.write("\n")

print("✅ Saved 10 k samples to data/ft_mix_10k.jsonl")
