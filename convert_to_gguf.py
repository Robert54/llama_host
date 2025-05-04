# convert_to_gguf.py
import types, torch
from unsloth import FastLanguageModel
from unsloth.save import patch_saving_functions, install_llama_cpp_blocking
from peft import PeftModel
from transformers import AutoTokenizer
import os

BASE_MODEL   = "/home/ubuntu/llama-hosting/Llama-4-Scout-17B-16E"
LORA_DIR     = "/home/ubuntu/llama-hosting/scout_4bit_lora"
OUT_DIR      = "/home/ubuntu/llama-hosting/scout_4bit_gguf"
# QUANT_TYPE   = "q8_0"               # 支持: bf16 / f16 / f32 / q8_0

# 1) 载入 4-bit 基座
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name               = BASE_MODEL,
    max_seq_length           = 4096,
    load_in_4bit             = False,   # 加载全精度，后续再统一量化
    device_map               = "auto",           # 自动把权重分布到所有可见 GPU
    gpu_memory_utilization   = 0.9,               # 每卡用 90% 显存
)

# 2) 注入 LoRA，并合并到基座，移除 Peft 包装
peft_model = PeftModel.from_pretrained(model, LORA_DIR, is_trainable=False)
model = peft_model.merge_and_unload()  # 返回合并后的 Llama 模型

# Re-patch saving functions on the new PeftModel wrapper so we can use high-end quantization
patch_saving_functions(model)

# ----- Fix Llama-4 tokenizer so tokenizer("A") works -----
def _make_tokenizer_simple(tok):
    if hasattr(tok, "_unsloth_patched_simple"):
        return tok

    original_class_call = tok.__class__.__call__  # unbound

    def _simple_call(self, *args, **kwargs):
        # If first positional arg is a str, treat it as text=
        if args and isinstance(args[0], str) and "text" not in kwargs:
            kwargs["text"] = args[0]
            args = args[1:]
        return original_class_call(self, *args, **kwargs)

    tok.__call__ = types.MethodType(_simple_call, tok)
    tok._unsloth_patched_simple = True
    return tok

# Patch class-level __call__ (special method lookup)
cls_tok = tokenizer.__class__
if not getattr(cls_tok, "_unsloth_patched_simple", False):
    orig_call_cls = cls_tok.__call__

    def _cls_simple_call(self, *args, **kwargs):
        if args and isinstance(args[0], str) and "text" not in kwargs:
            kwargs["text"] = args[0]
            args = args[1:]
        return orig_call_cls(self, *args, **kwargs)

    cls_tok.__call__ = _cls_simple_call
    cls_tok._unsloth_patched_simple = True

# Fallback: override fix_tokenizer_bos_token to no-op if still problematic
import unsloth.save as _us_save
def _no_fix(tok):
    return False, None
_us_save.fix_tokenizer_bos_token = _no_fix

tokenizer = _make_tokenizer_simple(tokenizer)

# Patch check_if_sentencepiece_model to bypass name_or_path error
def _no_sentencepiece(model_obj):
    return False
_us_save.check_if_sentencepiece_model = _no_sentencepiece

# 若没编译好的 llama.cpp, 先安装
if not (os.path.exists("llama.cpp/llama-quantize") or os.path.exists("llama.cpp/quantize")):
    print("[Info] llama.cpp not found. Building locally, this can take a few minutes...")
    install_llama_cpp_blocking()

# 3) 存成 GGUF（高端量化 q4_k_m）
print("save fn:", model.save_pretrained_gguf.__name__)
print("model type:", type(model))

model.save_pretrained_gguf(
    OUT_DIR,
    tokenizer           = tokenizer,
    quantization_method  = "q4_k_m",
)

tokenizer.save_pretrained(OUT_DIR)   # 方便 llama.cpp 调用
print("GGUF 已保存到:", OUT_DIR)