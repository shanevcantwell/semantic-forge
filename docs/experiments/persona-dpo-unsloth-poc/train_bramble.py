#!/usr/bin/env python
"""Persona-DPO P5 machinery PoC — bramble QLoRA-DPO on gemma-3-12b-it.

Pre-run record + environment manifest: README.md in this directory (committed before this run).
Effective env: ~/.unsloth/studio/unsloth_studio/bin/python (deltas E1–E6 in README).

Run:   python docs/experiments/persona-dpo-unsloth-poc/train_bramble.py     (cwd = repo root)
Exits: 0 complete · 1 integrity halt (non-finite loss, per record HALT rules)
       2 resource/gate failure · 3 disk/cache trap · other uncaught = traceback to log.

Pinned hyperparameters (see README "Budget & halt rules"): max seq 2048; prompt/completion caps 512/512;
LoRA r=16 alpha=32 target all-linear dropout 0 seed 0; DPO beta 0.1 sigmoid loss, lr 1e-6, warmup 0,
3 epochs × N=5 rows, batch 1 / grad-accum 1, no intermediate checkpoints.
"""
import glob
import json
import os
import shutil
import sys
import time
import urllib.request

t_start = time.time()

# --- environment pin BEFORE any heavy import (E5: /mnt/storage is RO in-container) ---
os.environ.setdefault("HF_HOME", "/home/node/hf-cache")
OUT_BASE = "/home/node/persona-dpo-poc"


def gate(cond, msg, code):
    if not cond:
        print(f"GATE FAIL [exit {code}]: {msg}", flush=True)
        sys.exit(code)


# --- start gates (readbacks land in the run log for banking) ---
free_g = shutil.disk_usage("/home/node").free / 2**30
print(f"[disk] /home/node free at start: {free_g:.1f} GiB", flush=True)
gate(free_g >= 10, f"need >=10 GiB free (7.26 GiB model + margin), have {free_g:.1f}", 3)

import torch

print(
    f"[torch] {torch.__version__} | cuda={torch.cuda.is_available()} "
    f"build={torch.version.cuda} dev={torch.cuda.get_device_name(0)}",
    flush=True,
)
vr_free, vr_total = torch.cuda.mem_get_info()
print(f"[vram] free {vr_free/2**30:.1f}/{vr_total/2**30:.1f} GiB (gate >= 25)", flush=True)
gate(vr_free / 2**30 >= 25, f"VRAM gate: have {vr_free/2**30:.1f} GiB free < 25", 2)

# --- E6 load order: unsloth FIRST (fix_trl_vllm_ascend + TRL DPO replacements), then trl ---
import unsloth  # noqa: F401 — import side effects are the point (E6 in README)

from huggingface_hub.constants import HF_HUB_CACHE

gate(
    str(HF_HUB_CACHE).startswith("/home/node"),
    f"HF cache redirect trap: effective hub cache = {HF_HUB_CACHE} (not under /home/node)",
    3,
)
print(f"[hf] hub cache = {HF_HUB_CACHE}", flush=True)

from unsloth import FastLanguageModel
from trl import DPOConfig, DPOTrainer  # TRL after unsloth (E6)
import torch.nn.functional as F

PROMPT_CAP, COMP_CAP, SEQ_CAP = 512, 512, 2048
MODEL_NAME = "unsloth/gemma-3-12b-it-bnb-4bit"  # pre-quant repo (7.26 GiB); no runtime-quant flag needed
POC = os.path.dirname(os.path.abspath(__file__))
DS_PATH = os.path.join(POC, "dataset", "bramble_dpo_v0.jsonl")
gate(os.path.isfile(DS_PATH), f"dataset missing: {DS_PATH} — run build_dataset.py first", 2)

# --- model + tokenizer (pre-quantized 4-bit base; bnb loading driven by the repo's own config) ---
print(f"[load] {MODEL_NAME} ...", flush=True)
t0 = time.time()
model, tokenizer = FastLanguageModel.from_pretrained(model_name=MODEL_NAME, max_seq_length=SEQ_CAP)
print(f"[load] done in {(time.time()-t0)/60:.1f} min", flush=True)

free_g2 = shutil.disk_usage("/home/node").free / 2**30
gate(
    free_g2 >= 6,
    f"post-download disk gate: only {free_g2:.1f} GiB free < 6 (training + artifact headroom)",
    3,
)
print(f"[disk] post-download free: {free_g2:.1f} GiB", flush=True)

# --- LoRA attach (pinned) ---
model = FastLanguageModel.get_peft_model(
    model, r=16, target_modules="all-linear", lora_alpha=32, lora_dropout=0, random_state=0
)
n_total = sum(p.numel() for p in model.parameters())
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
ratio = n_train / max(n_total, 1)
print(f"[lora] trainable {n_train/1e6:.2f}M / total {n_total/1e9:.2f}B ({100*ratio:.3f}%)", flush=True)
gate(ratio < 0.02, f"trainable ratio {100*ratio:.4f}% — not LoRA-only (<2% expected)", 2)

# --- dataset + preflight token caps (data problems surface HERE, before any optimizer step) ---
rows = [json.loads(l) for l in open(DS_PATH, encoding="utf-8") if l.strip()]
gate(len(rows) == 5, f"dataset drift: {len(rows)} rows != 5", 2)

max_p, max_c = 0, 0
for r_ in rows:
    p_ids = tokenizer.apply_chat_template(r_["prompt"], tokenize=True, add_generation_prompt=True)
    c_ids = tokenizer.encode(r_["chosen"], add_special_tokens=False)
    max_p = max(max_p, len(p_ids))
    max_c = max(max_c, len(c_ids))
print(
    f"[preflight] max prompt tokens={max_p} (cap {PROMPT_CAP}) | "
    f"max completion tokens={max_c} (cap {COMP_CAP})",
    flush=True,
)
gate(max_p < PROMPT_CAP and max_c < COMP_CAP, "token cap breach in dataset", 2)


# --- margin probe: mean-token logprob delta (chosen vs rejected), student policy ---
def compute_margins(tag):
    model.eval()
    recs = []
    with torch.no_grad():
        for i, r_ in enumerate(rows):
            p_ids = tokenizer.apply_chat_template(r_["prompt"], tokenize=True, add_generation_prompt=True)
            vals = {}
            for key in ("chosen", "rejected"):
                c_ids = torch.tensor(tokenizer.encode(r_[key], add_special_tokens=False))
                ids = torch.cat([torch.tensor(p_ids), c_ids]).unsqueeze(0).to(model.device)
                attn = torch.ones_like(ids)
                labels = torch.full_like(ids, -100)
                labels[0, len(p_ids):] = c_ids
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(input_ids=ids, attention_mask=attn).logits[0].float()  # T x V, f32
                sl = logits[:-1].contiguous().view(-1, logits.size(-1))
                sr = labels[:, 1:].view(-1)
                vals[key] = float(-F.cross_entropy(sl, sr, ignore_index=-100))  # mean token logprob
            recs.append(
                {
                    "scenario_index": i,
                    "chosen_mean_token_logprob": round(vals["chosen"], 6),
                    "rejected_mean_token_logprob": round(vals["rejected"], 6),
                    "delta_chosen_minus_rejected": round(vals["chosen"] - vals["rejected"], 6),
                }
            )
    os.makedirs(OUT_BASE, exist_ok=True)
    with open(os.path.join(OUT_BASE, f"margins_{tag}.json"), "w", encoding="utf-8") as fh:
        json.dump({"computed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rows": recs}, fh, indent=2)
    print(
        f"[margins:{tag}] "
        + " | ".join(f"idx{r_['scenario_index']} Δ={r_['delta_chosen_minus_rejected']:+.3f}" for r_ in recs),
        flush=True,
    )
    return recs


# --- DPO config (every kwarg verified present on trl 0.23.1 this session) ---
cfg = DPOConfig(
    output_dir=os.path.join(OUT_BASE, "bramble_dpo_v0-adapter"),
    beta=0.1,
    loss_type="sigmoid",
    max_prompt_length=PROMPT_CAP,
    max_completion_length=COMP_CAP,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=1e-6,
    warmup_steps=0,
    logging_steps=1,
    save_strategy="no",
    # no bf16=True: transformers 5.5's is_torch_bf16_gpu_available() is stale against torch 2.11+cu130
    # (wraps a removed torch attribute -> False, although torch.cuda.is_bf16_supported() is True — see README E7).
    # QLoRA precision rides on the checkpoint's own bnb_4bit_compute_dtype.
    seed=0,
    report_to=[],
    remove_unused_columns=False,
)

ds_rows = [{"prompt": r_["prompt"], "chosen": r_["chosen"], "rejected": r_["rejected"]} for r_ in rows]
trainer = DPOTrainer(
    model=model,  # PEFT-wrapped; ref_model=None → base weights as reference (adapters disabled)
    ref_model=None,
    args=cfg,
    train_dataset=ds_rows,
    processing_class=tokenizer,
)

margins_pre = compute_margins("pre")          # P-margin: pre-update baseline
history = trainer.train()                     # 15 optimizer steps (3 epochs × 5 rows, batch 1)
log_hist = list(getattr(trainer.state, "log_history", []))
losses = [e["loss"] for e in log_hist if "loss" in e]
gate(
    len(losses) >= 1 and all(l == l and abs(l) != float("inf") for l in losses),
    f"logged losses not all finite ({losses[:5]}...)",
    1,
)
print(f"[train] {len(log_hist)} log entries | loss first={losses[0]:.4f} last={losses[-1]:.4f}", flush=True)

# --- artifact: save adapter + tokenizer, verify reloadability (P-artifact) ---
AD = os.path.join(OUT_BASE, "bramble_dpo_v0-adapter")
os.makedirs(AD, exist_ok=True)
model.save_pretrained(AD)
tokenizer.save_pretrained(AD)
tensor_files = sorted(glob.glob(os.path.join(AD, "*.safetensors")) + glob.glob(os.path.join(AD, "*.bin")))
tbytes = sum(os.path.getsize(p) for p in tensor_files)
print(f"[artifact] {len(tensor_files)} tensor file(s), {tbytes/1e6:.1f} MB -> {AD}", flush=True)
gate(tbytes < 200 * 10**6, f"adapter tensors {tbytes/1e6:.0f} MB > ~200 MB — not LoRA-sized", 2)

from peft import PeftConfig as PCfg

ac = PCfg.from_pretrained(AD)
print(f"[artifact] adapter_config reload OK: r={getattr(ac, 'r', None)} task_type={getattr(ac, 'task_type', None)}", flush=True)

margins_post = compute_margins("post")        # P-margin: post-update (LoRA weights in place)


# --- coexistence probe (P-coexist): serving-lane identity unchanged at run end ---
def live_models(port):
    try:
        with urllib.request.urlopen(f"http://inference-host:{port}/v1/models", timeout=5) as resp:
            return [m.get("id") for m in json.load(resp)["data"]]
    except Exception as exc:
        return f"UNREACHABLE ({exc})"


print(f"[coexist] :8081 -> {live_models(8081)} | :8082 -> {live_models(8082)}", flush=True)

# --- VERDICT (self-contained for banking; all readbacks from this same run) ---
deltas_pre = [r_["delta_chosen_minus_rejected"] for r_ in margins_pre]
deltas_post = [r_["delta_chosen_minus_rejected"] for r_ in margins_post]
improved = sum(1 for a, b in zip(deltas_pre, deltas_post) if b > a)
print("=" * 64, flush=True)
print(f"POC VERDICT | wall {(time.time()-t_start)/60:.1f} min", flush=True)
print(f"  logged steps: {len(losses)} | loss first->last: {losses[0]:.4f} -> {losses[-1]:.4f}", flush=True)
print(
    f"  mean Δ pre={sum(deltas_pre)/5:.4f} post={sum(deltas_post)/5:.4f} "
    f"| rows improved: {improved}/5",
    flush=True,
)
print(f"  adapter: {AD} ({tbytes/1e6:.1f} MB) | hub cache: {HF_HUB_CACHE}", flush=True)
print("=" * 64, flush=True)
