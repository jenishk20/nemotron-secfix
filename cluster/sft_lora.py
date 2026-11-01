"""LoRA SFT of Nemotron 3.5 Lightning 30B-A3B on the security training set (one H200).

Trains with thinking OFF: the prompt is the chat template with enable_thinking=False
(ends in "<think></think>"), loss only on the answer. Eval must use thinking off too.
LoRA on attention q/k/v/o, Mamba in_proj, and the MoE shared experts. Not on Mamba
out_proj (the fused Mamba kernel reads its weight directly, so LoRA there would not train)
and not on the 128 routed experts. Needs transformers >= 5 (native nemotron_h), which loads
the checkpoint's "backbone.*" weights as "model.*".

--merge adds the LoRA deltas straight into a copy of the ORIGINAL safetensors shards (same
names, same layout), so vLLM loads the tuned model exactly like the base.

Usage: python sft_lora.py --model <path or HF id> --train secpy_train.jsonl --val secpy_val.jsonl --out runs/lora_v1
       python sft_lora.py --merge --model <base> --out runs/lora_v1   (writes runs/lora_v1/merged)
"""
import argparse, glob, json, os, shutil

import torch
from safetensors.torch import load_file, save_file
from datasets import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer,
                          TrainingArguments)
from peft import LoraConfig, get_peft_model

TARGETS = r".*layers\.\d+\.mixer\.(q_proj|k_proj|v_proj|o_proj|in_proj|shared_experts\.up_proj|shared_experts\.down_proj)"


def encode(tok, row, max_len):
    user, asst = row["messages"]
    prompt = tok.apply_chat_template([user], add_generation_prompt=True, enable_thinking=False, tokenize=False)
    full = tok.apply_chat_template([user, asst], enable_thinking=False, tokenize=False)
    assert full.startswith(prompt), "chat template: prompt is not a prefix of the full conversation"
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(full[len(prompt):], add_special_tokens=False)["input_ids"]
    ids = p_ids + a_ids
    if len(ids) > max_len:
        return None
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": [-100] * len(p_ids) + a_ids}


def load(tok, path, max_len):
    rows = [encode(tok, json.loads(l), max_len) for l in open(path)]
    kept = [r for r in rows if r]
    print(f"{path}: {len(kept)}/{len(rows)} examples (dropped {len(rows) - len(kept)} longer than {max_len} tokens)")
    return Dataset.from_list(kept)


def merge(a):
    adir, out = os.path.join(a.out, "adapter"), os.path.join(a.out, "merged")
    cfg = json.load(open(os.path.join(adir, "adapter_config.json")))
    scale = cfg["lora_alpha"] / cfg["r"]
    lora = load_file(os.path.join(adir, "adapter_model.safetensors"))
    # "base_model.model.model.layers.5.mixer.q_proj.lora_A.weight" -> "layers.5.mixer.q_proj"
    mods = {}
    for k, v in lora.items():
        name, part = k.rsplit(".lora_", 1)
        name = name.removeprefix("base_model.model.").removeprefix("model.").removeprefix("backbone.")
        mods.setdefault(name, {})[part.split(".")[0]] = v
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(a.model):  # config, tokenizer, chat template, index...
        if not f.endswith(".safetensors") and os.path.isfile(os.path.join(a.model, f)):
            shutil.copy(os.path.join(a.model, f), out)
    done = set()
    for shard in sorted(glob.glob(os.path.join(a.model, "*.safetensors"))):
        w = load_file(shard)
        for key in w:
            name = key.removesuffix(".weight").removeprefix("model.").removeprefix("backbone.")
            if key.endswith(".weight") and name in mods:
                A, B = mods[name]["A"].cuda().float(), mods[name]["B"].cuda().float()
                w[key] = (w[key].cuda().float() + scale * (B @ A)).to(w[key].dtype).cpu()
                done.add(name)
        save_file(w, os.path.join(out, os.path.basename(shard)), metadata={"format": "pt"})
        print("merged", os.path.basename(shard), flush=True)
    missing = set(mods) - done
    assert not missing, f"LoRA modules with no base weight: {sorted(missing)[:5]}"
    print(f"merged {len(done)} LoRA modules (scale {scale}) into {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train"); ap.add_argument("--val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=6144)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--report_to", default="none")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.merge:
        return merge(a)

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    train, val = load(tok, a.train, a.max_len), load(tok, a.val, a.max_len)

    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda")
    model.config.use_cache = False
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05,
                                             target_modules=TARGETS, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    n_lora = sum(1 for n, _ in model.named_modules() if n.endswith(".lora_A"))
    print("LoRA modules:", n_lora)
    assert n_lora > 0, "LoRA matched no layers; check TARGETS against model.named_modules()"

    args = TrainingArguments(
        output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr, lr_scheduler_type="cosine",
        warmup_steps=0.03, per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=a.grad_accum, bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, logging_steps=5,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2, report_to=a.report_to,
        remove_unused_columns=False, dataloader_num_workers=2,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=val,
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))
    trainer.train()
    model.save_pretrained(os.path.join(a.out, "adapter"))
    print("adapter written to", os.path.join(a.out, "adapter"))


if __name__ == "__main__":
    main()
