#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)


@dataclass
class RunConfig:
    base_model_id: str
    train_jsonl: str
    max_length: int
    max_steps: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    seed: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: List[str]
    eval_prompt: str
    eval_max_new_tokens: int


class TokenizedJsonlDataset(torch.utils.data.Dataset):
    def __init__(self, items: List[Dict[str, torch.Tensor]]):
        self._items = items

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._items[idx]


def load_train_pairs(path: Path) -> List[Dict[str, str]]:
    pairs: List[Dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        pairs.append({"prompt": obj["prompt"], "response": obj["response"]})
    return pairs


def build_training_example(tokenizer, prompt: str, response: str, max_length: int) -> Dict[str, torch.Tensor]:
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = enc["input_ids"][0]
    attention_mask = enc["attention_mask"][0]
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def generate_once(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def main() -> int:
    repo_root = Path(os.environ.get("REPO_ROOT", ".")).resolve()
    model_root = Path(os.environ.get("MODEL_ROOT", "/mnt/hf/models")).resolve()

    evidence_dir_raw = os.environ.get("EVIDENCE_DIR", "").strip()
    if evidence_dir_raw:
        evidence_dir = Path(evidence_dir_raw).expanduser()
        if not evidence_dir.is_absolute():
            evidence_dir = (repo_root / evidence_dir).resolve()
        else:
            evidence_dir = evidence_dir.resolve()
    else:
        stamp = time.strftime("%Y-%m-%d", time.gmtime())
        evidence_dir = (repo_root / f"reports/retest_{stamp}_finetune").resolve()
    evidence_dir.mkdir(parents=True, exist_ok=True)

    adapter_out_dir = model_root / "smollm2-135m-instruct-lora-demo"
    adapter_out_dir.mkdir(parents=True, exist_ok=True)

    cfg = RunConfig(
        base_model_id=os.environ.get("BASE_MODEL_ID", "HuggingFaceTB/SmolLM2-135M-Instruct"),
        train_jsonl=str(repo_root / "llm-finetune/data/train.jsonl"),
        max_length=int(os.environ.get("MAX_LENGTH", "256")),
        max_steps=int(os.environ.get("MAX_STEPS", "200")),
        per_device_train_batch_size=int(os.environ.get("BATCH_SIZE", "2")),
        gradient_accumulation_steps=int(os.environ.get("GRAD_ACCUM", "4")),
        learning_rate=float(os.environ.get("LEARNING_RATE", "2e-4")),
        seed=int(os.environ.get("SEED", "42")),
        lora_r=int(os.environ.get("LORA_R", "8")),
        lora_alpha=int(os.environ.get("LORA_ALPHA", "16")),
        lora_dropout=float(os.environ.get("LORA_DROPOUT", "0.05")),
        lora_target_modules=os.environ.get("LORA_TARGETS", "q_proj,k_proj,v_proj,o_proj").split(","),
        eval_prompt=os.environ.get("EVAL_PROMPT", "Say hello to me."),
        eval_max_new_tokens=int(os.environ.get("EVAL_MAX_NEW_TOKENS", "64")),
    )

    print("[finetune] repo_root=", repo_root)
    print("[finetune] model_root=", model_root)
    print("[finetune] adapter_out_dir=", adapter_out_dir)
    print("[finetune] evidence_dir=", evidence_dir)
    print("[finetune] config=", json.dumps(asdict(cfg), indent=2))

    torch.manual_seed(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Train LoRA on CPU for portability.
    base_model = AutoModelForCausalLM.from_pretrained(cfg.base_model_id, torch_dtype=torch.float32)
    base_model.config.use_cache = False

    lora_cfg = LoraConfig(
        task_type="CAUSAL_LM",
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    pairs = load_train_pairs(Path(cfg.train_jsonl))
    items = [build_training_example(tokenizer, p["prompt"], p["response"], cfg.max_length) for p in pairs]
    train_ds = TokenizedJsonlDataset(items)

    train_args = TrainingArguments(
        output_dir=str(evidence_dir / "trainer_out"),
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        max_steps=cfg.max_steps,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        seed=cfg.seed,
        dataloader_num_workers=0,
        fp16=False,
        bf16=False,
        optim="adamw_torch",
    )

    t0 = time.time()
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )
    trainer.train()
    train_seconds = time.time() - t0

    model.save_pretrained(adapter_out_dir)
    tokenizer.save_pretrained(adapter_out_dir)

    # Base vs tuned comparison.
    base_for_eval = AutoModelForCausalLM.from_pretrained(cfg.base_model_id, torch_dtype=torch.float32)
    tuned_for_eval = AutoModelForCausalLM.from_pretrained(cfg.base_model_id, torch_dtype=torch.float32)
    tuned_for_eval = PeftModel.from_pretrained(tuned_for_eval, adapter_out_dir)

    base_text = generate_once(base_for_eval, tokenizer, cfg.eval_prompt, cfg.eval_max_new_tokens)
    tuned_text = generate_once(tuned_for_eval, tokenizer, cfg.eval_prompt, cfg.eval_max_new_tokens)

    summary = {
        "config": asdict(cfg),
        "train_seconds": train_seconds,
        "adapter_out_dir": str(adapter_out_dir),
        "base_output": base_text,
        "tuned_output": tuned_text,
        "signature_expected": "-- Strix Halo",
        "base_has_signature": "-- Strix Halo" in base_text,
        "tuned_has_signature": "-- Strix Halo" in tuned_text,
    }

    out_path = evidence_dir / "finetune_comparison.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[finetune] wrote {out_path}")
    print("[finetune] train_seconds=", round(train_seconds, 2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
