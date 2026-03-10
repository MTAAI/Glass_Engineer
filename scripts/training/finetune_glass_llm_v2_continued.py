"""
Glass Expert AI — Continued Fine-tuning on v2
================================================
Compatible with TRL >= 0.15 (tested on 0.29.0)
Loads glass-expert-v2 LoRA adapter and continues
training on the detailed patch dataset.

Usage:
    python scripts/training/finetune_glass_llm_v2_continued.py
"""

import json
import logging
from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
BASE_MODEL       = "meta-llama/Meta-Llama-3.1-8B-Instruct"
V2_ADAPTER_PATH  = Path("models/glass-expert-v2/final")
PATCH_DATASET    = Path("data/qa_pairs/generated/glass_expert_detailed_patch.jsonl")
OUTPUT_DIR       = Path("models/glass-expert-v2-patched")

NUM_EPOCHS       = 2
LEARNING_RATE    = 5e-5
MAX_SEQ_LENGTH   = 1024
BATCH_SIZE       = 2
GRAD_ACCUM       = 8
WARMUP_STEPS     = 20
SAVE_STEPS       = 100
EVAL_STEPS       = 100
VAL_SPLIT        = 0.05
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def load_dataset(path: Path):
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    pairs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    log.info(f"Loaded {len(pairs)} patch pairs from {path}")
    split = max(1, int(len(pairs) * (1 - VAL_SPLIT)))
    log.info(f"Train: {split}, Val: {len(pairs) - split}")
    return pairs[:split], pairs[split:]


def main():
    log.info("=" * 65)
    log.info("Glass Expert AI — Continued Fine-tuning v2 → v2-patched")
    log.info(f"Base model    : {BASE_MODEL}")
    log.info(f"v2 adapter    : {V2_ADAPTER_PATH}")
    log.info(f"Patch dataset : {PATCH_DATASET}")
    log.info(f"Output        : {OUTPUT_DIR}")
    log.info("=" * 65)

    if not torch.cuda.is_available():
        log.error("CUDA not available! GPU required.")
        raise SystemExit(1)

    if not V2_ADAPTER_PATH.exists():
        log.error(f"v2 adapter not found at {V2_ADAPTER_PATH}")
        raise SystemExit(1)

    if not PATCH_DATASET.exists():
        log.error(f"Patch dataset not found at {PATCH_DATASET}")
        raise SystemExit(1)

    # ── Tokenizer ───────────────────────────────────────────────────
    log.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Base model in 4-bit ─────────────────────────────────────────
    log.info("Loading base model in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    # ── Load v2 adapter ─────────────────────────────────────────────
    log.info(f"Loading v2 LoRA adapter from {V2_ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(model, str(V2_ADAPTER_PATH), is_trainable=True)
    log.info("v2 adapter loaded — continuing training from v2 weights")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    log.info(f"Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    # ── Dataset ─────────────────────────────────────────────────────
    train_pairs, val_pairs = load_dataset(PATCH_DATASET)

    def format_text(example):
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )

    train_texts = [format_text(p) for p in train_pairs]
    val_texts   = [format_text(p) for p in val_pairs]

    train_dataset = Dataset.from_dict({"text": train_texts})
    val_dataset   = Dataset.from_dict({"text": val_texts})

    # ── SFTConfig (replaces TrainingArguments for SFTTrainer in TRL>=0.15) ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sft_config = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=WARMUP_STEPS,
        fp16=False,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        dataloader_num_workers=0,
        optim="paged_adamw_8bit",
        max_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        packing=False,
    )

    # ── Trainer ─────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    # ── Train ────────────────────────────────────────────────────────
    log.info("Starting continued fine-tuning...")
    trainer.train()

    # ── Save ─────────────────────────────────────────────────────────
    final_path = OUTPUT_DIR / "final"
    log.info(f"Saving final adapter to {final_path}...")
    trainer.model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    log.info("=" * 65)
    log.info("GLASS EXPERT v2-PATCHED FINE-TUNING COMPLETE!")
    log.info(f"Model saved to: {final_path}")
    log.info("Next steps:")
    log.info("  1. Update serve.py: MODEL_PATH = 'models/glass-expert-v2-patched/final'")
    log.info("  2. Update serve.py: max_new_tokens = 768")
    log.info("  3. Restart the model server")
    log.info("  4. Run the golden set evaluation to confirm improvement")
    log.info("=" * 65)


if __name__ == "__main__":
    main()
