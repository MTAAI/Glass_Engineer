"""
Glass Expert AI — Qwen2.5-14B QLoRA Fine-tuning
================================================
Fine-tunes Qwen2.5-14B-Instruct on upgraded glass science Q&A data.

Hardware requirements:
  - 24GB VRAM (RTX 4090 / RTX PRO 5000): QLoRA 4-bit ✅
  - 54GB VRAM (shared GPU): QLoRA with higher rank or FP16 LoRA ✅

Usage:
    python scripts/training/finetune_qwen14b.py                      # 4-bit QLoRA (24GB GPU)
    python scripts/training/finetune_qwen14b.py --high-vram           # higher quality (54GB GPU)
    python scripts/training/finetune_qwen14b.py --resume              # resume from checkpoint
    python scripts/training/finetune_qwen14b.py --merge-only          # merge LoRA + push

Output:
    models/glass-expert-qwen14b/         — LoRA adapter
    models/glass-expert-qwen14b-merged/  — merged model ready for serving
"""

import sys
import json
import logging
import argparse
import math
from pathlib import Path
from datetime import datetime

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel,
)
from trl import SFTTrainer, SFTConfig

# ── Model Config ──────────────────────────────────────────────────────────────
BASE_MODEL = "Qwen/Qwen2.5-14B-Instruct"

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("models/qwen14b-glass-expert/adapter")
MERGED_DIR = Path("models/qwen14b-glass-expert/merged")
TRAIN_FILE = Path("models/qwen14b-glass-expert/data/qwen_train.jsonl")
VAL_FILE = Path("models/qwen14b-glass-expert/data/qwen_val.jsonl")
LOG_DIR = Path("models/qwen14b-glass-expert/logs")

# ── LoRA Config ───────────────────────────────────────────────────────────────
# Qwen2.5 attention + MLP modules
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ── Training Hyperparams ─────────────────────────────────────────────────────
NUM_EPOCHS = 2
LEARNING_RATE = 5e-5       # lower LR for 14B (more stable)
MAX_LENGTH = 4096
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.03
SAVE_STEPS = 300
EVAL_STEPS = 300
LOGGING_STEPS = 25
SAVE_TOTAL_LIMIT = 3

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ── System Prompt (matches inference) ─────────────────────────────────────────
SYSTEM_PROMPT = """You are Glass Expert AI, a highly specialized assistant trained on thousands of glass science research papers, textbooks, and material databases. You provide accurate, detailed, and technically precise answers about glass composition, properties, manufacturing processes, defects, characterization, and applications. Always cite relevant glass science principles in your answers."""

SYSTEM_PROMPT_FA = """شما Glass Expert AI هستید، یک دستیار تخصصی با تخصص سطح دکترا برای دانشمندان شیشه و مهندسان تولید. پاسخ‌های دقیق و فنی بر اساس اصول علم شیشه ارائه دهید."""


def load_jsonl(filepath: Path) -> list[dict]:
    """Load JSONL file into list of dicts."""
    records = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info(f"Loaded {len(records):,} examples from {filepath.name}")
    return records


def prepare_training_data(input_file: Path, output_train: Path, output_val: Path, val_ratio: float = 0.05):
    """
    Convert augmented_all_43k.jsonl → Qwen chat format JSONL.

    Each example becomes:
    {"messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]}
    """
    logger.info(f"Preparing training data from {input_file}")

    pairs = []
    with open(input_file, encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line.strip())
                q = d.get("question", "").strip()
                a = d.get("answer", "").strip()
                if q and a:
                    pairs.append((q, a))
            except:
                pass

    logger.info(f"  Loaded {len(pairs):,} Q&A pairs")

    # Detect language and format as chat
    import random
    random.seed(42)
    random.shuffle(pairs)

    records = []
    for q, a in pairs:
        # Simple Farsi detection
        is_farsi = any('\u0600' <= c <= '\u06FF' for c in q[:50])
        sys_prompt = SYSTEM_PROMPT_FA if is_farsi else SYSTEM_PROMPT

        records.append({
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ]
        })

    # Split train/val
    val_size = max(int(len(records) * val_ratio), 100)
    val_records = records[:val_size]
    train_records = records[val_size:]

    # Save
    output_train.parent.mkdir(parents=True, exist_ok=True)
    for path, data in [(output_train, train_records), (output_val, val_records)]:
        with open(path, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info(f"  Saved {len(data):,} examples → {path}")

    return len(train_records), len(val_records)


def make_formatting_func(tokenizer):
    """Create formatting function for SFTTrainer."""
    def formatting_func(example):
        return tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
    return formatting_func


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-14B for Glass Expert AI")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--merge-only", action="store_true", help="Just merge LoRA and save")
    parser.add_argument("--high-vram", action="store_true", help="Use higher LoRA rank (54GB+ GPU)")
    parser.add_argument("--prepare-data", type=str, default="",
                        help="Path to augmented JSONL to prepare training data from")
    parser.add_argument("--batch-size", type=int, default=0, help="Override batch size")
    parser.add_argument("--grad-accum", type=int, default=0, help="Override gradient accumulation steps")
    args = parser.parse_args()

    logger.info("=" * 65)
    logger.info("Glass Expert AI — Qwen2.5-14B QLoRA Fine-tuning")
    logger.info(f"  Base model: {BASE_MODEL}")
    logger.info(f"  High VRAM mode: {args.high_vram}")
    logger.info("=" * 65)

    # ── Step 0: Prepare data if requested ─────────────────────────────────────
    if args.prepare_data:
        src = Path(args.prepare_data)
        if not src.exists():
            logger.error(f"Source file not found: {src}")
            sys.exit(1)
        prepare_training_data(src, TRAIN_FILE, VAL_FILE)
        logger.info("Data preparation complete. Run again without --prepare-data to train.")
        return

    # ── Step 0.5: Merge-only mode ────────────────────────────────────────────
    if args.merge_only:
        logger.info("Merge-only mode — loading base model + LoRA adapter")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, str(OUTPUT_DIR))
        model = model.merge_and_unload()

        MERGED_DIR.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(MERGED_DIR))
        tokenizer.save_pretrained(str(MERGED_DIR))
        logger.info(f"Merged model saved to: {MERGED_DIR}")
        return

    # ── Check training files exist ────────────────────────────────────────────
    for f in [TRAIN_FILE, VAL_FILE]:
        if not f.exists():
            logger.error(f"Training file not found: {f}")
            logger.error(f"Run with --prepare-data <augmented_file.jsonl> first")
            sys.exit(1)

    # ── LoRA configuration based on VRAM ──────────────────────────────────────
    if args.high_vram:
        # 54GB+ GPU: higher rank = better quality
        lora_r = 128
        lora_alpha = 256
        lora_dropout = 0.05
        batch_size = args.batch_size or 4
        grad_accum = args.grad_accum or 4
        # Can use 8-bit for better quality
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        logger.info(f"  High VRAM mode: LoRA r={lora_r}, 8-bit quantization, batch={batch_size}")
    else:
        # 24GB GPU: conservative settings
        lora_r = 64
        lora_alpha = 128
        lora_dropout = 0.05
        batch_size = args.batch_size or 1
        grad_accum = args.grad_accum or 16
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        logger.info(f"  Standard mode: LoRA r={lora_r}, 4-bit NF4, batch={batch_size}")

    # ── Load tokenizer ────────────────────────────────────────────────────────
    logger.info(f"Loading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Load model ────────────────────────────────────────────────────────────
    logger.info(f"Loading model with quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
    )
    model = prepare_model_for_kbit_training(model)

    # Print model size info
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model loaded — {total/1e9:.1f}B params total")

    # ── Apply LoRA ────────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"  LoRA applied — trainable: {trainable/1e6:.1f}M / {total/1e9:.1f}B ({100*trainable/total:.2f}%)")

    # ── Load datasets ─────────────────────────────────────────────────────────
    train_data = Dataset.from_list(load_jsonl(TRAIN_FILE))
    val_data = Dataset.from_list(load_jsonl(VAL_FILE))

    # ── Effective batch size ──────────────────────────────────────────────────
    effective_batch = batch_size * grad_accum
    steps_per_epoch = math.ceil(len(train_data) / effective_batch)
    total_steps = steps_per_epoch * NUM_EPOCHS
    logger.info(f"  Effective batch: {effective_batch} | Steps/epoch: {steps_per_epoch:,} | Total: {total_steps:,}")

    # ── Training config ───────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        max_seq_length=MAX_LENGTH,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        eval_strategy="steps",
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        report_to="none",
        seed=42,
        dataset_text_field="text",
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        eval_dataset=val_data,
        formatting_func=make_formatting_func(tokenizer),
        args=training_args,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info("Starting training...")
    if args.resume:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────────
    logger.info("Saving LoRA adapter...")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    # Final eval
    eval_result = trainer.evaluate()
    logger.info(f"Final eval loss: {eval_result['eval_loss']:.4f}")

    logger.info("=" * 65)
    logger.info("TRAINING COMPLETE")
    logger.info(f"  LoRA adapter: {OUTPUT_DIR}")
    logger.info(f"  To merge:     python {__file__} --merge-only")
    logger.info(f"  To serve:     python serve.py --model {MERGED_DIR}")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
