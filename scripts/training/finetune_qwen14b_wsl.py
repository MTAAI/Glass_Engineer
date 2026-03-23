"""
Glass Expert AI — Qwen2.5-14B QLoRA Fine-tuning (Unsloth / WSL2)
================================================================
Fine-tunes Qwen2.5-14B-Instruct on upgraded glass science Q&A data.
Uses Unsloth for 2-5x speedup — runs in WSL2 Ubuntu where Triton
works natively (no Windows Application Control blocking).

Hardware requirements:
  - 24GB VRAM (RTX PRO 5000 Blackwell): 4-bit QLoRA

Usage (from WSL2):
    source ~/qwen14b-train/bin/activate
    cd ~/glass-training
    python train.py
    python train.py --resume
    python train.py --merge-only

Output:
    ~/glass-training/output/adapter/  — LoRA adapter
    ~/glass-training/output/merged/   — merged model for serving
"""

import os
import sys
import json
import logging
import argparse
import math
from pathlib import Path
from datetime import datetime

os.environ["PYTHONIOENCODING"] = "utf-8"

import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

# ── Model Config ──────────────────────────────────────────────────────────────
BASE_MODEL = "Qwen/Qwen2.5-14B-Instruct"

# ── Paths (native Linux filesystem for speed) ────────────────────────────────
BASE_DIR = Path(os.environ.get("TRAIN_BASE_DIR", os.path.expanduser("~/glass-training")))
OUTPUT_DIR = BASE_DIR / "output" / "adapter"
MERGED_DIR = BASE_DIR / "output" / "merged"
TRAIN_FILE = BASE_DIR / "data" / "train.jsonl"
VAL_FILE = BASE_DIR / "data" / "val.jsonl"
LOG_DIR = BASE_DIR / "logs"

# ── LoRA Config ───────────────────────────────────────────────────────────────
LORA_R = 64
LORA_ALPHA = 128
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ── Training Hyperparams (matched to original run.log) ───────────────────────
NUM_EPOCHS = 1
LEARNING_RATE = 5e-5
MAX_LENGTH = 2048
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.03
SAVE_STEPS = 1000
EVAL_STEPS = 1000
LOGGING_STEPS = 25
SAVE_TOTAL_LIMIT = 5
BATCH_SIZE = 2
GRAD_ACCUM = 4            # original run used 4 (effective batch = 8)

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


def make_formatting_func(tokenizer):
    """Create formatting function for SFTTrainer."""
    def formatting_func(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return [text] if isinstance(text, str) else text
    return formatting_func


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-14B for Glass Expert AI")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--merge-only", action="store_true", help="Just merge LoRA and save")
    parser.add_argument("--batch-size", type=int, default=0, help="Override batch size")
    parser.add_argument("--grad-accum", type=int, default=0, help="Override gradient accumulation steps")
    args = parser.parse_args()

    batch_size = args.batch_size or BATCH_SIZE
    grad_accum = args.grad_accum or GRAD_ACCUM

    logger.info("=" * 65)
    logger.info("Glass Expert AI - Qwen2.5-14B Fine-tuning (Unsloth / WSL2)")
    logger.info(f"  Base model: {BASE_MODEL}")
    logger.info(f"  Batch: {batch_size} x {grad_accum} = {batch_size * grad_accum} effective")
    logger.info(f"  Max length: {MAX_LENGTH}")
    logger.info(f"  Base dir: {BASE_DIR}")
    logger.info("=" * 65)

    # ── Use local pre-quantized bnb-4bit model (matches original run) ────────
    local_model = os.path.expanduser("~/glass-training/models/Qwen2.5-14B-Instruct-bnb-4bit")
    if not os.path.exists(local_model):
        # Fallback to HF cache
        local_model = os.path.expanduser(
            "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
        )
    if not os.path.exists(local_model):
        logger.info("Local cache not found, will download from HuggingFace...")
        local_model = BASE_MODEL

    # ── Merge-only mode ───────────────────────────────────────────────────────
    if args.merge_only:
        logger.info("Merge-only mode - loading base model + LoRA adapter")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(OUTPUT_DIR),
            max_seq_length=MAX_LENGTH,
            load_in_4bit=True,
        )
        model = FastLanguageModel.for_inference(model)
        MERGED_DIR.mkdir(parents=True, exist_ok=True)
        model.save_pretrained_merged(str(MERGED_DIR), tokenizer)
        logger.info(f"Merged model saved to: {MERGED_DIR}")
        return

    # ── Check training files exist ────────────────────────────────────────────
    for f in [TRAIN_FILE, VAL_FILE]:
        if not f.exists():
            logger.error(f"Training file not found: {f}")
            sys.exit(1)

    # ── Load model with Unsloth ──────────────────────────────────────────────
    logger.info(f"Loading model (Unsloth 4-bit) from {local_model}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=local_model,
        max_seq_length=MAX_LENGTH,
        load_in_4bit=True,
        dtype=torch.bfloat16,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Apply LoRA via Unsloth ───────────────────────────────────────────────
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model loaded - {total/1e9:.1f}B params total")
    logger.info(f"  LoRA applied - trainable: {trainable/1e6:.1f}M / {total/1e9:.1f}B ({100*trainable/total:.2f}%)")

    # ── Load datasets ─────────────────────────────────────────────────────────
    train_data = Dataset.from_list(load_jsonl(TRAIN_FILE))
    val_data = Dataset.from_list(load_jsonl(VAL_FILE))

    # ── Force max_steps to match original run (checkpoint compatibility) ──────
    # Original run: bnb-4bit tokenizer expanded 82,703 JSONL → 141,505 examples
    # 141,505 / 8 effective batch = 17,689 steps
    # We MUST set max_steps=17689 so checkpoint-14000 resume works correctly.
    ORIGINAL_MAX_STEPS = 17689

    effective_batch = batch_size * grad_accum
    steps_per_epoch = math.ceil(len(train_data) / effective_batch)
    logger.info(f"  Effective batch: {effective_batch} | JSONL steps/epoch: {steps_per_epoch:,}")
    logger.info(f"  Overriding max_steps to {ORIGINAL_MAX_STEPS} (matching original run)")

    # ── Training config ───────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        max_steps=ORIGINAL_MAX_STEPS,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=int(ORIGINAL_MAX_STEPS * WARMUP_RATIO),
        lr_scheduler_type="cosine",
        max_length=MAX_LENGTH,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        eval_strategy="steps",
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        fp16=False,
        optim="adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        save_on_each_node=True,
        seed=42,
        dataset_text_field="text",
        dataset_num_proc=4,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_data,
        eval_dataset=val_data,
        formatting_func=make_formatting_func(tokenizer),
        args=training_args,
    )

    # ── Auto-detect checkpoint for resume ─────────────────────────────────────
    resume_checkpoint = None
    if args.resume:
        checkpoints = sorted(OUTPUT_DIR.glob("checkpoint-*"), key=os.path.getmtime)
        if checkpoints:
            resume_checkpoint = str(checkpoints[-1])
            logger.info(f"  Resuming from: {resume_checkpoint}")
        else:
            logger.warning("  --resume specified but no checkpoints found. Starting fresh.")

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # ── Save ──────────────────────────────────────────────────────────────────
    logger.info("Saving LoRA adapter...")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    # Final eval
    eval_result = trainer.evaluate()
    logger.info(f"Final eval loss: {eval_result['eval_loss']:.4f}")

    # ── Copy adapter back to Windows for serving ─────────────────────────────
    win_adapter = "/mnt/c/Users/zainm/glass-ai-data/glass-expert-ai/models/qwen14b-glass-expert/adapter"
    logger.info(f"Copying adapter to Windows: {win_adapter}")
    os.makedirs(win_adapter, exist_ok=True)
    import shutil
    for f in OUTPUT_DIR.iterdir():
        shutil.copy2(f, win_adapter)

    logger.info("=" * 65)
    logger.info("TRAINING COMPLETE")
    logger.info(f"  LoRA adapter: {OUTPUT_DIR}")
    logger.info(f"  Copied to Windows: {win_adapter}")
    logger.info(f"  To merge:     python train.py --merge-only")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
