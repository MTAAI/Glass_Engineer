"""
Glass Expert AI — QLoRA Fine-tuning Script
============================================
Model  : Meta Llama 3.1 8B Instruct
Method : QLoRA (4-bit quantization + LoRA adapters)
GPU    : RTX PRO 5000 Blackwell (25.7 GB VRAM)
Data   : data/processed/llama_train.jsonl + llama_val.jsonl

Estimated training time : 5–7 hours on RTX PRO 5000
Estimated VRAM usage    : ~18–22 GB
Output                  : models/glass-expert-v1/  (LoRA adapter)

FIXES applied vs previous version:
  - Removed `group_by_length` (removed from TrainingArguments in newer transformers)
  - Changed `torch_dtype` → `dtype` (torch_dtype is deprecated)

Usage:
    python scripts\training\finetune_glass_llm.py
    python scripts\training\finetune_glass_llm.py --resume
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    EarlyStoppingCallback,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from trl import SFTTrainer

# ── Configuration ──────────────────────────────────────────────────────────────

BASE_MODEL   = "meta-llama/Meta-Llama-3.1-8B-Instruct"
OUTPUT_DIR   = Path("models/glass-expert-v1")
TRAIN_FILE   = Path("data/processed/llama_train.jsonl")
VAL_FILE     = Path("data/processed/llama_val.jsonl")
LOG_DIR      = Path("logs/finetune")

# LoRA hyperparameters
LORA_R           = 64       # rank — higher = more capacity, more VRAM
LORA_ALPHA       = 128      # scaling factor (usually 2x rank)
LORA_DROPOUT     = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Training hyperparameters
NUM_EPOCHS          = 3
BATCH_SIZE          = 2       # per GPU — increase if VRAM allows
GRAD_ACCUM_STEPS    = 8       # effective batch = 2 * 8 = 16
LEARNING_RATE       = 2e-4
MAX_SEQ_LENGTH      = 2048    # max tokens per example
WARMUP_RATIO        = 0.03
LR_SCHEDULER        = "cosine"
WEIGHT_DECAY        = 0.001
SAVE_STEPS          = 200
EVAL_STEPS          = 200
LOGGING_STEPS       = 50
SAVE_TOTAL_LIMIT    = 3       # keep only last 3 checkpoints

# Quantization
USE_4BIT            = True
BNB_4BIT_COMPUTE    = "float16"
BNB_4BIT_QUANT_TYPE = "nf4"
USE_NESTED_QUANT    = True

# ── Logging ────────────────────────────────────────────────────────────────────

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


# ── Data loading ───────────────────────────────────────────────────────────────

def load_jsonl_dataset(filepath: Path) -> Dataset:
    """Load a JSONL file where each line has {"messages": [...]}"""
    records = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info(f"Loaded {len(records):,} examples from {filepath.name}")
    return Dataset.from_list(records)


def format_chat_prompt(example: dict, tokenizer) -> dict:
    """
    Apply the Llama 3 chat template to convert messages list into a
    single formatted string that the model is trained on.
    """
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


# ── Main ───────────────────────────────────────────────────────────────────────

def main(resume: bool = False):
    logger.info("=" * 65)
    logger.info("Glass Expert AI — QLoRA Fine-tuning")
    logger.info(f"Base model : {BASE_MODEL}")
    logger.info(f"Output dir : {OUTPUT_DIR}")
    logger.info(f"Train file : {TRAIN_FILE}")
    logger.info(f"Val file   : {VAL_FILE}")
    logger.info("=" * 65)

    # ── Validate files ────────────────────────────────────────────────────────
    for f in [TRAIN_FILE, VAL_FILE]:
        if not f.exists():
            logger.error(f"File not found: {f}")
            logger.error("Run prepare_training_data.py first!")
            sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── GPU check ─────────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        logger.error("CUDA not available! Fine-tuning requires a GPU.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"GPU: {gpu_name}  ({gpu_vram:.1f} GB VRAM)")

    # ── Load tokenizer ────────────────────────────────────────────────────────
    logger.info("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    )
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"   # required for training
    logger.info("Tokenizer loaded.")

    # ── Load datasets ─────────────────────────────────────────────────────────
    logger.info("\nLoading datasets...")
    train_dataset = load_jsonl_dataset(TRAIN_FILE)
    val_dataset   = load_jsonl_dataset(VAL_FILE)

    # Apply chat template
    train_dataset = train_dataset.map(
        lambda x: format_chat_prompt(x, tokenizer),
        remove_columns=train_dataset.column_names,
    )
    val_dataset = val_dataset.map(
        lambda x: format_chat_prompt(x, tokenizer),
        remove_columns=val_dataset.column_names,
    )
    logger.info(f"Train: {len(train_dataset):,} examples")
    logger.info(f"Val:   {len(val_dataset):,} examples")

    # ── Quantization config ───────────────────────────────────────────────────
    logger.info("\nConfiguring 4-bit quantization (QLoRA)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit               = USE_4BIT,
        bnb_4bit_compute_dtype     = getattr(torch, BNB_4BIT_COMPUTE),
        bnb_4bit_quant_type        = BNB_4BIT_QUANT_TYPE,
        bnb_4bit_use_double_quant  = USE_NESTED_QUANT,
    )

    # ── Load base model ───────────────────────────────────────────────────────
    logger.info(f"\nLoading base model: {BASE_MODEL}")
    logger.info("(Model already cached — loading from disk, no download needed)")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config = bnb_config,
        device_map          = "auto",
        trust_remote_code   = True,
        dtype               = torch.float16,   # fixed: was torch_dtype (deprecated)
        attn_implementation = "eager",         # flash_attention_2 not required
    )
    model.config.use_cache = False   # required for gradient checkpointing
    model.config.pretraining_tp = 1

    # Prepare model for QLoRA training
    model = prepare_model_for_kbit_training(model)

    # ── LoRA config ───────────────────────────────────────────────────────────
    logger.info("\nApplying LoRA adapters...")
    lora_config = LoraConfig(
        r                  = LORA_R,
        lora_alpha         = LORA_ALPHA,
        lora_dropout       = LORA_DROPOUT,
        bias               = "none",
        task_type          = TaskType.CAUSAL_LM,
        target_modules     = LORA_TARGET_MODULES,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Training arguments ────────────────────────────────────────────────────
    logger.info("\nConfiguring training arguments...")

    # Find last checkpoint if resuming
    resume_from = None
    if resume:
        checkpoints = sorted(OUTPUT_DIR.glob("checkpoint-*"))
        if checkpoints:
            resume_from = str(checkpoints[-1])
            logger.info(f"Resuming from: {resume_from}")
        else:
            logger.warning("No checkpoint found, starting from scratch.")

    training_args = TrainingArguments(
        output_dir                  = str(OUTPUT_DIR),
        num_train_epochs            = NUM_EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM_STEPS,
        learning_rate               = LEARNING_RATE,
        weight_decay                = WEIGHT_DECAY,
        lr_scheduler_type           = LR_SCHEDULER,
        warmup_ratio                = WARMUP_RATIO,
        fp16                        = True,
        bf16                        = False,
        max_grad_norm               = 0.3,
        gradient_checkpointing      = True,       # saves VRAM
        # group_by_length removed — no longer valid in newer transformers
        optim                       = "paged_adamw_32bit",
        save_strategy               = "steps",
        save_steps                  = SAVE_STEPS,
        save_total_limit            = SAVE_TOTAL_LIMIT,
        eval_strategy               = "steps",
        eval_steps                  = EVAL_STEPS,
        logging_dir                 = str(LOG_DIR),
        logging_steps               = LOGGING_STEPS,
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
        report_to                   = "none",
        dataloader_num_workers      = 0,          # 0 avoids Windows multiprocessing issues
        remove_unused_columns       = False,
        resume_from_checkpoint      = resume_from,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    logger.info("\nInitialising SFTTrainer...")
    trainer = SFTTrainer(
        model              = model,
        tokenizer          = tokenizer,
        train_dataset      = train_dataset,
        eval_dataset       = val_dataset,
        dataset_text_field = "text",
        max_seq_length     = MAX_SEQ_LENGTH,
        packing            = False,
        args               = training_args,
        callbacks          = [EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 65)
    logger.info("STARTING TRAINING")
    logger.info(f"  Epochs            : {NUM_EPOCHS}")
    logger.info(f"  Effective batch   : {BATCH_SIZE * GRAD_ACCUM_STEPS}")
    logger.info(f"  Learning rate     : {LEARNING_RATE}")
    logger.info(f"  Max seq length    : {MAX_SEQ_LENGTH}")
    logger.info(f"  Training examples : {len(train_dataset):,}")
    logger.info(f"  Steps per epoch   : {len(train_dataset) // (BATCH_SIZE * GRAD_ACCUM_STEPS):,}")
    logger.info("=" * 65)

    start_time = datetime.now()
    trainer.train(resume_from_checkpoint=resume_from)
    elapsed = datetime.now() - start_time

    logger.info(f"\nTraining complete! Time: {elapsed}")

    # ── Save final model ──────────────────────────────────────────────────────
    logger.info("\nSaving final LoRA adapter...")
    final_model_path = OUTPUT_DIR / "final"
    trainer.model.save_pretrained(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))
    logger.info(f"Model saved to: {final_model_path}")

    # ── Save training summary ─────────────────────────────────────────────────
    summary = {
        "base_model":       BASE_MODEL,
        "output_dir":       str(final_model_path),
        "train_examples":   len(train_dataset),
        "val_examples":     len(val_dataset),
        "epochs":           NUM_EPOCHS,
        "lora_r":           LORA_R,
        "lora_alpha":       LORA_ALPHA,
        "learning_rate":    LEARNING_RATE,
        "batch_size":       BATCH_SIZE * GRAD_ACCUM_STEPS,
        "max_seq_length":   MAX_SEQ_LENGTH,
        "training_time":    str(elapsed),
        "gpu":              gpu_name,
        "completed_at":     datetime.now().isoformat(),
    }
    summary_path = OUTPUT_DIR / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("\n" + "=" * 65)
    logger.info("FINE-TUNING COMPLETE!")
    logger.info(f"  LoRA adapter saved : {final_model_path}")
    logger.info(f"  Training summary   : {summary_path}")
    logger.info(f"  Total time         : {elapsed}")
    logger.info("=" * 65)
    logger.info("\nNext steps:")
    logger.info("  1. Run evaluate_model.py to test on the golden eval set")
    logger.info("  2. Deploy with vLLM: vllm serve models/glass-expert-v1/final")
    logger.info("  3. Set LLM_BASE_URL=http://localhost:8000/v1 in .env" )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Glass Expert AI Fine-tuning")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last checkpoint",
    )
    args = parser.parse_args()
    main(resume=args.resume)
