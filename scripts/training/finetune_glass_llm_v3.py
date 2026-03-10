"""
Glass Expert AI — QLoRA Fine-tuning v3
=======================================
Changes from v2:
  - Points to glass_expert_v3_train.jsonl (10,119 examples, Kelvin fixed, patch Q&A added)
  - Auto-splits train/val (95/5) — no separate val file needed
  - NUM_EPOCHS reduced to 2 (correction task, not full learning)
  - LEARNING_RATE reduced to 1e-4 (preserve existing knowledge)
  - MAX_LENGTH reduced to 1024 (patch answers are longer, saves VRAM)
  - LORA_R=16, LORA_ALPHA=32 (leaner adapters, faster training)
  - Output saved to models/glass-expert-v3/

Usage:
    python scripts\\training\\finetune_glass_llm_v3.py
    python scripts\\training\\finetune_glass_llm_v3.py --resume
"""
import sys, json, logging, argparse, math, random
from pathlib import Path
from datetime import datetime
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer, SFTConfig

# ── Configuration ──────────────────────────────────────────────────────────────
BASE_MODEL  = "meta-llama/Meta-Llama-3.1-8B-Instruct"
OUTPUT_DIR  = Path("models/glass-expert-v3")
TRAIN_FILE  = Path("data/qa_pairs/generated/glass_expert_v3_train.jsonl")
LOG_DIR     = Path("logs/finetune")

# LoRA — leaner than v2 (faster, less risk of overfitting on patch data)
LORA_R      = 16
LORA_ALPHA  = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Training — 2 epochs, lower LR, longer sequences
NUM_EPOCHS       = 2
BATCH_SIZE       = 2
GRAD_ACCUM_STEPS = 8        # effective batch = 16
LEARNING_RATE    = 1e-4
MAX_LENGTH       = 1024     # patch answers avg 705 chars, need more than 512
WEIGHT_DECAY     = 0.001
SAVE_STEPS       = 200
EVAL_STEPS       = 200
LOGGING_STEPS    = 50
SAVE_TOTAL_LIMIT = 3
VAL_SPLIT        = 0.05     # 5% validation split (~506 examples)

# ── Logging setup ──────────────────────────────────────────────────────────────
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / f"train_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def load_and_split_dataset(filepath, val_split=0.05, seed=42):
    """Load JSONL and split into train/val."""
    records = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info(f"Loaded {len(records):,} total examples from {filepath.name}")

    random.seed(seed)
    random.shuffle(records)
    split_idx = int(len(records) * (1 - val_split))
    train_records = records[:split_idx]
    val_records   = records[split_idx:]
    logger.info(f"Split → Train: {len(train_records):,}  Val: {len(val_records):,}")
    return Dataset.from_list(train_records), Dataset.from_list(val_records)


def make_formatting_func(tokenizer):
    def formatting_func(example):
        return tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
    return formatting_func


def main(resume=False):
    logger.info("=" * 65)
    logger.info("Glass Expert AI — QLoRA Fine-tuning v3")
    logger.info(f"Base model : {BASE_MODEL}")
    logger.info(f"Dataset    : {TRAIN_FILE}")
    logger.info(f"Output     : {OUTPUT_DIR}")
    logger.info("=" * 65)

    if not TRAIN_FILE.exists():
        logger.error(f"Training file not found: {TRAIN_FILE}")
        logger.error("Please save glass_expert_v3_train.jsonl to:")
        logger.error(f"  {TRAIN_FILE.resolve()}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        logger.error("CUDA not available! GPU required for training.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"GPU: {gpu_name}  ({gpu_vram:.1f} GB VRAM)")

    # ── Load and split dataset ─────────────────────────────────────────────────
    logger.info("\nLoading and splitting dataset...")
    train_dataset, val_dataset = load_and_split_dataset(TRAIN_FILE, VAL_SPLIT)

    # ── Tokenizer ──────────────────────────────────────────────────────────────
    logger.info("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    logger.info("Tokenizer loaded.")

    # ── Model (4-bit quantized) ────────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    logger.info("\nLoading base model (from cache)...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model = prepare_model_for_kbit_training(model)

    # ── LoRA adapters ──────────────────────────────────────────────────────────
    logger.info("\nApplying LoRA adapters (r=16, alpha=32)...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=LORA_TARGET_MODULES,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Training config ────────────────────────────────────────────────────────
    resume_from = None
    if resume:
        checkpoints = sorted(OUTPUT_DIR.glob("checkpoint-*"))
        if checkpoints:
            resume_from = str(checkpoints[-1])
            logger.info(f"Resuming from: {resume_from}")
        else:
            logger.warning("No checkpoint found, starting fresh.")

    steps_per_epoch = math.ceil(len(train_dataset) / (BATCH_SIZE * GRAD_ACCUM_STEPS))
    total_steps     = steps_per_epoch * NUM_EPOCHS
    warmup_steps    = max(1, int(total_steps * 0.03))

    logger.info(f"\nTraining plan:")
    logger.info(f"  Epochs: {NUM_EPOCHS}  |  Steps/epoch: {steps_per_epoch:,}  |  Total: {total_steps:,}")
    logger.info(f"  Effective batch: {BATCH_SIZE * GRAD_ACCUM_STEPS}  |  Warmup: {warmup_steps}")
    logger.info(f"  Learning rate: {LEARNING_RATE}  |  Max seq length: {MAX_LENGTH}")

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        logging_steps=LOGGING_STEPS,
        load_best_model_at_end=False,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        resume_from_checkpoint=resume_from,
        max_length=MAX_LENGTH,
        packing=False,
        dataset_text_field=None,
    )

    logger.info("\nInitialising SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        formatting_func=make_formatting_func(tokenizer),
    )

    logger.info("\n" + "=" * 65)
    logger.info("STARTING v3 TRAINING RUN")
    logger.info("=" * 65)

    start_time = datetime.now()
    trainer.train(resume_from_checkpoint=resume_from)
    elapsed = datetime.now() - start_time
    logger.info(f"\nTraining complete! Time: {elapsed}")

    # ── Save final model ───────────────────────────────────────────────────────
    logger.info("\nSaving final LoRA adapter...")
    final_model_path = OUTPUT_DIR / "final"
    trainer.model.save_pretrained(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))
    logger.info(f"Model saved to: {final_model_path}")

    summary = {
        "model_version": "glass-expert-v3",
        "base_model": BASE_MODEL,
        "train_examples": len(train_dataset),
        "val_examples": len(val_dataset),
        "epochs": NUM_EPOCHS,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "learning_rate": LEARNING_RATE,
        "max_length": MAX_LENGTH,
        "training_time": str(elapsed),
        "gpu": gpu_name,
        "completed_at": datetime.now().isoformat(),
        "improvements": [
            "All Kelvin temperatures converted to Celsius",
            "426 targeted patch Q&A pairs added",
            "System prompt enforcing units added",
            "Deduplication applied",
        ],
    }
    with open(OUTPUT_DIR / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("\n" + "=" * 65)
    logger.info("GLASS EXPERT v3 FINE-TUNING COMPLETE!")
    logger.info(f"Model: {final_model_path.resolve()}")
    logger.info("Next: update model_service/serve.py to point to this model")
    logger.info("      and increase max_new_tokens to 768")
    logger.info("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Glass Expert AI v3 Fine-tuning")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()
    main(resume=args.resume)