"""
Glass Expert AI — QLoRA Fine-tuning v2
Full training run — no early stopping, all 7191 steps
"""
import sys, json, logging, argparse, math
from pathlib import Path
from datetime import datetime
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer, SFTConfig

BASE_MODEL  = "meta-llama/Meta-Llama-3.1-8B-Instruct"
OUTPUT_DIR  = Path("models/glass-expert-v2")
TRAIN_FILE  = Path("data/processed/llama_train.jsonl")
VAL_FILE    = Path("data/processed/llama_val.jsonl")
LOG_DIR     = Path("logs/finetune")
LORA_R=64; LORA_ALPHA=128; LORA_DROPOUT=0.05
LORA_TARGET_MODULES=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
NUM_EPOCHS=3; BATCH_SIZE=2; GRAD_ACCUM_STEPS=8; LEARNING_RATE=1e-4; MAX_LENGTH=2048
WEIGHT_DECAY=0.001; SAVE_STEPS=200; EVAL_STEPS=200; LOGGING_STEPS=50; SAVE_TOTAL_LIMIT=5

LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

def load_jsonl_dataset(filepath):
    records = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line: records.append(json.loads(line))
    logger.info(f"Loaded {len(records):,} examples from {filepath.name}")
    return Dataset.from_list(records)

def make_formatting_func(tokenizer):
    def formatting_func(example):
        return tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False)
    return formatting_func

def main(resume=False):
    logger.info("="*65)
    logger.info("Glass Expert AI — QLoRA Fine-tuning v2 (Full Run)")
    logger.info(f"Base model : {BASE_MODEL}")
    logger.info("="*65)
    for f in [TRAIN_FILE, VAL_FILE]:
        if not f.exists(): logger.error(f"File not found: {f}"); sys.exit(1)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available(): logger.error("CUDA not available!"); sys.exit(1)
    gpu_name = torch.cuda.get_device_name(0)
    gpu_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"GPU: {gpu_name}  ({gpu_vram:.1f} GB VRAM)")

    logger.info("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    logger.info("Tokenizer loaded.")

    logger.info("\nLoading datasets...")
    train_dataset = load_jsonl_dataset(TRAIN_FILE)
    val_dataset   = load_jsonl_dataset(VAL_FILE)
    logger.info(f"Train: {len(train_dataset):,}  Val: {len(val_dataset):,}")

    bnb_config = BitsAndBytesConfig(load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)

    logger.info(f"\nLoading base model (from cache)...")
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True, dtype=torch.bfloat16,
        attn_implementation="eager")
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model = prepare_model_for_kbit_training(model)

    logger.info("\nApplying LoRA adapters...")
    lora_config = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        bias="none", task_type=TaskType.CAUSAL_LM, target_modules=LORA_TARGET_MODULES)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    logger.info("\nConfiguring training arguments...")
    resume_from = None
    if resume:
        checkpoints = sorted(OUTPUT_DIR.glob("checkpoint-*"))
        if checkpoints: resume_from = str(checkpoints[-1]); logger.info(f"Resuming: {resume_from}")
        else: logger.warning("No checkpoint found.")

    steps_per_epoch = math.ceil(len(train_dataset) / (BATCH_SIZE * GRAD_ACCUM_STEPS))
    total_steps = steps_per_epoch * NUM_EPOCHS
    warmup_steps = max(1, int(total_steps * 0.03))

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR), num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE, per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS, learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY, lr_scheduler_type="cosine", warmup_steps=warmup_steps,
        fp16=False, bf16=True,
        max_grad_norm=0.3, gradient_checkpointing=True,
        optim="paged_adamw_8bit", save_strategy="steps", save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT, eval_strategy="steps", eval_steps=EVAL_STEPS,
        logging_steps=LOGGING_STEPS, load_best_model_at_end=False,
        report_to="none", dataloader_num_workers=0, remove_unused_columns=False,
        resume_from_checkpoint=resume_from,
        max_length=MAX_LENGTH, packing=False, dataset_text_field=None,
    )

    logger.info("\nInitialising SFTTrainer...")
    trainer = SFTTrainer(
        model=model, train_dataset=train_dataset, eval_dataset=val_dataset,
        args=training_args, formatting_func=make_formatting_func(tokenizer),
    )

    logger.info("\n" + "="*65)
    logger.info("STARTING FULL TRAINING RUN (no early stopping)")
    logger.info(f"  Epochs: {NUM_EPOCHS}  Effective batch: {BATCH_SIZE*GRAD_ACCUM_STEPS}")
    logger.info(f"  Steps/epoch: {steps_per_epoch:,}  Total: {total_steps:,}  Warmup: {warmup_steps}")
    logger.info(f"  Learning rate: {LEARNING_RATE} (lowered from 2e-4)")
    logger.info("="*65)

    start_time = datetime.now()
    trainer.train(resume_from_checkpoint=resume_from)
    elapsed = datetime.now() - start_time
    logger.info(f"\nTraining complete! Time: {elapsed}")

    logger.info("\nSaving final LoRA adapter...")
    final_model_path = OUTPUT_DIR / "final"
    trainer.model.save_pretrained(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))
    logger.info(f"Saved to: {final_model_path}")

    summary = {"base_model": BASE_MODEL, "train_examples": len(train_dataset),
        "val_examples": len(val_dataset), "epochs": NUM_EPOCHS, "lora_r": LORA_R,
        "learning_rate": LEARNING_RATE, "training_time": str(elapsed), "gpu": gpu_name,
        "completed_at": datetime.now().isoformat()}
    with open(OUTPUT_DIR / "training_summary.json", "w") as f: json.dump(summary, f, indent=2)
    logger.info("\nFINE-TUNING COMPLETE!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    main(resume=args.resume)
