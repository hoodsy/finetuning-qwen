"""QLoRA fine-tune Qwen2.5-3B-Instruct on the chat-formatted augmented dataset.

Designed to run on a CUDA GPU pod (RunPod). Not runnable on Mac —
bitsandbytes has no Apple Silicon backend.

Required deps on the pod:

    pip install "transformers>=4.46" "trl>=0.12" "peft>=0.13" \\
                "bitsandbytes>=0.44" accelerate datasets

Usage:

    cd finetuning
    python training/train_qlora.py --output-dir training/output

Expected wall time: ~30-45 min on a single A40 or 4090 for 2 epochs over
the 3K train subset. Saves the trained LoRA adapter (only) to
<output-dir>/final_adapter/. The adapter is then merged with the fresh
bf16 base via training/merge_and_push.py in a later step.
"""

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DATA_DIR = Path(__file__).parent.parent / "data" / "splits"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="training/output")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--per-device-batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--train-file",
        default=str(DATA_DIR / "train_subset.chat.jsonl"),
    )
    p.add_argument("--val-file", default=str(DATA_DIR / "val.chat.jsonl"))
    args = p.parse_args()

    # ---------- Datasets ----------
    train_ds = load_dataset("json", data_files=args.train_file, split="train")
    val_ds = load_dataset("json", data_files=args.val_file, split="train")
    print(f"train rows: {len(train_ds)}  val rows: {len(val_ds)}")

    # ---------- Tokenizer ----------
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    # Qwen already defines pad_token, but be defensive:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------- Base model: 4-bit NF4 with double-quant, bf16 compute ----------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    # device_map={"": 0} forces all modules onto GPU 0. Using "auto" sometimes
    # triggers false-positive CPU offload decisions even when the 4-bit model
    # fits easily in VRAM (common gotcha on H100 with small models).
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )
    # Silence cache warnings during training — generation cache is irrelevant here.
    model.config.use_cache = False

    # ---------- LoRA ----------
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ---------- SFT config ----------
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_seq_length=args.max_seq_length,
        packing=True,
        bf16=True,
        # paged_adamw_8bit keeps optimizer state in 8-bit on CPU with paging —
        # big memory win for QLoRA, minimal speed cost.
        optim="paged_adamw_8bit",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Only compute loss on the assistant turn — don't train the model to
        # regenerate the user's message or the system prompt.
        completion_only_loss=True,
        report_to="none",
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    trainer.train()

    # ---------- Save adapter only ----------
    adapter_dir = Path(args.output_dir) / "final_adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"Saved adapter + tokenizer to {adapter_dir}")


if __name__ == "__main__":
    main()
