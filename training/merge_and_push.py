"""Merge the trained LoRA adapter into the base bf16 weights, save locally,
and optionally push the merged model to HF Hub.

Critical: the base is loaded fresh in bf16 here, NOT in 4-bit. QLoRA
quantization during training was a memory optimization only — the permanent
artifact must be full precision so that downstream quantization (Q4_K_M for
llama.cpp serving) happens from a clean baseline.

Designed to run on the same CUDA pod as train_qlora.py (or any machine with
enough RAM/VRAM to hold 3B in bf16, which is ~6 GB).

Usage:

    # Merge only, save locally
    python training/merge_and_push.py --adapter-dir training/output/final_adapter

    # Merge and push to HF Hub (requires `huggingface-cli login` first)
    python training/merge_and_push.py \\
        --adapter-dir training/output/final_adapter \\
        --push-to-hub \\
        --hub-id <your-username>/ticket-classifier-qwen2.5-3b
"""

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter-dir", required=True)
    p.add_argument("--output-dir", default="training/output/merged")
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--hub-id", default=None, help="HF Hub repo id (e.g. user/name)")
    p.add_argument("--private", action="store_true", default=True)
    args = p.parse_args()

    adapter_dir = Path(args.adapter_dir)
    output_dir = Path(args.output_dir)

    print(f"Loading base model ({BASE_MODEL}) in bf16...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )

    print(f"Attaching adapter from {adapter_dir}...")
    model = PeftModel.from_pretrained(base, str(adapter_dir))

    print("Merging adapter into base weights (bf16)...")
    merged = model.merge_and_unload()

    print(f"Saving merged model to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output_dir, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(output_dir)
    print(f"Done. Merged model at {output_dir}")

    if args.push_to_hub:
        if not args.hub_id:
            raise SystemExit("--push-to-hub requires --hub-id")
        print(f"Pushing to HF Hub: {args.hub_id} (private={args.private})")
        merged.push_to_hub(args.hub_id, private=args.private)
        tokenizer.push_to_hub(args.hub_id, private=args.private)
        print("Pushed.")


if __name__ == "__main__":
    main()
