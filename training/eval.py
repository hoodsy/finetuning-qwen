"""Evaluate fine-tuned Qwen2.5-3B against two baselines on the test set:
the base Qwen2.5-3B-Instruct (no fine-tune) and GPT-4o-mini via the OpenAI API.

Metrics reported per model:
  - per-field accuracy (category, intent, urgency, sentiment)
  - minority-class recall (urgency=high, sentiment=frustrated)
  - format-failure rate (non-parseable JSON or wrong enum values)

Local models run via transformers.generate in bf16, greedy decoding (no
sampling) so numbers are reproducible. No guided decoding — we want to see
real format reliability, not a constrained-decode upper bound.

Usage:

    # Full eval (all three models)
    python training/eval.py \\
        --fine-tuned-dir training/output/merged \\
        --test-file data/splits/test.chat.jsonl \\
        --output-file training/output/eval_results.json

    # Skip the GPT-4o-mini baseline (no API key handy)
    python training/eval.py --fine-tuned-dir ... --skip-openai

    # Quick smoke test on 50 rows
    python training/eval.py --fine-tuned-dir ... --limit 50
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
FIELDS = ("category", "intent", "urgency", "sentiment")
MINORITY_CLASSES = (("urgency", "high"), ("sentiment", "frustrated"))
MAX_NEW_TOKENS = 80


# ---------- Output parsing ----------


def parse_output(text: str) -> dict | None:
    """Parse a model response into a dict with the four required fields.
    Returns None on any parse or schema failure."""
    text = text.strip()
    # Some models wrap JSON in markdown fences
    if text.startswith("```"):
        lines = text.split("\n", 1)
        text = lines[1] if len(lines) == 2 else ""
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        if text.startswith("json"):
            text = text[4:].strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    for field in FIELDS:
        if field not in obj or not isinstance(obj[field], str):
            return None
    return obj


# ---------- Local inference (transformers) ----------


def generate_local(model, tokenizer, message_lists: list[list[dict]], batch_size: int) -> list[str]:
    """Run greedy generation over a list of message-lists. Returns decoded response strings."""
    # Left padding is required for batched decoder-only generation so the
    # response tokens line up at the right-hand side of the output tensor.
    tokenizer.padding_side = "left"

    outputs: list[str] = []
    t0 = time.monotonic()
    for i in range(0, len(message_lists), batch_size):
        batch = message_lists[i : i + batch_size]
        prompts = [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in batch
        ]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        input_len = enc["input_ids"].shape[1]
        response_ids = gen[:, input_len:]
        for ids in response_ids:
            outputs.append(tokenizer.decode(ids, skip_special_tokens=True).strip())

        if (i // batch_size) % 10 == 0:
            rate = (i + len(batch)) / max(time.monotonic() - t0, 1e-6)
            print(f"  {i + len(batch)}/{len(message_lists)}  {rate:.1f} req/s")

    return outputs


# ---------- OpenAI inference ----------


async def generate_openai(
    model_name: str, message_lists: list[list[dict]], concurrency: int = 16
) -> list[str]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI()  # reads OPENAI_API_KEY from env
    sem = asyncio.Semaphore(concurrency)

    async def one(messages):
        async with sem:
            try:
                resp = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=MAX_NEW_TOKENS,
                    response_format={"type": "json_object"},
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                return f"__ERROR__ {e!r}"

    results = await asyncio.gather(*(one(m) for m in message_lists))
    return list(results)


# ---------- Metrics ----------


def compute_metrics(predictions: list[dict | None], gold: list[dict]) -> dict:
    n = len(gold)
    format_failures = sum(1 for p in predictions if p is None)
    m: dict[str, float] = {
        "n_rows": n,
        "format_failures": format_failures,
        "format_failure_rate": format_failures / n,
    }

    for field in FIELDS:
        correct = sum(
            1 for p, g in zip(predictions, gold) if p is not None and p[field] == g[field]
        )
        m[f"{field}_accuracy"] = correct / n

    # Recall on minority classes (true positives / all actual positives).
    # Penalizes models that never predict rare classes even if their overall
    # accuracy looks reasonable.
    for field, cls in MINORITY_CLASSES:
        actual_pos = sum(1 for g in gold if g[field] == cls)
        tp = sum(
            1
            for p, g in zip(predictions, gold)
            if p is not None and g[field] == cls and p[field] == cls
        )
        m[f"{field}_{cls}_recall"] = tp / actual_pos if actual_pos else 0.0

    return m


# ---------- Orchestration ----------


def load_test(path: Path, limit: int) -> tuple[list[list[dict]], list[dict]]:
    """Returns (input_messages, gold_labels). input_messages strips the assistant turn."""
    rows = [json.loads(line) for line in open(path)]
    if limit:
        rows = rows[:limit]
    inputs = []
    gold = []
    for row in rows:
        messages = row["messages"]
        gold_assistant = messages[-1]["content"]
        inputs.append(messages[:-1])  # system + user only
        gold.append(json.loads(gold_assistant))
    return inputs, gold


def run_local(label: str, model_path: str, inputs: list[list[dict]], batch_size: int) -> list[str]:
    print(f"\n=== {label}: loading {model_path} ===")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"=== {label}: generating ({len(inputs)} rows) ===")
    raw = generate_local(model, tokenizer, inputs, batch_size)
    # Free GPU memory for the next model
    del model
    torch.cuda.empty_cache()
    return raw


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fine-tuned-dir", required=True, help="Path to merged fine-tuned model")
    p.add_argument("--test-file", default="data/splits/test.chat.jsonl")
    p.add_argument("--output-file", default="training/output/eval_results.json")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--skip-openai", action="store_true")
    p.add_argument("--openai-model", default="gpt-4o-mini")
    args = p.parse_args()

    inputs, gold = load_test(Path(args.test_file), args.limit)
    print(f"Test rows: {len(inputs)}")

    all_results: dict[str, dict] = {}

    # 1. Fine-tuned Qwen2.5-3B
    raw_ft = run_local("fine_tuned", args.fine_tuned_dir, inputs, args.batch_size)
    preds_ft = [parse_output(t) for t in raw_ft]
    all_results["fine_tuned"] = {
        "metrics": compute_metrics(preds_ft, gold),
        "raw_sample": raw_ft[:5],
    }

    # 2. Base Qwen2.5-3B-Instruct (no fine-tune)
    raw_base = run_local("base", BASE_MODEL, inputs, args.batch_size)
    preds_base = [parse_output(t) for t in raw_base]
    all_results["base_qwen2.5_3b"] = {
        "metrics": compute_metrics(preds_base, gold),
        "raw_sample": raw_base[:5],
    }

    # 3. GPT-4o-mini
    if not args.skip_openai:
        if not os.environ.get("OPENAI_API_KEY"):
            print("\n[SKIP] OPENAI_API_KEY not set — skipping GPT-4o-mini.")
        else:
            print(f"\n=== {args.openai_model}: calling OpenAI API ===")
            raw_oa = asyncio.run(generate_openai(args.openai_model, inputs))
            preds_oa = [parse_output(t) for t in raw_oa]
            all_results[args.openai_model] = {
                "metrics": compute_metrics(preds_oa, gold),
                "raw_sample": raw_oa[:5],
            }

    # ---------- Report ----------
    print("\n" + "=" * 80)
    print(f"{'model':<28s} {'cat':>7s} {'int':>7s} {'urg':>7s} {'sent':>7s}  {'fmt-fail':>9s}  {'urg=high':>10s} {'frust':>8s}")
    print("-" * 80)
    for name, result in all_results.items():
        m = result["metrics"]
        print(
            f"{name:<28s} "
            f"{m['category_accuracy']:>6.1%} "
            f"{m['intent_accuracy']:>6.1%} "
            f"{m['urgency_accuracy']:>6.1%} "
            f"{m['sentiment_accuracy']:>6.1%}  "
            f"{m['format_failure_rate']:>8.1%}  "
            f"{m['urgency_high_recall']:>9.1%} "
            f"{m['sentiment_frustrated_recall']:>7.1%}"
        )
    print("=" * 80)

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nWrote detailed results to {out_path}")


if __name__ == "__main__":
    main()
