"""Augmentation prompt and batch inference runner.

Labels synthetic `urgency` and `sentiment` fields onto Bitext rows using a
teacher model (default: Qwen2.5-72B-Instruct) served via vLLM's
OpenAI-compatible endpoint. Designed to run on a RunPod pod with vLLM
launched like:

    python -m vllm.entrypoints.openai.api_server \\
        --model Qwen/Qwen2.5-72B-Instruct \\
        --dtype bfloat16 \\
        --max-model-len 2048

Usage:
    # Local smoke test against any OpenAI-compatible endpoint:
    AUGMENT_BASE_URL=https://api.together.xyz/v1 \\
    AUGMENT_API_KEY=$TOGETHER_API_KEY \\
    AUGMENT_MODEL=Qwen/Qwen2.5-72B-Instruct-Turbo \\
    uv run python data/augment.py --input data/splits/train_subset.jsonl \\
        --output /tmp/aug_sample.jsonl --limit 20

    # Full run on RunPod pod (defaults point at localhost:8000):
    uv run python data/augment.py \\
        --input data/splits/train_subset.jsonl \\
        --output data/splits/train_subset.augmented.jsonl
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from openai import AsyncOpenAI

URGENCY_VALUES = ("low", "medium", "high")
SENTIMENT_VALUES = ("frustrated", "confused", "neutral")

SYSTEM_PROMPT = """You are a customer support ticket classifier. For each message, output ONLY a JSON object with two fields: `urgency` and `sentiment`.

urgency — how quickly this ticket should be acted on:
- high: account/payment/access blockers, explicit urgency language, anything preventing the customer from using the service
- medium: issues that need resolution but aren't actively blocking (refund tracking, order changes, standard complaints)
- low: informational queries, browsing questions, routine requests

sentiment — the customer's emotional posture:
- frustrated: explicit anger, profanity, stress markers, or emotional distress
- confused: uncertainty about the process ("I don't know", "how do I") without emotional markers
- neutral: routine, calm tone, no strong emotion

Precedence: if both frustration AND confusion are present, label `frustrated`.

Output format: a single JSON object with keys `urgency` and `sentiment`. No preamble, no explanation, no markdown fences.

Examples:

Message: "I cannot retrieve the bloody PIN code of my profile"
{"urgency": "high", "sentiment": "frustrated"}

Message: "where could I check purchase {{Order Number}} current status?"
{"urgency": "low", "sentiment": "neutral"}

Message: "show cancellation charges"
{"urgency": "low", "sentiment": "neutral"}

Message: "i dont know what to do to obtain my money back"
{"urgency": "medium", "sentiment": "confused"}

Message: "help filing a cosnumer reclamation against ur company"
{"urgency": "medium", "sentiment": "frustrated"}

Message: "how can I write a comment?"
{"urgency": "low", "sentiment": "confused"}"""

# JSON schema passed to vLLM's guided_json to guarantee well-formed,
# enum-constrained output. Removes JSON parse failures and hallucinated labels.
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "urgency": {"type": "string", "enum": list(URGENCY_VALUES)},
        "sentiment": {"type": "string", "enum": list(SENTIMENT_VALUES)},
    },
    "required": ["urgency", "sentiment"],
    "additionalProperties": False,
}


def build_messages(instruction: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f'Message: "{instruction}"'},
    ]


async def augment_one(
    client: AsyncOpenAI,
    row: dict,
    idx: int,
    model: str,
    sem: asyncio.Semaphore,
    use_guided_json: bool,
    max_retries: int,
) -> tuple[int, dict]:
    async with sem:
        extra_body = {"guided_json": OUTPUT_SCHEMA} if use_guided_json else None
        for attempt in range(max_retries + 1):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=build_messages(row["instruction"]),
                    temperature=0.0,
                    max_tokens=60,
                    extra_body=extra_body,
                )
                content = (resp.choices[0].message.content or "").strip()
                parsed = json.loads(content)
                urg, sent = parsed.get("urgency"), parsed.get("sentiment")
                if urg in URGENCY_VALUES and sent in SENTIMENT_VALUES:
                    return idx, {**row, "urgency": urg, "sentiment": sent}
            except (json.JSONDecodeError, KeyError, ValueError):
                pass
            except Exception:
                if attempt == max_retries:
                    break
                await asyncio.sleep(1 + attempt)
        return idx, {**row, "urgency": None, "sentiment": None, "_augment_error": True}


async def run(args: argparse.Namespace) -> None:
    in_path = Path(args.input)
    out_path = Path(args.output)
    rows = [json.loads(line) for line in open(in_path)]
    if args.limit:
        rows = rows[: args.limit]
    print(f"Loaded {len(rows)} rows from {in_path}", file=sys.stderr)
    print(f"  model: {args.model}", file=sys.stderr)
    print(f"  endpoint: {args.base_url}", file=sys.stderr)
    print(f"  concurrency: {args.concurrency}", file=sys.stderr)
    print(f"  guided_json: {args.guided_json}", file=sys.stderr)

    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    sem = asyncio.Semaphore(args.concurrency)

    tasks = [
        augment_one(client, r, i, args.model, sem, args.guided_json, args.max_retries)
        for i, r in enumerate(rows)
    ]

    results: list[dict | None] = [None] * len(rows)
    completed = 0
    errors = 0
    t0 = time.monotonic()

    for coro in asyncio.as_completed(tasks):
        idx, result = await coro
        results[idx] = result
        completed += 1
        if result.get("_augment_error"):
            errors += 1
        if completed % 100 == 0 or completed == len(rows):
            elapsed = time.monotonic() - t0
            rate = completed / elapsed if elapsed else 0
            print(
                f"  {completed}/{len(rows)}  "
                f"errors={errors}  "
                f"{rate:.1f} req/s  "
                f"elapsed={elapsed:.0f}s",
                file=sys.stderr,
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {out_path}  ({len(results)} rows, {errors} errors)", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int, default=0, help="Cap rows for smoke testing")
    p.add_argument(
        "--model",
        default=os.environ.get("AUGMENT_MODEL", "Qwen/Qwen2.5-72B-Instruct"),
    )
    p.add_argument(
        "--base-url",
        default=os.environ.get("AUGMENT_BASE_URL", "http://localhost:8000/v1"),
    )
    p.add_argument(
        "--api-key",
        default=os.environ.get("AUGMENT_API_KEY", "EMPTY"),
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=int(os.environ.get("AUGMENT_CONCURRENCY", "32")),
    )
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument(
        "--guided-json",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use vLLM's guided_json for schema-constrained output. Disable for hosted APIs that don't support it.",
    )
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
