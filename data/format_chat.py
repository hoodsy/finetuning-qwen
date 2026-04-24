"""Convert augmented JSONL rows into conversational (messages) format for SFTTrainer.

Each output row is a single {"messages": [...]} record with a short student
system prompt, a user turn mirroring the teacher's `Message: "..."` framing,
and an assistant turn containing the compact target JSON with a fixed field
order (category, intent, urgency, sentiment).

`flags` is intentionally dropped from training data — it remains available on
the unformatted augmented files for post-training robustness analysis.
"""

import json
from pathlib import Path

SPLITS_DIR = Path(__file__).parent / "splits"
INPUT_FILES = {
    "train_subset": "train_subset.augmented.jsonl",
    "val": "val.augmented.jsonl",
    "test": "test.augmented.jsonl",
}

STUDENT_SYSTEM_PROMPT = """You are a customer support ticket classifier. For each message, output only a JSON object with these fields in order: category, intent, urgency, sentiment.

Allowed values:
- category: ACCOUNT, CANCEL, CONTACT, DELIVERY, FEEDBACK, INVOICE, ORDER, PAYMENT, REFUND, SHIPPING, SUBSCRIPTION
- intent: one of 27 support intents (cancel_order, track_order, recover_password, change_order, get_refund, etc.)
- urgency: low, medium, high
- sentiment: frustrated, confused, neutral

Precedence: if a message shows both frustration and confusion, label `frustrated`.

Output: compact JSON with no whitespace. No preamble, no explanation."""


def format_target(row: dict) -> str:
    """Build the compact target JSON the assistant turn should emit.

    Fixed field order (category → intent → urgency → sentiment) is enforced
    by dict insertion order. `separators=(',', ':')` strips whitespace.
    """
    target = {
        "category": row["category"],
        "intent": row["intent"],
        "urgency": row["urgency"],
        "sentiment": row["sentiment"],
    }
    return json.dumps(target, separators=(",", ":"))


def format_row(row: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
            {"role": "user", "content": f'Message: "{row["instruction"]}"'},
            {"role": "assistant", "content": format_target(row)},
        ]
    }


def main() -> None:
    for name, filename in INPUT_FILES.items():
        in_path = SPLITS_DIR / filename
        out_path = SPLITS_DIR / filename.replace(".augmented.jsonl", ".chat.jsonl")

        rows_in = [json.loads(line) for line in open(in_path)]
        rows_out = [format_row(r) for r in rows_in]

        with open(out_path, "w") as f:
            for r in rows_out:
                f.write(json.dumps(r) + "\n")

        print(f"{name:14s}  {in_path.name}  →  {out_path.name}  ({len(rows_out)} rows)")

    print("\n=== Example formatted row (train_subset[0]) ===")
    first = json.loads(open(SPLITS_DIR / "train_subset.chat.jsonl").readline())
    for m in first["messages"]:
        print(f"\n[{m['role']}]")
        print(m["content"])


if __name__ == "__main__":
    main()
