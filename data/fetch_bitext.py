"""Fetch the Bitext customer support dataset and write stratified
train/val/test JSONL splits to data/splits/.

Stratification is done on `intent` (27 nearly-uniform classes), which
balances the coarser `category` label as a side effect. The `response`
column is dropped since we're not training a reply generator. The
`flags` column is kept through splitting so we can do per-flag
robustness analysis at eval time, even though it isn't a training input.
"""

from collections import Counter
from pathlib import Path

from datasets import load_dataset

DATASET_ID = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
OUT_DIR = Path(__file__).parent / "splits"
SEED = 42


def main() -> None:
    ds = load_dataset(DATASET_ID, split="train")
    ds = ds.remove_columns(["response"])

    # train_test_split(stratify_by_column=...) needs a ClassLabel feature, but
    # we want `intent` to stay as a readable string in the JSONL output. Add a
    # throwaway encoded column, stratify on that, then drop it after splitting.
    ds = ds.add_column("_strat_intent", ds["intent"])
    ds = ds.class_encode_column("_strat_intent")

    # Two-stage stratified split: 80/20, then split the 20 into 50/50 → 80/10/10.
    stage1 = ds.train_test_split(
        test_size=0.2, stratify_by_column="_strat_intent", seed=SEED
    )
    stage2 = stage1["test"].train_test_split(
        test_size=0.5, stratify_by_column="_strat_intent", seed=SEED
    )
    splits = {
        "train": stage1["train"].remove_columns(["_strat_intent"]),
        "val": stage2["train"].remove_columns(["_strat_intent"]),
        "test": stage2["test"].remove_columns(["_strat_intent"]),
    }

    OUT_DIR.mkdir(exist_ok=True)
    for name, split in splits.items():
        path = OUT_DIR / f"{name}.jsonl"
        split.to_json(str(path), lines=True)
        print(f"Wrote {path}  ({len(split)} rows)")
    print()

    print("=== Intent distribution (first + last 3 per split) ===")
    for name, split in splits.items():
        counts = Counter(split["intent"])
        total = len(split)
        print(f"\n-- {name} (n={total}, unique intents={len(counts)}) --")
        ordered = counts.most_common()
        for label, n in ordered[:3] + ordered[-3:]:
            print(f"  {label:30s} {n:5d}  ({100 * n / total:5.2f}%)")

    print()
    print("=== Category distribution per split ===")
    for name, split in splits.items():
        counts = Counter(split["category"])
        total = len(split)
        print(f"\n-- {name} (n={total}) --")
        for label, n in counts.most_common():
            print(f"  {label:30s} {n:5d}  ({100 * n / total:5.2f}%)")


if __name__ == "__main__":
    main()
