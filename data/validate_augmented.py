"""Validate augmented JSONL outputs produced by data/augment.py.

Checks:
1. Schema integrity — every row has required fields in allowed enum values
2. Error rate — rows flagged with _augment_error during inference
3. Label distributions for urgency and sentiment per split
4. Cross-tabulation of urgency/sentiment against existing labels
5. Sample rows for qualitative inspection
"""

import json
from collections import Counter
from pathlib import Path

from augment import URGENCY_VALUES, SENTIMENT_VALUES

SPLITS_DIR = Path(__file__).parent / "splits"
FILES = {
    "train_subset": "train_subset.augmented.jsonl",
    "val": "val.augmented.jsonl",
    "test": "test.augmented.jsonl",
}
REQUIRED_FIELDS = ("flags", "instruction", "category", "intent", "urgency", "sentiment")


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path)]


def check_schema(rows: list[dict]) -> dict:
    missing_fields = 0
    bad_urgency = Counter()
    bad_sentiment = Counter()
    errors = 0
    for r in rows:
        if r.get("_augment_error"):
            errors += 1
            continue
        if not all(f in r and r[f] is not None for f in REQUIRED_FIELDS):
            missing_fields += 1
            continue
        if r["urgency"] not in URGENCY_VALUES:
            bad_urgency[r["urgency"]] += 1
        if r["sentiment"] not in SENTIMENT_VALUES:
            bad_sentiment[r["sentiment"]] += 1
    return {
        "errors": errors,
        "missing_fields": missing_fields,
        "bad_urgency": bad_urgency,
        "bad_sentiment": bad_sentiment,
    }


def label_distribution(rows: list[dict], key: str) -> Counter:
    return Counter(r[key] for r in rows if r.get(key) is not None)


def cross_tab(rows: list[dict], row_key: str, col_key: str) -> dict[str, Counter]:
    out: dict[str, Counter] = {}
    for r in rows:
        if r.get(row_key) is None or r.get(col_key) is None:
            continue
        out.setdefault(r[row_key], Counter())[r[col_key]] += 1
    return out


def print_distribution(label: str, counts: Counter, total: int) -> None:
    print(f"  {label}:")
    for key, n in counts.most_common():
        print(f"    {key:12s} {n:5d}  ({100 * n / total:5.2f}%)")


def print_cross_tab(title: str, table: dict[str, Counter], cols: tuple[str, ...]) -> None:
    print(f"\n  {title}")
    header = f"    {'':18s}" + "".join(f"{c:>12s}" for c in cols) + f"{'total':>10s}"
    print(header)
    for row_key in sorted(table):
        counts = table[row_key]
        total = sum(counts.values())
        line = f"    {row_key:18s}"
        for c in cols:
            n = counts.get(c, 0)
            pct = 100 * n / total if total else 0
            line += f"  {n:4d} ({pct:4.0f}%)"
        line += f"{total:>10d}"
        print(line)


def print_samples(rows: list[dict], n: int = 6) -> None:
    import random
    rng = random.Random(0)
    samples = rng.sample(rows, min(n, len(rows)))
    for r in samples:
        print(f'    [{r["intent"]:28s}] urg={r["urgency"]:6s} sent={r["sentiment"]:11s}  "{r["instruction"]}"')


def main() -> None:
    for name, filename in FILES.items():
        path = SPLITS_DIR / filename
        if not path.exists():
            print(f"MISSING: {path}")
            continue

        rows = load(path)
        print(f"\n{'=' * 70}")
        print(f"{name}  —  {len(rows)} rows  ({path.name})")
        print("=" * 70)

        schema = check_schema(rows)
        valid = [r for r in rows if not r.get("_augment_error")]
        print(f"\n  Schema check:")
        print(f"    errors (augment failures): {schema['errors']}")
        print(f"    missing required fields:   {schema['missing_fields']}")
        print(f"    out-of-enum urgency:       {dict(schema['bad_urgency']) or 'none'}")
        print(f"    out-of-enum sentiment:     {dict(schema['bad_sentiment']) or 'none'}")

        print("\n  Label distributions:")
        print_distribution("urgency", label_distribution(valid, "urgency"), len(valid))
        print_distribution("sentiment", label_distribution(valid, "sentiment"), len(valid))

        print_cross_tab(
            "urgency × sentiment",
            cross_tab(valid, "sentiment", "urgency"),
            URGENCY_VALUES,
        )

        print_cross_tab(
            "category × urgency",
            cross_tab(valid, "category", "urgency"),
            URGENCY_VALUES,
        )

        print_cross_tab(
            "category × sentiment",
            cross_tab(valid, "category", "sentiment"),
            SENTIMENT_VALUES,
        )

        print(f"\n  Random samples (seed=0):")
        print_samples(valid, n=8)


if __name__ == "__main__":
    main()
