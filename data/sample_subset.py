"""Sample a 3K-row stratified subset of train.jsonl for augmentation.

Stratifies on `intent` so all 27 classes stay balanced in the subset.
Writes to data/splits/train_subset.jsonl. Val and test are augmented in
full, so no subset is needed for them.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

IN_PATH = Path(__file__).parent / "splits" / "train.jsonl"
OUT_PATH = Path(__file__).parent / "splits" / "train_subset.jsonl"
TARGET_SIZE = 3000
SEED = 42


def main() -> None:
    rows = [json.loads(line) for line in open(IN_PATH)]

    by_intent: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_intent[r["intent"]].append(r)

    n_intents = len(by_intent)
    per_intent = TARGET_SIZE // n_intents
    remainder = TARGET_SIZE - per_intent * n_intents

    rng = random.Random(SEED)
    subset: list[dict] = []
    for i, intent in enumerate(sorted(by_intent)):
        n = per_intent + (1 if i < remainder else 0)
        subset.extend(rng.sample(by_intent[intent], n))

    rng.shuffle(subset)

    with open(OUT_PATH, "w") as f:
        for r in subset:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {OUT_PATH}")
    print(f"  rows: {len(subset)}")
    print(f"  intents: {n_intents}")
    print(f"  per intent: {per_intent} (+1 for the first {remainder} intents)")


if __name__ == "__main__":
    main()
