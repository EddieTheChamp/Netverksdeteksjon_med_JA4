"""Iteration 2: Evaluate the custom (Egenlagd) dictionary against the test split.

Builds an in-memory exact-match index from the 80% training split of
categorized_custom_db.json and tests on the 20% test split.

Fallback priority:
    ja4+ja4s+ja4t+ja4ts → ja4+ja4s+ja4ts → ja4+ja4s+ja4t → ja4+ja4s → ja4 only

Usage:
    python eval_egenlagd.py
"""

from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "prototype"))

from pipeline_model import database_lookup
from pipeline_model.models import DatabaseRecord

# ── paths ─────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent  # Modeller/
RESULTS_DIR = _ROOT.parent / "results"
OUTPUT_FILE = RESULTS_DIR / "egenlagd_results.json"

# Lookup plans in priority order (most strict → least strict)
_LOOKUP_PLANS: list[tuple[str, tuple[str, ...]]] = [
    ("ja4_ja4s_ja4ts",      ("ja4", "ja4s", "ja4ts")),
    ("ja4_ja4s",            ("ja4", "ja4s")),
    ("ja4_ja4ts",           ("ja4", "ja4ts")),
    ("ja4_only",            ("ja4",)),
]


def _field(record: DatabaseRecord, name: str) -> str:
    return ((getattr(record, name) or "").strip().lower())


def build_index(train_records: list[DatabaseRecord]) -> dict[str, dict[str, list[DatabaseRecord]]]:
    """Build multi-key index from training records."""
    index: dict[str, dict[str, list[DatabaseRecord]]] = {
        mode: defaultdict(list) for mode, _ in _LOOKUP_PLANS
    }
    for record in train_records:
        for mode, fields in _LOOKUP_PLANS:
            values = [_field(record, f) for f in fields]
            if all(values):
                key = "|".join(values)
                index[mode][key].append(record)
    return index


def lookup(index: dict, record: DatabaseRecord) -> tuple[str, list[str], int, str]:
    """Find best match for a test record using the priority fallback chain.
    If ja4_ja4s fails, combines hits from ja4_only and ja4s_only.
    Returns (top_prediction, top_k_apps, matches_count, match_mode).
    """
    # 1. Full match: JA4 + JA4S + JA4TS
    mode = "ja4_ja4s_ja4ts"
    values = [_field(record, f) for f in ("ja4", "ja4s", "ja4ts")]
    if all(values):
        hits = index[mode].get("|".join(values), [])
        if hits:
            return _process_hits(hits, mode)

    # 2a. Delvis match: JA4 + JA4S
    mode = "ja4_ja4s"
    values = [_field(record, f) for f in ("ja4", "ja4s")]
    if all(values):
        hits = index[mode].get("|".join(values), [])
        if hits:
            return _process_hits(hits, mode)

    # 2b. Delvis match: JA4 + JA4TS
    mode = "ja4_ja4ts"
    values = [_field(record, f) for f in ("ja4", "ja4ts")]
    if all(values):
        hits = index[mode].get("|".join(values), [])
        if hits:
            return _process_hits(hits, mode)

    # 3. Individuell: JA4 alene
    ja4_val = _field(record, "ja4")
    if ja4_val:
        hits = index["ja4_only"].get(ja4_val, [])
        if hits:
            return _process_hits(hits, "ja4_only")

    return "Unknown", [], 0, "none"

def _process_hits(hits: list[DatabaseRecord], mode: str) -> tuple[str, list[str], int, str]:
    app_counts: dict[str, int] = defaultdict(int)
    for hit in hits:
        if hit.application:
            app_counts[hit.application] += max(hit.count, 1)

    if not app_counts:
        return "Unknown", [], 0, "none"

    ranked = sorted(app_counts.keys(), key=lambda a: app_counts[a], reverse=True)
    return ranked[0], ranked, len(ranked), mode


def run() -> list[dict]:
    """Run the Egenlagd evaluation. Returns result records."""
    train_records, test_records = database_lookup.train_test_split(seed=42, test_ratio=0.2)
    print(f"[eval_egenlagd] Training on {len(train_records)} records...")
    index = build_index(train_records)
    print(f"[eval_egenlagd] Evaluating {len(test_records)} test records...")

    results = []
    correct = 0
    for record in test_records:
        true_app = record.application or "Unknown"
        prediction, top_k, matches_count, match_mode = lookup(index, record)
        is_correct = prediction.lower() == true_app.lower()
        if is_correct:
            correct += 1
        results.append({
            "true_app": true_app,
            "prediction": prediction,
            "top_k": top_k[:5],
            "matches_count": matches_count,
            "match_mode": match_mode,
            "correct": is_correct,
            "ja4": record.ja4,
        })

    total = len(results)
    acc = correct / total * 100 if total else 0
    print(f"\n[eval_egenlagd] Results: {correct}/{total} correct — Top-1 Accuracy: {acc:.1f}%")

    unique_count = sum(1 for r in results if r["matches_count"] == 1)
    collision_count = sum(1 for r in results if r["matches_count"] > 1)
    unknown_count = sum(1 for r in results if r["matches_count"] == 0)
    print(f"           Unique: {unique_count} | Collisions: {collision_count} | Unknown: {unknown_count}")

    return results


if __name__ == "__main__":
    res = run()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(res, f, indent=4)
    print(f"[eval_egenlagd] Results saved to {OUTPUT_FILE}")
