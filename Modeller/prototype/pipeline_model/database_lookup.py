"""Load and query the local JA4+ database.

This file is the ONLY place in the pipeline that reads the JSON file from disk.
All other scripts import from here — never load the JSON directly.

WHY THIS EXISTS:
  The raw JSON has inconsistent key names (e.g. "ja4_fingerprint" vs "ja4").
  This file normalises every row into a typed DatabaseRecord so the rest of
  the code gets clean, consistent fields.

  It also caches the loaded records in memory (_CACHE) so the file is only
  read once per process, no matter how many times load_db() is called.
"""

from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

# DatabaseRecord is the typed row shape; infer_category guesses the category
# from the application name when the JSON row has no explicit category field.
from .models import DatabaseRecord, infer_category

# ── hardcoded dataset path ────────────────────────────────────────────────────
# Walk up one level from pipeline_model/ to reach the repo root, then into
# "Datasett/".
_ROOT   = Path(__file__).resolve().parents[3]   # pipeline_model/ → prototype/ → Modeller/ → repo root
DB_PATH = _ROOT / "Datasett" / "categorized_custom_db.json"

# ── module-level cache (loaded once per process) ──────────────────────────────
# The first call to load_db() fills this; subsequent calls return it instantly.
_CACHE: list[DatabaseRecord] | None = None


def load_db() -> list[DatabaseRecord]:
    """Load the database from disk (cached after the first call).

    Returns a list of DatabaseRecord objects — one per JSON entry.
    Raises FileNotFoundError if the JSON file is missing.
    """
    global _CACHE
    if _CACHE is not None:
        return _CACHE   # already in memory — skip disk I/O
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found at {DB_PATH}")
    with DB_PATH.open("r", encoding="utf-8") as f:
        raw: list[dict[str, Any]] = json.load(f)
    # Convert every raw dict row into a typed DatabaseRecord.
    _CACHE = [_normalize(row) for row in raw if isinstance(row, dict)]
    print(f"[database_lookup] Loaded {len(_CACHE)} records from {DB_PATH.name}")
    return _CACHE


def _normalize(row: dict[str, Any]) -> DatabaseRecord:
    """Convert one raw JSON dict into a clean DatabaseRecord.

    Because different data sources use slightly different key names
    (e.g. "ja4" vs "ja4_fingerprint"), this function checks both variants
    and picks whichever is present.
    """
    # Accept both "ja4_fingerprint" (older format) and "ja4" (newer format).
    ja4         = row.get("ja4_fingerprint")    or row.get("ja4")
    ja4s        = row.get("ja4s_fingerprint")   or row.get("ja4s")
    ja4_string  = row.get("ja4_fingerprint_string")
    ja4s_string = row.get("ja4s_fingerprint_string")
    ja4t        = row.get("ja4t_fingerprint")   or row.get("ja4t")
    ja4ts       = row.get("ja4ts_fingerprint")  or row.get("ja4ts")

    application  = (row.get("application") or "").strip() or None
    explicit_cat = row.get("category") or row.get("traffic_category") or row.get("family")
    # infer_category uses keyword rules when no explicit category is provided.
    category = infer_category(application, explicit_cat)

    # Parse count (how many times this fingerprint was observed).
    count_raw = row.get("count", 1)
    try:
        count = int(count_raw)
    except (TypeError, ValueError):
        count = 1

    # Anything that isn't a reserved/known field goes into metadata for reference.
    reserved = {
        "application", "category", "traffic_category", "family",
        "ja4", "ja4s", "ja4t", "ja4ts",
        "ja4_fingerprint", "ja4s_fingerprint", "ja4t_fingerprint", "ja4_fingerprint_string",
        "ja4s_fingerprint_string", "ja4ts_fingerprint",
        "count",
    }
    metadata = {k: v for k, v in row.items() if k not in reserved and v is not None}

    return DatabaseRecord(
        ja4=ja4, ja4s=ja4s, ja4_string=ja4_string, ja4s_string=ja4s_string,
        ja4t=ja4t, ja4ts=ja4ts,
        application=application, category=category,
        count=count, metadata=metadata,
    )


# ── query helpers ─────────────────────────────────────────────────────────────
# These are thin wrappers over load_db() for common lookup patterns.

def find_by_ja4(ja4: str) -> list[DatabaseRecord]:
    """Return all records that have this exact JA4 hash."""
    key = (ja4 or "").strip().lower()
    return [r for r in load_db() if (r.ja4 or "").strip().lower() == key]


def find_by_ja4_and_ja4s(ja4: str, ja4s: str) -> list[DatabaseRecord]:
    """Return all records that match BOTH the JA4 and JA4S hashes."""
    k4 = (ja4  or "").strip().lower()
    ks = (ja4s or "").strip().lower()
    return [
        r for r in load_db()
        if (r.ja4  or "").strip().lower() == k4
        and (r.ja4s or "").strip().lower() == ks
    ]


def train_test_split(
    seed: int = 42,
    test_ratio: float = 0.2,
) -> tuple[list[DatabaseRecord], list[DatabaseRecord]]:
    """Return a reproducible 80 / 20 train / test split of the database.

    Splits PER APPLICATION so every application appears in both halves,
    even if it has very few records.  Unlabeled rows (no application) are
    put in the training set and not used during evaluation.

    Parameters
    ----------
    seed       — random seed for reproducibility
    test_ratio — fraction of each app's records to put in test (default 20%)
    """
    import random
    rng = random.Random(seed)

    # Group all records by their application label.
    groups: dict[str, list[DatabaseRecord]] = defaultdict(list)
    unlabeled: list[DatabaseRecord] = []
    for record in load_db():
        if record.application:
            groups[record.application].append(record)
        else:
            unlabeled.append(record)  # no label → only useful for training context

    train: list[DatabaseRecord] = []
    test:  list[DatabaseRecord] = []

    # For each application, shuffle its records and carve off the test slice.
    for app_records in groups.values():
        shuffled = list(app_records)
        rng.shuffle(shuffled)
        n_test   = max(1, int(len(shuffled) * test_ratio))  # at least 1 test record
        test.extend(shuffled[:n_test])
        train.extend(shuffled[n_test:])

    print(f"[database_lookup] Train/Test Split specs:")
    print(f"  Seed: {seed}, Test Ratio: {test_ratio*100:.0f}%")
    print(f"  Training Set: {len(train)} records, Test Set: {len(test)} records")

    return train, test
