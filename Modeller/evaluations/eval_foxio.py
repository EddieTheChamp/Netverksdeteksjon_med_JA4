"""eval_dictionary.py — Evaluate the FoxIO JA4+ dictionary per fingerprint type.

For each fingerprint type (JA4, JA4S, JA4T, JA4TS), every test record is
looked up independently in the dictionary.  Each lookup is classified as:

  * Unik match  — exactly 1 distinct application maps to that fingerprint value
  * Kollisjon   — more than 1 distinct application maps to that fingerprint value
  * Ukjent      — no entry in the dictionary for that fingerprint value

The result is a stacked horizontal bar chart (one bar per fingerprint type)
saved to results/dictionary_fingerprint_evaluation.png.

Usage:
    python eval_dictionary.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "prototype"))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

_ROOT = Path(__file__).resolve().parent        # Modeller/
sys.path.insert(0, str(_ROOT))

from pipeline_model import database_lookup

# ── paths ─────────────────────────────────────────────────────────────────────
FOXIO_DB_PATH = _ROOT / "data" / "models" / "ja4+_db.json"
RESULTS_DIR   = _ROOT / "results"
OUTPUT_PNG    = RESULTS_DIR / "dictionary_fingerprint_evaluation.png"

FINGERPRINTS = ["ja4", "ja4s", "ja4t", "ja4ts"]

# ── Norwegian display labels ──────────────────────────────────────────────────
LABEL_MAP = {
    "ja4":   "JA4",
    "ja4s":  "JA4S",
    "ja4t":  "JA4T",
    "ja4ts": "JA4TS",
}

PALETTE = {
    "Unik match": "#2ecc71",   # green
    "Kollisjon":  "#e67e22",   # orange
    "Ukjent":     "#e74c3c",   # red
}


# ── load FoxIO dictionary ─────────────────────────────────────────────────────

def _load_foxio_indices() -> dict[str, dict[str, int]]:
    """Build a per-fingerprint-type index: value → number of DB entries.

    NOTE: Most FoxIO records have no 'application' field, so we count *entries*
    (rows) per fingerprint value rather than distinct applications.  This gives
    the true picture of collisions: if two different rows share the same JA4
    hash, that fingerprint is ambiguous regardless of whether both rows have an
    application name.
    """
    if not FOXIO_DB_PATH.exists():
        raise FileNotFoundError(f"FoxIO database not found at {FOXIO_DB_PATH}")

    with FOXIO_DB_PATH.open("r", encoding="utf-8") as f:
        raw: list[dict] = json.load(f)

    # Canonical field names for each fingerprint in the FoxIO DB
    field_candidates: dict[str, list[str]] = {
        "ja4":   ["ja4_fingerprint", "ja4"],
        "ja4s":  ["ja4s_fingerprint", "ja4s"],
        "ja4t":  ["ja4t_fingerprint", "ja4t"],
        "ja4ts": ["ja4ts_fingerprint", "ja4ts"],
    }

    # value → count of DB rows that carry this fingerprint value
    indices: dict[str, dict[str, int]] = {fp: defaultdict(int) for fp in FINGERPRINTS}

    for entry in raw:
        for fp, candidates in field_candidates.items():
            val = ""
            for c in candidates:
                v = entry.get(c)
                if v:
                    val = str(v).strip().lower()
                    break
            if val:
                indices[fp][val] += 1

    print(f"[eval_dictionary] FoxIO DB loaded: {len(raw)} raw entries")
    for fp in FINGERPRINTS:
        print(f"  {fp.upper()}: {len(indices[fp])} unique fingerprint values indexed")
    return indices


# ── classify a single lookup ──────────────────────────────────────────────────

def _classify(entry_count: int | None) -> str:
    """Classify a lookup result.

    entry_count = number of rows in the FoxIO DB that share this fingerprint.
    0  → Ukjent   (fingerprint not found at all)
    1  → Unik match (only one DB row → unambiguous)
    >1 → Kollisjon  (multiple rows → ambiguous)
    """
    if not entry_count:
        return "Ukjent"
    if entry_count == 1:
        return "Unik match"
    return "Kollisjon"


# ── run evaluation ────────────────────────────────────────────────────────────

def run() -> dict[str, dict[str, int]]:
    indices = _load_foxio_indices()

    _, test_records = database_lookup.train_test_split(seed=42, test_ratio=0.2)
    print(f"[eval_dictionary] Evaluating {len(test_records)} test records across {len(FINGERPRINTS)} fingerprint types...\n")

    # counts[fingerprint][outcome] = count
    counts: dict[str, dict[str, int]] = {
        fp: {"Unik match": 0, "Kollisjon": 0, "Ukjent": 0}
        for fp in FINGERPRINTS
    }

    for record in test_records:
        for fp in FINGERPRINTS:
            val = (getattr(record, fp) or "").strip().lower()
            if not val:
                # Record has no value for this fingerprint → treat as Ukjent
                counts[fp]["Ukjent"] += 1
                continue
            entry_count = indices[fp].get(val, 0)
            outcome = _classify(entry_count)
            counts[fp][outcome] += 1

    # Print summary
    total = len(test_records)
    header = f"{'Fingerprint':<10}  {'Unik match':>12}  {'Kollisjon':>12}  {'Ukjent':>12}"
    print("=" * len(header))
    print("  Dictionary Evaluation — per fingerprint type".center(len(header)))
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for fp in FINGERPRINTS:
        u = counts[fp]["Unik match"]
        k = counts[fp]["Kollisjon"]
        unk = counts[fp]["Ukjent"]
        print(f"{fp.upper():<10}  {u:>5} ({u/total*100:5.1f}%)  {k:>5} ({k/total*100:5.1f}%)  {unk:>5} ({unk/total*100:5.1f}%)")
    print("=" * len(header))

    return counts


# ── plot ──────────────────────────────────────────────────────────────────────

def plot(counts: dict[str, dict[str, int]]) -> None:
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)

    labels    = [LABEL_MAP[fp] for fp in FINGERPRINTS]
    categories = ["Unik match", "Kollisjon", "Ukjent"]

    # Compute percentages
    totals = {fp: sum(counts[fp].values()) for fp in FINGERPRINTS}
    pct: dict[str, list[float]] = {
        cat: [counts[fp][cat] / totals[fp] * 100 for fp in FINGERPRINTS]
        for cat in categories
    }

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    x      = np.arange(len(FINGERPRINTS))
    width  = 0.55
    bottom = np.zeros(len(FINGERPRINTS))

    bars_list = []
    for cat in categories:
        values = np.array(pct[cat])
        bars = ax.bar(x, values, width, bottom=bottom,
                      label=cat, color=PALETTE[cat], edgecolor="#1a1a2e", linewidth=0.8)
        bars_list.append((cat, values, bottom.copy(), bars))
        bottom += values

    # Annotate segments (only if tall enough)
    for cat, values, bot, bars in bars_list:
        for i, (v, b, bar) in enumerate(zip(values, bot, bars)):
            if v >= 4:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    b + v / 2,
                    f"{v:.1f}%",
                    ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white",
                )

    # Styling
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14, color="white", fontweight="bold")
    ax.set_ylabel("Andel av testmengden (%)", fontsize=12, color="#cccccc")
    ax.set_ylim(0, 108)
    ax.set_xlim(-0.5, len(FINGERPRINTS) - 0.5)
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

    ax.set_title(
        "Ordbokoppslag per fingeravtrykktype\n(Unik match · Kollisjon · Ukjent)",
        fontsize=15, fontweight="bold", color="white", pad=18,
    )

    # Legend
    patches = [mpatches.Patch(color=PALETTE[c], label=c) for c in categories]
    ax.legend(
        handles=patches, loc="upper right",
        framealpha=0.25, facecolor="#0f3460", edgecolor="#555577",
        labelcolor="white", fontsize=11,
    )

    ax.yaxis.grid(True, color="#2a2a4a", linestyle="--", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n[eval_dictionary] Chart saved -> {OUTPUT_PNG}")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    counts = run()
    plot(counts)
