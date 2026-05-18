"""Combined JA4+ classifier pipeline.

This is the core of the project. It runs three independent classifiers and
then combines their results into one final answer.

─────────────────────────────────────────────────────────────────────────────
HOW THE THREE CLASSIFIERS WORK
─────────────────────────────────────────────────────────────────────────────

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  1. EGENLAGD EXACT MATCH                                                │
  │     Looks up the fingerprint in our own custom database.                │
  │     Tries progressively looser combinations (ja4+ja4s+ja4t+ja4ts →     │
  │     ja4 alone). Returns "unique", "ambiguous", or "unknown".            │
  ├─────────────────────────────────────────────────────────────────────────┤
  │  2. RANDOM FOREST (RF)                                                  │
  │     A pre-trained ML model. Parses the JA4 strings into features, runs  │
  │     predict_proba(), and returns a ranked list of applications with      │
  │     confidence percentages.                                             │
  └─────────────────────────────────────────────────────────────────────────┘

PRIORITY ORDER (decide() function):
  1. Unique Egenlagd exact match → use it directly (highest trust)
  2. Ambiguous Egenlagd match + RF agrees → RF picks between candidates
  3. No local match + RF is confident enough → RF is the primary answer
  4. All else fails → try to agree at least on a category
  5. Nothing works → return "unknown"

Usage:
    python pipeline.py
    python pipeline.py --ja4 "t13d1516h2_aaa_bbb" --ja4s "t130200_1301_ccc"
"""

from __future__ import annotations
import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

from . import database_lookup
from . import ja4_parser
from .models import (
    ClassificationResult,
    DatabaseRecord,
    FinalDecision,
    Observation,
    infer_category,
)

# ── file paths ────────────────────────────────────────────────────────────────
_ROOT          = Path(__file__).resolve().parent          # pipeline_model/
MODEL_PATH     = _ROOT.parents[1] / "data" / "models" / "ja4_rf.pkl"

# ── confidence thresholds ─────────────────────────────────────────────────────
RF_ACCEPT_THRESHOLD = 0.60   # RF must be at least this confident to be trusted
RF_HIGH_THRESHOLD   = 0.75   # above this → "high" confidence band

# ── lookup plans ──────────────────────────────────────────────────────────────
_LOOKUP_PLANS: list[tuple[str, tuple[str, ...]]] = [
    ("ja4_ja4s_ja4ts",      ("ja4", "ja4s", "ja4ts")),
    ("ja4_ja4s",            ("ja4", "ja4s")),
    ("ja4_ja4ts",           ("ja4", "ja4ts")),
    ("ja4_only",            ("ja4",)),
]
# Plans that use both client and server fingerprints — higher-confidence group.
_STRICT_MODES = {"ja4_ja4s_ja4ts", "ja4_ja4s"}


# ── lazy-loaded singletons ────────────────────────────────────────────────────
# Each index / model is loaded from disk exactly once and then cached here.
_EGENLAGD_INDEX: dict | None = None   # the local DB structured for fast lookup
_RF_BUNDLE:      dict | None = None   # the pickled Random Forest + metadata


def _field(record: DatabaseRecord, name: str) -> str:
    """Read a field from a DatabaseRecord, normalised to lowercase with no spaces."""
    return (getattr(record, name) or "").strip().lower()


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 1 — EGENLAGD EXACT MATCH
# ═════════════════════════════════════════════════════════════════════════════

def _get_egenlagd_index() -> dict:
    """Build (once) the Egenlagd lookup index from the 80% training split.

    The index is a nested dict:
      index[mode][key] = [list of matching DatabaseRecord objects]

    where key = "ja4_value|ja4s_value|..." — the fingerprint values joined by '|'.
    We use the training split only so evaluation on the test split is fair.
    """
    global _EGENLAGD_INDEX
    if _EGENLAGD_INDEX is not None:
        return _EGENLAGD_INDEX

    train_records, _ = database_lookup.train_test_split(seed=42, test_ratio=0.2)

    # Pre-allocate one sub-dict per lookup plan.
    index: dict[str, dict[str, list[DatabaseRecord]]] = {
        mode: defaultdict(list) for mode, _ in _LOOKUP_PLANS
    }
    # Index every training record under every applicable plan.
    for record in train_records:
        for mode, fields in _LOOKUP_PLANS:
            values = [_field(record, f) for f in fields]
            if all(values):   # only index if all required fields are non-empty
                index[mode]["|".join(values)].append(record)

    _EGENLAGD_INDEX = index
    print("[pipeline] Egenlagd index built.")
    return _EGENLAGD_INDEX


def _egenlagd_lookup(obs: Observation) -> dict:
    """Run the Egenlagd exact-match lookup for one observation.

    Tries every lookup plan in priority order, stopping at the first hit.

    Returns a dict with:
      status       — "unique" (one app matched), "ambiguous" (multiple),
                     or "unknown" (no match at all)
      match_mode   — which lookup plan produced the hit (e.g. "ja4_ja4s")
      candidates   — list of application names, most frequent first
      top_app      — the best candidate
      top_category — category of the best candidate
      confidence   — float 0-1, base confidence assigned by the plan type
    """
    index     = _get_egenlagd_index()

    # 1. Full match: JA4 + JA4S + JA4TS
    mode = "ja4_ja4s_ja4ts"
    values = [(getattr(obs, f) or "").strip().lower() for f in ("ja4", "ja4s", "ja4ts")]
    if all(values):
        hits = index.get(mode, {}).get("|".join(values), [])
        if hits:
            return _process_egenlagd_hits(hits, mode)

    # 2a. Delvis match: JA4 + JA4S
    mode = "ja4_ja4s"
    values = [(getattr(obs, f) or "").strip().lower() for f in ("ja4", "ja4s")]
    if all(values):
        hits = index.get(mode, {}).get("|".join(values), [])
        if hits:
            return _process_egenlagd_hits(hits, mode)

    # 2b. Delvis match: JA4 + JA4TS
    mode = "ja4_ja4ts"
    values = [(getattr(obs, f) or "").strip().lower() for f in ("ja4", "ja4ts")]
    if all(values):
        hits = index.get(mode, {}).get("|".join(values), [])
        if hits:
            return _process_egenlagd_hits(hits, mode)

    # 3. Individuell: JA4 alene
    ja4_val = (obs.ja4 or "").strip().lower()
    if ja4_val:
        hits = index.get("ja4_only", {}).get(ja4_val, [])
        if hits:
            return _process_egenlagd_hits(hits, "ja4_only")

    # Ingen treff på noen nivå.
    return {
        "status": "unknown", "match_mode": None,
        "candidates": [], "top_app": None,
        "top_category": None, "confidence": 0.0,
    }

def _process_egenlagd_hits(hits: list[DatabaseRecord], mode: str) -> dict:
    app_counts: dict[str, int] = defaultdict(int)
    cat_map:    dict[str, str] = {}
    for hit in hits:
        if hit.application:
            app_counts[hit.application] += max(hit.count, 1)
            cat_map[hit.application] = hit.category or "unknown"

    if not app_counts:
        return {
            "status": "unknown", "match_mode": None,
            "candidates": [], "top_app": None,
            "top_category": None, "confidence": 0.0,
        }

    ranked = sorted(app_counts.keys(), key=lambda a: app_counts[a], reverse=True)
    status = "unique" if len(ranked) == 1 else "ambiguous"

    confidence = 0.9 if mode in _STRICT_MODES else 0.7
    if status == "ambiguous":
        confidence *= 0.6

    return {
        "status": status, "match_mode": mode,
        "candidates": ranked, "top_app": ranked[0],
        "top_category": cat_map.get(ranked[0], "unknown"),
        "confidence": round(confidence, 3),
    }


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 2 — RANDOM FOREST INFERENCE
# ═════════════════════════════════════════════════════════════════════════════

def _get_rf_bundle():
    """Load (once) the pickled RF model bundle from disk.

    The bundle is a dict saved by eval_random_forest.py with keys:
      classifier       — the trained sklearn RandomForestClassifier
      feature_columns  — ordered list of feature names the model expects
      categorical_maps — dict of {feature → {value_str → int}} label encodings
      idx_to_label     — dict of {int index → application name string}
      label_to_category — dict of {application name → category}
    """
    global _RF_BUNDLE
    if _RF_BUNDLE is not None:
        return _RF_BUNDLE
    if not MODEL_PATH.exists():
        print("[pipeline] RF model not found — RF disabled. Run eval_random_forest.py first.")
        return None
    with MODEL_PATH.open("rb") as f:
        _RF_BUNDLE = pickle.load(f)
    
    # Speed up row-by-row prediction by disabling multiprocessing inside the classifier
    if hasattr(_RF_BUNDLE.get("classifier"), "n_jobs"):
        _RF_BUNDLE["classifier"].n_jobs = 1
        
    print("[pipeline] RF model loaded.")
    return _RF_BUNDLE


def _rf_predict(obs: Observation) -> dict:
    """Run Random Forest inference for one observation.

    Steps:
      1. Parse the JA4 strings into a flat feature dict (via ja4_parser).
      2. Label-encode categorical features using the bundle's mappings.
      3. Build a one-row pandas DataFrame in the exact column order the model
         was trained on.
      4. Call predict_proba() → get a probability for every known application.
      5. Return the top-5 predictions with their probabilities.

    Returns a dict with:
      status         — "predicted" or "unavailable" (model not loaded)
      prediction     — the top-1 application name
      top_k          — list of top-5 application names
      top_k_with_prob — list of (app_name, probability_percent) tuples
      confidence     — float 0-1 (top-1 probability)
      category       — category of the top-1 prediction
    """
    bundle = _get_rf_bundle()
    if bundle is None:
        return {"status": "unavailable", "prediction": None, "top_k": [], "confidence": 0.0, "category": None}

    try:
        import pandas as pd
    except ImportError:
        return {"status": "unavailable", "prediction": None, "top_k": [], "confidence": 0.0, "category": None}

    # Step 1: parse JA4 strings into a raw feature dict.
    raw = ja4_parser.observation_to_features(obs)

    # Step 2: encode each feature the way the model expects it.
    encoded = {}
    for col in bundle["feature_columns"]:
        if col in ja4_parser.CATEGORICAL_FEATURES:
            # Categorical → look up the integer code from training-time mapping.
            # -1 means "unseen value" (the model was not trained on this string).
            mapping = bundle["categorical_maps"].get(col, {})
            encoded[col] = mapping.get(str(raw.get(col, ja4_parser.MISSING)), -1)
        else:
            # Numeric → cast to int (binary flags and counts).
            encoded[col] = int(raw.get(col, 0))

    # Step 3: build the DataFrame and run inference.
    frame = pd.DataFrame([encoded], columns=bundle["feature_columns"])
    probs     = bundle["classifier"].predict_proba(frame)[0]
    class_ids = bundle["classifier"].classes_

    # Step 4: sort by probability descending, keep top 5.
    top_pos = probs.argsort()[::-1][:5]
    top_k   = [
        (bundle["idx_to_label"][int(class_ids[i])], round(float(probs[i]) * 100, 2))
        for i in top_pos
    ]

    best_label, best_prob_pct = top_k[0]
    best_conf     = best_prob_pct / 100.0
    best_category = bundle["label_to_category"].get(best_label) or infer_category(best_label)

    return {
        "status":          "predicted",
        "prediction":      best_label,
        "top_k":           [t[0] for t in top_k],
        "confidence":      round(best_conf, 4),
        "category":        best_category,
        "top_k_with_prob": top_k,   # raw (app, %) tuples — useful for eval
    }




# ═════════════════════════════════════════════════════════════════════════════
# STAGE 4 — DECISION ENGINE
# ═════════════════════════════════════════════════════════════════════════════

def _rf_clear_enough(rf: dict) -> bool:
    """Return True if the RF result is trustworthy enough to act on.

    Currently requires RF confidence >= RF_ACCEPT_THRESHOLD (0.60).
    """
    return rf["status"] == "predicted" and rf["confidence"] >= RF_ACCEPT_THRESHOLD


def decide(local: dict, rf: dict) -> FinalDecision:
    """Combine the two classifier signals into exactly one final answer.

    Parameters
    ----------
    local   — result dict from _egenlagd_lookup()
    rf      — result dict from _rf_predict()

    Decision waterfall (first matching branch wins):
    ─────────────────────────────────────────────────
    Branch 1 — Unique Egenlagd exact match
    Branch 2 — Ambiguous Egenlagd match resolved by RF
    Branch 3 — No local match, RF is the primary signal
    Branch 4 — Category fallback
    Branch 5 — Give up
    """

    # ── Branch 1: Unique exact match ─────────────────────────────────────────
    if local["status"] == "unique":
        app = local["top_app"]
        cat = local["top_category"] or infer_category(app)
        confidence = "high" if local["match_mode"] in _STRICT_MODES else "medium"
        reasoning  = f"Unique Egenlagd exact match via {local['match_mode']}."
        return FinalDecision(
            application_prediction=app,
            category_prediction=cat,
            application_confidence=confidence,
            category_confidence="high",
            decision_source="egenlagd_exact",
            reasoning=reasoning,
        )

    # ── Branch 2: Ambiguous local match, RF resolves it ──────────────────────
    if local["status"] == "ambiguous" and local["candidates"] and _rf_clear_enough(rf):
        candidates_lower = {c.lower(): c for c in local["candidates"]}
        rf_pred = (rf["prediction"] or "").lower()
        if rf_pred in candidates_lower:
            resolved = candidates_lower[rf_pred]
            cat = rf["category"] or infer_category(resolved)
            confidence = "high" if rf["confidence"] >= RF_HIGH_THRESHOLD else "medium"
            return FinalDecision(
                application_prediction=resolved,
                category_prediction=cat,
                application_confidence=confidence,
                category_confidence="high",
                decision_source="ambiguous_resolved_by_rf",
                reasoning=(
                    f"Egenlagd returned {len(local['candidates'])} candidates via "
                    f"{local['match_mode']}. RF resolved to {resolved} with "
                    f"{rf['confidence']*100:.1f}% confidence."
                ),
            )

    # ── Branch 3: No local match — use RF as primary ──────────────────────────
    if local["status"] in ("unknown", "ambiguous") and _rf_clear_enough(rf):
        app     = rf["prediction"]
        cat     = rf["category"] or infer_category(app)
        confidence = "high" if rf["confidence"] >= RF_HIGH_THRESHOLD else "medium"
        reasoning = (
            f"No strong local match — RF predicts {app} with "
            f"{rf['confidence']*100:.1f}% confidence."
        )
        return FinalDecision(
            application_prediction=app,
            category_prediction=cat,
            application_confidence=confidence,
            category_confidence="medium",
            decision_source="random_forest",
            reasoning=reasoning,
        )

    # ── Branch 4: All local candidates share a category ─────────────────────
    if local["candidates"]:
        categories = {infer_category(c) for c in local["candidates"]} - {"unknown"}
        if len(categories) == 1:
            cat = next(iter(categories))
            # If RF has a prediction among these candidates, use it, otherwise pick the first.
            rf_pred = rf.get("prediction")
            candidates_lower = [c.lower() for c in local["candidates"]]
            app = None
            if rf_pred and rf_pred.lower() in candidates_lower:
                app = rf_pred
            else:
                app = local["candidates"][0]

            return FinalDecision(
                application_prediction=app,
                category_prediction=cat,
                application_confidence="low",
                category_confidence="high",
                decision_source="category_fallback_local",
                reasoning=(
                    f"Application ambiguous across {len(local['candidates'])} "
                    f"candidates but all share category: {cat}. Picking {app} as best guess."
                ),
            )

    # ── Branch 5: Last resort — use RF even if confidence is low ─────────────
    app = rf.get("prediction")
    cat = rf.get("category") or (infer_category(app) if app else None)
    
    return FinalDecision(
        application_prediction=app,
        category_prediction=cat,
        application_confidence="low" if app else "none",
        category_confidence="low" if cat else "none",
        decision_source="random_forest_fallback",
        reasoning="No strong evidence found. Returning Random Forest's best guess as a low-confidence fallback.",
    )


# ═════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def classify(
    ja4:        str | None,
    ja4s:       str | None = None,
    ja4ts:      str | None = None,
    ja4_string: str | None = None,
    ja4s_string: str | None = None,
    observation_id: str = "query",
) -> ClassificationResult:
    """Classify one set of JA4 fingerprints through the full pipeline."""
    # Pack all fingerprints into the typed input container.
    obs = Observation(
        observation_id=observation_id,
        ja4=ja4, ja4s=ja4s, ja4ts=ja4ts,
        ja4_string=ja4_string, ja4s_string=ja4s_string,
    )

    # Run the classifiers independently.
    local  = _egenlagd_lookup(obs)
    rf     = _rf_predict(obs)

    # Combine outputs into one final answer.
    decision = decide(local, rf)

    return ClassificationResult(
        observation_id=observation_id,
        ja4=ja4,
        true_application=None,   # unknown at query time — only set during eval
        true_category=None,
        predicted_application=decision.application_prediction,
        predicted_category=decision.category_prediction,
        is_correct=None,
        confidence=decision.application_confidence,
        decision_source=decision.decision_source,
        reasoning=decision.reasoning,
        model_details={
            "local": local,
            "rf":    rf,
        },
    )


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JA4+ combined pipeline classifier")
    parser.add_argument("--ja4",        default=None)
    parser.add_argument("--ja4s",       default=None)
    parser.add_argument("--ja4ts",      default=None)
    parser.add_argument("--ja4_string", default=None)
    parser.add_argument("--ja4s_string",default=None)
    args = parser.parse_args()

    if args.ja4:
        result = classify(
            args.ja4, args.ja4s, args.ja4ts,
            args.ja4_string, args.ja4s_string,
        )
        print(json.dumps(result.to_dict(), indent=2))
    else:
        # No arguments supplied → run a quick demo.
        _, test = database_lookup.train_test_split(seed=42)
        print("=== Pipeline demo (first 3 test records) ===\n")
        for record in test[:3]:
            result = classify(
                record.ja4, record.ja4s, record.ja4ts,
                record.ja4_string, record.ja4s_string,
            )
            print(f"True app  : {record.application}")
            print(f"Predicted : {result.predicted_application} ({result.confidence})")
            print(f"Category  : {result.predicted_category}")
            print(f"Source    : {result.decision_source}")
            print(f"Reasoning : {result.reasoning}\n")
