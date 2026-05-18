"""Iteration 3: Train and evaluate a Random Forest model on JA4 features.

Uses the same 80/20 split (seed=42) as eval_egenlagd.py so all three
evaluations are directly comparable.

Usage:
    python eval_random_forest.py
"""

from __future__ import annotations
import json
import pickle
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "prototype"))

try:
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    _ML_OK = True
except ImportError as _e:
    _ML_OK = False
    _ML_ERROR = str(_e)

from pipeline_model import database_lookup
from pipeline_model import ja4_parser

# ── paths ─────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent  # Modeller/evaluations
MODEL_PATH = _ROOT.parent / "data" / "models" / "ja4_rf.pkl"
RESULTS_DIR = _ROOT.parent / "results"
OUTPUT_FILE = RESULTS_DIR / "rf_results.json"


def _ensure_ml() -> None:
    if not _ML_OK:
        raise ImportError(f"ML dependencies missing: {_ML_ERROR}\nInstall with: pip install scikit-learn pandas numpy")


# ── training ──────────────────────────────────────────────────────────────────

def train(train_records, *, n_estimators: int = 200, random_state: int = 42):
    """Train a Random Forest model on training records. Returns the model bundle."""
    _ensure_ml()
    samples = ja4_parser.records_to_training_samples(train_records)
    if not samples:
        raise ValueError("No labeled training samples found.")

    frame = pd.DataFrame(samples)
    categorical_maps: dict[str, dict[str, int]] = {}

    for col in ja4_parser.CATEGORICAL_FEATURES:
        cats = sorted(frame[col].astype(str).unique().tolist())
        categorical_maps[col] = {v: i for i, v in enumerate(cats)}
        frame[col] = frame[col].astype(str).map(categorical_maps[col]).astype(int)

    for col in ja4_parser.NUMERIC_FEATURES:
        if col in frame.columns:
            frame[col] = frame[col].fillna(0).astype(int)

    base_ignore = {"label", "category", "weight", "target"}
    static_features = set(ja4_parser.CATEGORICAL_FEATURES) | set(ja4_parser.NUMERIC_FEATURES)
    dynamic_cols = [c for c in frame.columns if c not in static_features and c not in base_ignore]

    for col in dynamic_cols:
        frame[col] = frame[col].fillna(0).astype(int)
        
    final_feature_columns = ja4_parser.CATEGORICAL_FEATURES + ja4_parser.NUMERIC_FEATURES + dynamic_cols

    labels = sorted(frame["label"].astype(str).unique().tolist())
    label_to_idx = {lb: i for i, lb in enumerate(labels)}
    idx_to_label = {i: lb for lb, i in label_to_idx.items()}
    frame["target"] = frame["label"].astype(str).map(label_to_idx).astype(int)

    label_to_cat: dict[str, str | None] = {}
    for _, row in frame[["label", "category"]].drop_duplicates().iterrows():
        label_to_cat[str(row["label"])] = row["category"] if pd.notna(row["category"]) else None

    clf = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=5, random_state=random_state, n_jobs=-1)
    clf.fit(frame[final_feature_columns], frame["target"], sample_weight=frame["weight"])

    bundle = {
        "classifier": clf,
        "categorical_maps": categorical_maps,
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "label_to_category": label_to_cat,
        "feature_columns": final_feature_columns,
    }
    print(f"[eval_rf] Trained RF on {len(samples)} samples, {len(labels)} classes")
    return bundle


def save_model(bundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(bundle, f)
    print(f"[eval_rf] Model saved -> {path}")


def load_model(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def predict(bundle, record) -> tuple[str, list[str], int]:
    """Predict application for one record. Returns (prediction, top_k_apps, matches_count)."""
    from pipeline_model.models import Observation
    obs = Observation(
        observation_id="eval",
        ja4=record.ja4, ja4s=record.ja4s,
        ja4_string=record.ja4_string, ja4s_string=record.ja4s_string,
        ja4ts=record.ja4ts,
    )
    raw = ja4_parser.observation_to_features(obs)
    encoded = {}
    
    for col in bundle["feature_columns"]:
        if col in ja4_parser.CATEGORICAL_FEATURES:
            mapping = bundle["categorical_maps"].get(col, {})
            encoded[col] = mapping.get(str(raw.get(col, ja4_parser.MISSING)), -1)
        else:
            encoded[col] = int(raw.get(col, 0))

    frame = pd.DataFrame([encoded], columns=bundle["feature_columns"])
    probs = bundle["classifier"].predict_proba(frame)[0]
    class_ids = bundle["classifier"].classes_

    top_positions = probs.argsort()[::-1][:5]
    top_k = [bundle["idx_to_label"][int(class_ids[i])] for i in top_positions]
    top_prob = float(probs[top_positions[0]])
    return top_k[0], top_k, 1 if top_prob >= 0.5 else len(top_k)


# ── main evaluation ───────────────────────────────────────────────────────────

def run() -> list[dict]:
    """Run the RF evaluation. Returns result records."""
    _ensure_ml()
    train_records, test_records = database_lookup.train_test_split(seed=42, test_ratio=0.2)

    # Always train a new model to ensure new feature set is used
    print(f"[eval_rf] Training new model on {len(train_records)} records...")
    bundle = train(train_records)
    save_model(bundle, MODEL_PATH)

    print(f"[eval_rf] Evaluating {len(test_records)} test records...")
    results = []
    correct = 0
    for record in test_records:
        true_app = record.application or "Unknown"
        prediction, top_k, matches_count = predict(bundle, record)
        is_correct = prediction.lower() == true_app.lower()
        if is_correct:
            correct += 1
        results.append({
            "true_app": true_app,
            "prediction": prediction,
            "top_k": top_k,
            "matches_count": matches_count,
            "correct": is_correct,
            "ja4": record.ja4,
        })

    total = len(results)
    acc = correct / total * 100 if total else 0
    print(f"\n[eval_rf] Results: {correct}/{total} correct — Top-1 Accuracy: {acc:.1f}%")

    # Feature importances
    clf = bundle["classifier"]
    importances = dict(zip(bundle["feature_columns"], clf.feature_importances_))
    top5 = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\n  Top 5 feature importances:")
    for feat, imp in top5:
        print(f"    {feat:<20} {imp:.4f}")

    return results


if __name__ == "__main__":
    res = run()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(res, f, indent=4)
    print(f"[eval_rf] Results saved to {OUTPUT_FILE}")
