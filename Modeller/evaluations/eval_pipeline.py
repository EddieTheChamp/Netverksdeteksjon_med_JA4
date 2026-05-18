"""Evaluate the combined JA4+ pipeline performance.

Runs the full pipeline (Egenlagd + RF + FoxIO) against the 20% test split.
Saves results to results/pipeline_results.json.

Usage:
    python eval_pipeline.py
"""

from __future__ import annotations
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "prototype"))

from pipeline_model import database_lookup
from pipeline_model.pipeline import classify

_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = _ROOT.parent / "results"
OUTPUT_FILE = RESULTS_DIR / "pipeline_results.json"


def run() -> list[dict]:
    """Run the pipeline evaluation on the 20% test split."""
    _, test_records = database_lookup.train_test_split(seed=42, test_ratio=0.2)
    print(f"[eval_pipeline] Evaluating {len(test_records)} test records through the full pipeline...")

    results = []
    correct = 0
    app_hits = 0
    cat_hits = 0

    for record in test_records:
        true_app = record.application or "Unknown"
        true_cat = record.category or "unknown"
        
        res = classify(
            ja4=record.ja4,
            ja4s=record.ja4s,
            ja4ts=record.ja4ts,
            ja4_string=record.ja4_string,
            ja4s_string=record.ja4s_string,
        )
        
        pred_app = res.predicted_application or "Unknown"
        pred_cat = res.predicted_category or "unknown"
        
        # Build Top-K for comparison
        # 1. Start with the final decision
        top_k = [pred_app]
        
        # 2. Add RF top_k if it was used or available
        rf_top_k = res.model_details.get("rf", {}).get("top_k", [])
        for app in rf_top_k:
            if app not in top_k:
                top_k.append(app)
        
        # 3. Add Egenlagd candidates if available
        eg_top_k = res.model_details.get("local", {}).get("candidates", [])
        for app in eg_top_k:
            if app not in top_k:
                top_k.append(app)

        is_app_correct = pred_app.lower() == true_app.lower()
        is_cat_correct = pred_cat.lower() == true_cat.lower()
        
        if is_app_correct:
            app_hits += 1
        if is_cat_correct:
            cat_hits += 1
            
        results.append({
            "true_app": true_app,
            "true_cat": true_cat,
            "pred_app": pred_app,
            "pred_cat": pred_cat,
            "top_k": top_k[:5],
            "confidence": res.confidence,
            "source": res.decision_source,
            "reasoning": res.reasoning,
            "correct_app": is_app_correct,
            "correct_cat": is_cat_correct,
            "ja4": record.ja4,
        })

    total = len(results)
    app_acc = app_hits / total * 100 if total else 0
    cat_acc = cat_hits / total * 100 if total else 0
    
    print(f"\n[eval_pipeline] Final Pipeline Results:")
    print(f"  Total records    : {total}")
    print(f"  Application Acc  : {app_acc:.1f}% ({app_hits}/{total})")
    print(f"  Category Acc     : {cat_acc:.1f}% ({cat_hits}/{total})")
    
    # Decision source breakdown
    sources = {}
    for r in results:
        src = r["source"]
        sources[src] = sources.get(src, 0) + 1
        
    print("\n  Decision Source Breakdown:")
    for src, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
        print(f"    {src:<25}: {count} ({count/total*100:.1f}%)")

    return results


if __name__ == "__main__":
    res = run()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(res, f, indent=4)
    print(f"[eval_pipeline] Results saved to {OUTPUT_FILE}")
