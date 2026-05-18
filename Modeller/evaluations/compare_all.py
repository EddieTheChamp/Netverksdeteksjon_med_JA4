"""Generate Dual Confusion Matrix for Egenlagd and Random Forest."""

from __future__ import annotations
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "prototype"))

try:
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    _PLOT_OK = True
except ImportError:
    _PLOT_OK = False

import eval_egenlagd
import eval_random_forest

_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = _ROOT.parent / "results"

def _plot_dual_confusion_matrix(egenlagd_res: list[dict], rf_res: list[dict], top_n: int = 10) -> None:
    if not _PLOT_OK:
        print("[compare_all] matplotlib/sklearn missing. Skipping plot.")
        return

    # Find top classes based on true_app from both
    class_counts: dict[str, int] = {}
    for r in egenlagd_res:
        t_app = r["true_app"]
        if t_app == "Unknown": t_app = "Ukjent"
        class_counts[t_app] = class_counts.get(t_app, 0) + 1
        
    top_classes = [c for c, _ in sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]]
    if "Ukjent" not in top_classes and class_counts.get("Ukjent", 0) > 0:
        top_classes.append("Ukjent")
        
    # Sort labels alphabetically, but remove "Ukjent" and "Andre" if they exist, then append them at the end.
    base_labels = sorted(set(top_classes) - {"Ukjent", "Andre"})
    labels = base_labels
    if "Ukjent" in top_classes:
        labels.append("Ukjent")
    labels.append("Andre")
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    for ax, results, title in zip(axes, [egenlagd_res, rf_res], ["Egenlagd Datasett", "Random Forest"]):
        y_true, y_pred = [], []
        for r in results:
            t = r["true_app"]
            p = r["prediction"]
            if t == "Unknown": t = "Ukjent"
            if p == "Unknown": p = "Ukjent"
            
            t = t if t in top_classes else "Andre"
            p = p if p in top_classes else "Andre"
            
            y_true.append(t)
            y_pred.append(p)
            
        disp = ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, labels=labels, display_labels=labels,
            ax=ax, cmap="Blues", xticks_rotation="vertical",
        )
        disp.ax_.set_xlabel("Predikert")
        disp.ax_.set_ylabel("Faktisk")
        ax.set_title(f"Forvirringsmatrise: {title}", fontsize=15)
        
    plt.tight_layout()
    out = RESULTS_DIR / "confusion_matrices_dual.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"[compare_all] Saved -> {out}")


def run() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    print("\n[compare_all] Running Egenlagd evaluation...")
    egenlagd_res = eval_egenlagd.run()

    print("\n[compare_all] Running Random Forest evaluation...")
    rf_res = eval_random_forest.run()

    print("\n[compare_all] Generating dual confusion matrix...")
    _plot_dual_confusion_matrix(egenlagd_res, rf_res)


if __name__ == "__main__":
    run()
