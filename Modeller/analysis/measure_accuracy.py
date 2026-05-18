import pandas as pd
import ja4_parser
import database_lookup
import eval_random_forest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "prototype"))

def measure_accuracy():
    print("[accuracy_script] Loading data...")
    train_recs, test_recs = database_lookup.train_test_split(seed=42)
    
    print("[accuracy_script] Loading model...")
    bundle = eval_random_forest.load_model(eval_random_forest.MODEL_PATH)
    
    def get_acc(recs, name):
        correct = 0
        total = len(recs)
        for i, r in enumerate(recs):
            pred, _, _ = eval_random_forest.predict(bundle, r)
            true_app = (r.application or "").lower()
            if pred.lower() == true_app:
                correct += 1
        acc = (correct / total * 100) if total > 0 else 0
        print(f"{name} Accuracy: {acc:.2f}% ({correct}/{total})")
        return acc

    train_acc = get_acc(train_recs, "Training")
    test_acc = get_acc(test_recs, "Test")

if __name__ == "__main__":
    measure_accuracy()
