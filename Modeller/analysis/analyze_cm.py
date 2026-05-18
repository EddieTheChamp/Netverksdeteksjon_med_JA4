import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "prototype"))
from collections import Counter

_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = _ROOT.parent / "results"

eg = json.loads((RESULTS_DIR / 'egenlagd_results.json').read_text(encoding='utf-8'))
rf = json.loads((RESULTS_DIR / 'rf_results.json').read_text(encoding='utf-8'))

eg_acc = sum(1 for r in eg if r.get('correct')) / len(eg) * 100
rf_acc = sum(1 for r in rf if r.get('correct')) / len(rf) * 100

print("=== GRUNNLEGGENDE STATS ===")
print(f"Egenlagd DB : {len(eg)} test-records, Top-1 accuracy = {eg_acc:.1f}%")
print(f"Random Forest: {len(rf)} test-records, Top-1 accuracy = {rf_acc:.1f}%")

# Egenlagd match distribution
eg_unique = sum(1 for r in eg if r.get('matches_count',0) == 1)
eg_coll   = sum(1 for r in eg if r.get('matches_count',0) > 1)
eg_unk    = sum(1 for r in eg if r.get('matches_count',0) == 0)
total = len(eg)
print(f"\n=== EGENLAGD: oppslags-distribusjon ===")
print(f"  Unik match  : {eg_unique:4d}  ({eg_unique/total*100:.1f}%)")
print(f"  Kollisjon   : {eg_coll:4d}  ({eg_coll/total*100:.1f}%)")
print(f"  Ukjent      : {eg_unk:4d}  ({eg_unk/total*100:.1f}%)")

# Accuracy by match type
eg_unique_corr = sum(1 for r in eg if r.get('matches_count',0) == 1 and r.get('correct'))
eg_coll_corr   = sum(1 for r in eg if r.get('matches_count',0) > 1 and r.get('correct'))
print(f"\n  Acc: unik match = {eg_unique_corr}/{eg_unique} ({eg_unique_corr/eg_unique*100 if eg_unique else 0:.1f}%)")
print(f"  Acc: kollisjon  = {eg_coll_corr}/{eg_coll} ({eg_coll_corr/eg_coll*100 if eg_coll else 0:.1f}%)")

# Egenlagd: top misclassifications
print("\n=== EGENLAGD: vanligste feilklassifiseringer ===")
eg_errors = Counter()
for r in eg:
    if not r.get('correct') and r.get('prediction','Unknown') != 'Unknown':
        eg_errors[(r.get('true_app','?'), r.get('prediction','?'))] += 1
for (true_app, pred), c in eg_errors.most_common(10):
    print(f"  {true_app} -> {pred}: {c}")

# RF: top misclassifications
print("\n=== RF: vanligste feilklassifiseringer ===")
rf_errors = Counter()
for r in rf:
    if not r.get('correct') and r.get('prediction','Unknown') != 'Unknown':
        rf_errors[(r.get('true_app','?'), r.get('prediction','?'))] += 1
for (true_app, pred), c in rf_errors.most_common(10):
    print(f"  {true_app} -> {pred}: {c}")

# Per-app breakdown (top 15)
print("\n=== EGENLAGD: per-app korrekthet ===")
eg_by_app = {}
for r in eg:
    app = r.get('true_app','Unknown')
    pred = r.get('prediction','Unknown')
    correct = r.get('correct', False)
    mc = r.get('matches_count', 0)
    if app not in eg_by_app:
        eg_by_app[app] = {'correct':0,'wrong':0,'unknown':0,'collision':0,'total':0}
    eg_by_app[app]['total'] += 1
    if correct:
        eg_by_app[app]['correct'] += 1
    elif pred == 'Unknown':
        eg_by_app[app]['unknown'] += 1
    else:
        eg_by_app[app]['wrong'] += 1
    if mc > 1:
        eg_by_app[app]['collision'] += 1

print(f"  {'App':<30} {'N':>4} {'Corr':>5} {'Wrong':>6} {'Unk':>5} {'Coll':>5}")
for app, s in sorted(eg_by_app.items(), key=lambda x: x[1]['total'], reverse=True)[:15]:
    print(f"  {app:<30} {s['total']:>4} {s['correct']:>5} {s['wrong']:>6} {s['unknown']:>5} {s['collision']:>5}")

print("\n=== RF: per-app korrekthet ===")
rf_by_app = {}
for r in rf:
    app = r.get('true_app','Unknown')
    pred = r.get('prediction','Unknown')
    correct = r.get('correct', False)
    if app not in rf_by_app:
        rf_by_app[app] = {'correct':0,'wrong':0,'total':0}
    rf_by_app[app]['total'] += 1
    if correct:
        rf_by_app[app]['correct'] += 1
    else:
        rf_by_app[app]['wrong'] += 1

print(f"  {'App':<30} {'N':>4} {'Corr':>5} {'Wrong':>6} {'Acc%':>7}")
for app, s in sorted(rf_by_app.items(), key=lambda x: x[1]['total'], reverse=True)[:15]:
    acc = s['correct']/s['total']*100 if s['total'] else 0
    print(f"  {app:<30} {s['total']:>4} {s['correct']:>5} {s['wrong']:>6} {acc:>6.1f}%")
