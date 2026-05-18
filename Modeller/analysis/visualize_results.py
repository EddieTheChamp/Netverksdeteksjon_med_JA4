import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "prototype"))

_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = _ROOT.parent / "results"
IMAGES_DIR = RESULTS_DIR

def load_results(filename: str):
    path = RESULTS_DIR / filename
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def plot_accuracies():
    eg = load_results("egenlagd_results.json")
    rf = load_results("rf_results.json")
    pl = load_results("pipeline_results.json")
    
    if not eg or not rf or not pl:
        print("Missing result files. Run evaluations first.")
        return

    # Egenlagd stats
    eg_total = len(eg)
    eg_correct = sum(1 for r in eg if r.get("correct"))
    eg_acc = (eg_correct / eg_total * 100) if eg_total else 0
    
    # RF stats
    rf_total = len(rf)
    rf_correct = sum(1 for r in rf if r.get("correct"))
    rf_acc = (rf_correct / rf_total * 100) if rf_total else 0
    
    # Pipeline stats
    pl_total = len(pl)
    pl_correct = sum(1 for r in pl if r.get("correct_app"))
    pl_acc = (pl_correct / pl_total * 100) if pl_total else 0

    labels = ['Egenlagd DB', 'Random Forest', 'Komplett Pipeline']
    accuracies = [eg_acc, rf_acc, pl_acc]
    colors = ['#3498db', '#e74c3c', '#2ecc71']

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, accuracies, color=colors)
    plt.ylim(0, 100)
    plt.ylabel('Topp-1 NÃ¸yaktighet (%)', fontsize=12)
    plt.title('Sammenligning av Applikasjons-nÃ¸yaktighet', fontsize=14, pad=15)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}%", ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "accuracy_comparison.png", dpi=300)
    plt.close()
    print("Saved accuracy_comparison.png")

def plot_pipeline_sources():
    pl = load_results("pipeline_results.json")
    if not pl:
        return

    source_map = {
        "egenlagd_exact":           "Steg 1: Eksakt DB-treff",
        "ambiguous_resolved_by_rf": "Steg 2: Random Forest",
        "random_forest":            "Steg 2: Random Forest",
        "category_fallback_local":  "Steg 3: Kategori",
        "random_forest_fallback":   "Steg 4: RF Fallback",
        "unknown":                  "Steg 4: RF Fallback",
    }

    sources = {}
    for r in pl:
        raw_src = r.get("source", "unknown")
        src = source_map.get(raw_src, raw_src)
        sources[src] = sources.get(src, 0) + 1

    # Fast rekkefÃ¸lge og farger per steg
    step_order  = [
        "Steg 1: Eksakt DB-treff",
        "Steg 2: Random Forest",
        "Steg 3: Kategori",
        "Steg 4: RF Fallback",
    ]
    step_colors = ["#3498db", "#e67e22", "#9b59b6", "#e74c3c"]

    labels = [s for s in step_order if s in sources]
    sizes  = [sources[s] for s in labels]
    colors = [step_colors[step_order.index(s)] for s in labels]

    plt.figure(figsize=(10, 7))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
    plt.title('Beslutningskilder i Prototypen', fontsize=14, pad=15)
    plt.axis('equal')

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "pipeline_sources.png", dpi=300)
    plt.close()
    print("Saved pipeline_sources.png")

def plot_egenlagd_collisions():
    eg = load_results("egenlagd_results.json")
    if not eg:
        return
        
    unique = 0
    collisions = 0
    unknown = 0
    
    for r in eg:
        mc = r.get("matches_count", 0)
        if mc == 0:
            unknown += 1
        elif mc == 1:
            unique += 1
        else:
            collisions += 1
            
    labels = ['Unikt treff (1 app)', 'Kollisjon (>1 app)', 'Ukjent (0 apper)']
    sizes = [unique, collisions, unknown]
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    
    plt.figure(figsize=(9, 6))
    bars = plt.bar(labels, sizes, color=colors)
    plt.ylabel('Antall observasjoner', fontsize=12)
    plt.title('Status for Databaseoppslag (Egenlagd)', fontsize=14, pad=15)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 2, str(int(yval)), ha='center', va='bottom', fontsize=11)
        
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "egenlagd_collisions.png", dpi=300)
    plt.close()
    print("Saved egenlagd_collisions.png")

def plot_comparative_collisions():
    eg = load_results("egenlagd_results.json")
    rf = load_results("rf_results.json")
    pl = load_results("pipeline_results.json")
    
    if not eg or not rf or not pl:
        return
        
    def get_counts(results):
        unique, collision, unknown = 0, 0, 0
        total = len(results)
        for r in results:
            mc = r.get("matches_count", 0)
            if "source" in r:
                mc = 1 if r["source"] in ("egenlagd_exact", "ambiguous_resolved_by_rf", "random_forest") else 0
            if mc == 0:
                unknown += 1
            elif mc == 1:
                unique += 1
            else:
                collision += 1
        return (unique/total*100, collision/total*100, unknown/total*100) if total else (0,0,0)

    eg_u, eg_c, eg_un = get_counts(eg)
    rf_u, rf_c, rf_un = get_counts(rf)
    pl_u, pl_c, pl_un = get_counts(pl)

    import numpy as np
    names = ['Egenlagd', 'Random Forest', 'Komplett Pipeline']
    unique = np.array([eg_u, rf_u, pl_u])
    collision = np.array([eg_c, rf_c, pl_c])
    unknown = np.array([eg_un, rf_un, pl_un])

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(names, unique,    label="Unikt treff",       color="#2ca02c")
    ax.bar(names, collision, label="Kollisjon (>1 app)", color="#ff7f0e", bottom=unique)
    ax.bar(names, unknown,   label="Ukjent / Tvetydig", color="#d62728", bottom=unique + collision)

    for i, (u, col) in enumerate(zip(unique, collision)):
        if u > 3:
            ax.text(i, u / 2, f"{u:.1f}%", ha="center", va="center", color="white", fontweight="bold")
        if col > 3:
            ax.text(i, u + col / 2, f"{col:.1f}%", ha="center", va="center", color="white", fontweight="bold")

    ax.set_ylabel('Andel av test-spÃ¸rringer (%)', fontsize=12)
    ax.set_title('OpplÃ¸sning av kollisjoner: DB vs RF vs Pipeline', fontsize=13)
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "comparative_collision_matrix.png", dpi=120)
    plt.close()
    print("Saved comparative_collision_matrix.png")

def plot_top_k_accuracy():
    eg = load_results("egenlagd_results.json")
    rf = load_results("rf_results.json")
    
    if not eg or not rf:
        return
        
    def get_top_k(results):
        total = len(results)
        top1, top3, top5 = 0, 0, 0
        for r in results:
            true_app = (r.get("true_app") or "").lower()
            top_k = [str(x).lower() for x in r.get("top_k", [])]
            if top_k and len(top_k) > 0 and true_app == top_k[0]:
                top1 += 1
            if true_app in top_k[:3]:
                top3 += 1
            if true_app in top_k[:5]:
                top5 += 1
        return (top1/total*100, top3/total*100, top5/total*100) if total else (0,0,0)

    eg_metrics = get_top_k(eg)
    rf_metrics = get_top_k(rf)

    import numpy as np
    labels = ["Topp-1", "Topp-3", "Topp-5"]
    x = np.arange(len(labels))
    width = 0.30

    fig, ax = plt.subplots(figsize=(10, 6))
    
    rects1 = ax.bar(x - width/2, [eg_metrics[0], eg_metrics[1], eg_metrics[2]], width, label='Egenlagd DB', color='#3498db')
    rects2 = ax.bar(x + width/2, [rf_metrics[0], rf_metrics[1], rf_metrics[2]], width, label='Random Forest', color='#e74c3c')

    ax.set_ylabel('NÃ¸yaktighet (%)', fontsize=12)
    ax.set_title('Top-K NÃ¸yaktighet Sammenligning', fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 115)

    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "top_k_accuracy.png", dpi=300)
    plt.close()
    print("Saved top_k_accuracy.png")

def plot_fp_vs_abstentions():
    eg = load_results("egenlagd_results.json")
    rf = load_results("rf_results.json")
    pl = load_results("pipeline_results.json")
    
    if not eg or not rf or not pl:
        return
        
    def get_stats(results, model_type):
        total = len(results)
        correct = 0
        abstention = 0
        incorrect = 0
        for r in results:
            if model_type == "pipeline":
                is_correct = r.get("correct_app", False)
                pred = r.get("pred_app", "Unknown")
            else:
                is_correct = r.get("correct", False)
                pred = r.get("prediction", "Unknown")
                
            if is_correct:
                correct += 1
            elif pred == "Unknown":
                abstention += 1
            else:
                incorrect += 1
        return (correct/total*100, incorrect/total*100, abstention/total*100) if total else (0,0,0)

    eg_c, eg_i, eg_a = get_stats(eg, "egenlagd")
    rf_c, rf_i, rf_a = get_stats(rf, "rf")
    pl_c, pl_i, pl_a = get_stats(pl, "pipeline")

    import numpy as np
    labels = ['Egenlagd DB', 'Random Forest', 'Komplett Pipeline']
    correct = np.array([eg_c, rf_c, pl_c])
    incorrect = np.array([eg_i, rf_i, pl_i])
    abstain = np.array([eg_a, rf_a, pl_a])

    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(labels, correct, label='Riktig gjettet (True Positive)', color='#2ecc71')
    ax.bar(labels, incorrect, label='Feil gjettet (False Positive)', color='#e74c3c', bottom=correct)
    ax.bar(labels, abstain, label='Avstod fra Ã¥ gjette (Unknown)', color='#95a5a6', bottom=correct + incorrect)

    for i, (c, inc, a) in enumerate(zip(correct, incorrect, abstain)):
        if c > 5:
            ax.text(i, c/2, f"{c:.1f}%", ha='center', va='center', color='white', fontweight='bold')
        if inc > 5:
            ax.text(i, c + inc/2, f"{inc:.1f}%", ha='center', va='center', color='white', fontweight='bold')
        if a > 5:
            ax.text(i, c + inc + a/2, f"{a:.1f}%", ha='center', va='center', color='white', fontweight='bold')

    ax.set_ylabel('Prosentandel av testsettet (%)', fontsize=12)
    ax.set_title('Feilanalyse: Riktige vs Falske Positive vs AvstÃ¥tte', fontsize=14, pad=15)
    ax.legend(loc='lower left')
    
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "fp_vs_abstentions.png", dpi=300)
    plt.close()
    print("Saved fp_vs_abstentions.png")

def plot_db_vs_rf_accuracy():
    eg = load_results("egenlagd_results.json")
    rf = load_results("rf_results.json")
    if not eg or not rf: return
    
    eg_acc = sum(1 for r in eg if r.get("correct")) / len(eg) * 100
    rf_acc = sum(1 for r in rf if r.get("correct")) / len(rf) * 100
    
    plt.figure(figsize=(7, 5))
    bars = plt.bar(['Egenlagd DB', 'Random Forest'], [eg_acc, rf_acc], color=['#3498db', '#e74c3c'])
    plt.ylim(0, 100)
    plt.ylabel('Topp-1 NÃ¸yaktighet (%)')
    plt.title('NÃ¸yaktighet: Database vs Random Forest')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "db_vs_rf.png", dpi=300)
    plt.close()
    print("Saved db_vs_rf.png")

def plot_pipeline_accuracy_only():
    pl = load_results("pipeline_results.json")
    if not pl: return
    
    app_acc = sum(1 for r in pl if r.get("correct_app")) / len(pl) * 100
    cat_acc = sum(1 for r in pl if r.get("correct_cat")) / len(pl) * 100
    
    plt.figure(figsize=(7, 5))
    bars = plt.bar(['Applikasjon', 'Kategori'], [app_acc, cat_acc], color=['#2ecc71', '#9b59b6'])
    plt.ylim(0, 100)
    plt.ylabel('NÃ¸yaktighet (%)')
    plt.title('Komplett Pipeline: NÃ¸yaktighet')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "pipeline_only.png", dpi=300)
    plt.close()
    print("Saved pipeline_only.png")

def plot_pipeline_confidence_analysis():
    pl = load_results("pipeline_results.json")
    if not pl: return
    
    # Define groups
    levels = ["HÃ¸y", "Medium", "Lav"]
    conf_map = {"high": "HÃ¸y", "medium": "Medium", "low": "Lav", "none": "Lav"}
    
    stats = {l: {"correct": 0, "incorrect": 0, "total": 0} for l in levels}
    
    for r in pl:
        l = conf_map.get(r.get("confidence", "none"), "Lav")
        if l not in stats: continue
        stats[l]["total"] += 1
        
        if r.get("correct_app", False):
            stats[l]["correct"] += 1
        else:
            # Everything that is not correct is now counted as 'incorrect' (Feil)
            stats[l]["incorrect"] += 1
            
    import numpy as np
    correct = np.array([stats[l]["correct"] for l in levels])
    incorrect = np.array([stats[l]["incorrect"] for l in levels])
    totals = np.array([stats[l]["total"] for l in levels])
    
    # Normalize to 100%
    norm_correct = np.array([(c/t*100 if t > 0 else 0) for c, t in zip(correct, totals)])
    norm_incorrect = np.array([(i/t*100 if t > 0 else 0) for i, t in zip(incorrect, totals)])
    
    plt.figure(figsize=(10, 7))
    
    p1 = plt.bar(levels, norm_correct, label='Riktig', color='#2ecc71')
    p2 = plt.bar(levels, norm_incorrect, bottom=norm_correct, label='Feil', color='#e74c3c')
    
    plt.ylabel('Andel av svar (%)', fontsize=12)
    plt.ylim(0, 115) 
    plt.title('Pipelinen: Kvalitetsfordeling per konfidensnivÃ¥', fontsize=14, pad=15)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    
    for i in range(len(levels)):
        t = totals[i]
        if t == 0: continue
        
        if norm_correct[i] > 5:
            plt.text(i, norm_correct[i]/2, f"{norm_correct[i]:.1f}%", ha='center', va='center', color='white', fontweight='bold')
        if norm_incorrect[i] > 5:
            plt.text(i, norm_correct[i] + norm_incorrect[i]/2, f"{norm_incorrect[i]:.1f}%", ha='center', va='center', color='white', fontweight='bold')
            
        plt.text(i, 102, f"n={t}", ha='center', va='bottom', fontsize=12, fontweight='bold', color='#2c3e50')

        plt.text(i, 102, f"n={t}", ha='center', va='bottom', fontsize=12, fontweight='bold', color='#2c3e50')

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(IMAGES_DIR / "pipeline_confidence.png", dpi=300)
    plt.close()
    print("Saved pipeline_confidence.png")

def plot_confusion_matrices():
    from sklearn.metrics import confusion_matrix
    
    eg = load_results("egenlagd_results.json")
    rf = load_results("rf_results.json")
    pl = load_results("pipeline_results.json")

    if not eg or not rf or not pl:
        return

    from collections import Counter
    true_counts = Counter()
    for r in pl:
        true_counts[r.get("true_app") or "Unknown"] += 1
        
    top_15 = [app for app, count in true_counts.most_common(15)]
    labels = sorted(top_15) + ["Andre"]
    
    def extract_true_pred(results, is_pipeline):
        y_true = []
        y_pred = []
        for r in results:
            t = r.get("true_app") or "Unknown"
            p = r.get("pred_app") if is_pipeline else r.get("prediction")
            p = p or "Unknown"
            
            # Map everything outside top 15 to 'Andre'
            t_mapped = t if t in top_15 else "Andre"
            p_mapped = p if p in top_15 else "Andre"
            
            y_true.append(t_mapped)
            y_pred.append(p_mapped)
        return y_true, y_pred

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    models = [
        ("Egenlagd Database", eg, False),
        ("Random Forest", rf, False)
    ]

    for ax, (title, results, is_pipeline) in zip(axes, models):
        y_true, y_pred = extract_true_pred(results, is_pipeline)
        
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        # Using a logarithmic color scale can help highlight rare misclassifications
        import matplotlib.colors as mcolors
        norm = mcolors.LogNorm(vmin=1, vmax=cm.max() + 1)
        # Adding 1 to cm to avoid log(0)
        cm_to_plot = cm + 1
        
        sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", 
                    xticklabels=labels, yticklabels=labels, ax=ax,
                    annot_kws={"size": 11, "weight": "bold"})
        ax.set_title(title, fontsize=16, pad=15)
        ax.set_xlabel('Predikert Applikasjon', fontsize=12)
        ax.set_ylabel('Faktisk Applikasjon', fontsize=12)
        
        ax.tick_params(axis='x', rotation=90, labelsize=9)
        ax.tick_params(axis='y', rotation=0, labelsize=9)

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "confusion_matrices.png", dpi=300)
    plt.close()
    print("Saved confusion_matrices.png")

def plot_app_distribution():
    db_path = _ROOT.parent.parent / "Datasett" / "categorized_custom_db.json"
    if not db_path.exists():
        print("Could not find full database.")
        return

    import json
    from collections import Counter
    import numpy as np
    
    with db_path.open("r", encoding="utf-8") as f:
        db_data = json.load(f)
    
    true_counts = Counter()
    for r in db_data:
        app = r.get("application") or "Unknown"
        true_counts[app] += 1
        
    top_15 = true_counts.most_common(15)
    top_15_apps = [app for app, count in top_15]
    
    # Beregn "Andre"
    other_count = sum(count for app, count in true_counts.items() if app not in top_15_apps)
    
    # KlargjÃ¸r data for plotting
    labels = [app for app, count in top_15] + ["Andre"]
    values = [count for app, count in top_15] + [other_count]
    
    plt.figure(figsize=(14, 6))
    
    # Sett en litt mer nÃ¸ytral farge pÃ¥ "Andre"
    colors = sns.color_palette("viridis", 15) + [(0.7, 0.7, 0.7)] 
    
    bars = plt.bar(labels, values, color=colors)
    
    plt.ylabel('Antall observasjoner', fontsize=12)
    plt.title('Datagrunnlag: Fordeling av Applikasjoner i Hele Datasettet', fontsize=14, pad=15)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    
    # Legg til tall pÃ¥ toppen av hver stolpe
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max(values)*0.01), int(yval), 
                 ha='center', va='bottom', fontweight='bold', fontsize=10)
        
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "app_distribution.png", dpi=300)
    plt.close()
    print("Saved app_distribution.png")

def plot_foxio_match_rates():
    fox = load_results("foxio_results.json")
    if not fox: return
    
    types = ["ja4", "ja4s", "ja4ts"]
    categories = ["Ingen Treff", "Unik Match", "Kollisjon"]
    colors = ["#95a5a6", "#2ecc71", "#e74c3c"] # Grey, Green, Red
    
    data = {cat: [] for cat in categories}
    for t in types:
        total = sum(fox[t].values())
        for cat in categories:
            val = fox[t].get(cat, 0)
            data[cat].append(val / total * 100 if total > 0 else 0)
            
    plt.figure(figsize=(10, 6))
    bottom = np.zeros(len(types))
    
    for i, cat in enumerate(categories):
        plt.bar([t.upper() for t in types], data[cat], bottom=bottom, label=cat, color=colors[i])
        bottom += data[cat]
        
    plt.ylabel("Prosent av testsett (%)")
    plt.title("SÃ¸k i ekstern database (FoxIO) per felt")
    plt.ylim(0, 110)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add n=X on top
    for i, t in enumerate(types):
        total = sum(fox[t].values())
        plt.text(i, 102, f"n={total}", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "foxio_matches.png", dpi=300)
    plt.close()
    print("Saved foxio_matches.png")

if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)
    IMAGES_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    plot_accuracies()
    plot_pipeline_sources()
    plot_egenlagd_collisions()
    plot_comparative_collisions()
    plot_top_k_accuracy()
    plot_fp_vs_abstentions()
    plot_db_vs_rf_accuracy()
    plot_pipeline_accuracy_only()
    plot_pipeline_confidence_analysis()
    plot_confusion_matrices()
    plot_app_distribution()
    plot_foxio_match_rates()
    print("All visualizations completed successfully.")
