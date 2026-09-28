""" Routing, accuracy and false-negative charts from the telemetry CSVs. """
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


# telemetry CSVs -> evaluation charts
def generate_thesis_charts():
    """ Reads the three telemetry CSVs and saves comparison charts.
        inputs: none (CSV names set below)
        outputs: none (PNGs in evaluation_charts/)
    """
    print("--- Starting Visuals ---")
    os.makedirs("evaluation_charts", exist_ok=True)

    model_files = {
        "Baseline CNN": "baseline_cnn_triage_telemetry.csv",
        "ViT-Only": "vit_only_triage_telemetry.csv",
        "Hybrid ViT+CNN": "hybrid_topology_triage_telemetry.csv",
    }

    dfs = {}
    for name, path in model_files.items():
        try:
            dfs[name] = pd.read_csv(path)
        except FileNotFoundError:
            print(f"CRITICAL: {path} missing. Run triage daemon for this model first.")
            return

    models = list(dfs.keys())

    deferral_rates, automation_rates, accuracy_rates, fn_rates = [], [], [], []

    for name in models:
        df = dfs[name]
        total = len(df)
        deferred = (df['Model_Status'] == 'DEFERRED').sum()
        def_rate = deferred / total * 100
        deferral_rates.append(def_rate)
        automation_rates.append(100 - def_rate)

        # accuracy over automated passes only
        auto = df[df['Model_Status'] == 'AUTOMATED_PASS'].copy()
        auto['Predicted_Class'] = auto['Predicted_Class'].astype(int)
        acc = (auto['True_Label'] == auto['Predicted_Class']).sum() / len(auto) * 100 if len(auto) > 0 else 0
        accuracy_rates.append(acc)

        # false negatives as % of all malignant scans
        total_malignant = (df['True_Label'] == 1).sum()
        fn = df['Is_False_Negative'].sum() if 'Is_False_Negative' in df.columns else 0
        fn_rate = fn / total_malignant * 100 if total_malignant > 0 else 0
        fn_rates.append(fn_rate)

    x = np.arange(len(models))
    width = 0.35  # bar width

    # chart 1: automated vs deferred share
    fig, ax = plt.subplots(figsize=(9, 6))
    rects1 = ax.bar(x - width/2, automation_rates, width, label='Automated Pass', color='#2ca02c')
    rects2 = ax.bar(x + width/2, deferral_rates, width, label='Deferred to Clinician', color='#d62728')
    ax.set_ylabel('Percentage of Clinical Scans (%)')
    ax.set_title('Conformal Triage Routing by Architecture')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.bar_label(rects1, fmt='%.1f%%', padding=3, fontweight='bold')
    ax.bar_label(rects2, fmt='%.1f%%', padding=3, fontweight='bold')
    plt.tight_layout()
    plt.savefig("evaluation_charts/triage_distribution.png", dpi=300)
    plt.close(fig)

    # chart 2: accuracy on automated passes
    plt.figure(figsize=(9, 6))
    bars = plt.bar(models, accuracy_rates, width=0.4, color='#1f77b4')
    plt.ylabel('Diagnostic Accuracy (%)')
    plt.title('Automated Classification Accuracy by Architecture')
    plt.ylim(0, 110)  # headroom for value labels
    for i, p in enumerate(bars):
        plt.text(p.get_x() + p.get_width()/2., p.get_height() + 2,
                  f"{accuracy_rates[i]:.1f}%", ha='center', va='bottom', color='black', fontweight='bold')
    plt.tight_layout()
    plt.savefig("evaluation_charts/automated_accuracy.png", dpi=300)
    plt.close()

    # chart 3: malignant scans missed by automation
    plt.figure(figsize=(9, 6))
    bars = plt.bar(models, fn_rates, width=0.4, color='#d62728')
    plt.ylabel('False Negatives (% of all malignant cases)')
    plt.title('Malignant Cases Missed by Automation, by Architecture')
    plt.ylim(0, max(fn_rates + [5]) * 1.5)  # min 5% axis so zero bars stay visible
    for i, p in enumerate(bars):
        plt.text(p.get_x() + p.get_width()/2., p.get_height() + 0.1,
                  f"{fn_rates[i]:.1f}%", ha='center', va='bottom', color='black', fontweight='bold')
    plt.tight_layout()
    plt.savefig("evaluation_charts/false_negative_rate.png", dpi=300)
    plt.close()

    print("Charts generated in /evaluation_charts.")

if __name__ == "__main__":
    generate_thesis_charts()
