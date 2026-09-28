""" ROC curves per architecture from the logged P_Malignant values. """
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_DIR = "."  # folder with the three *_triage_telemetry.csv files
OUT_DIR = "evaluation_charts"

MODELS = {
    "Baseline CNN": "baseline_cnn_triage_telemetry.csv",
    "ViT-Only (DINOv2)": "vit_only_triage_telemetry.csv",
    "Hybrid Dual-Topology": "hybrid_topology_triage_telemetry.csv",
}
COLORS = {"Baseline CNN": "#1f77b4", "ViT-Only (DINOv2)": "#2ca02c", "Hybrid Dual-Topology": "#d62728"}


# avoids a scikit-learn dependency
def roc_curve(y_true, scores):
    """ Computes ROC points and AUC.
        inputs: y_true (1 = malignant), scores (P_Malignant)
        outputs: fpr, tpr, auc
    """
    order = np.argsort(-scores, kind="mergesort")  # stable sort keeps ties in order
    y = y_true[order]
    s = scores[order]
    tps = np.cumsum(y == 1)
    fps = np.cumsum(y == 0)
    # keep the last point of each run of tied scores
    distinct = np.where(np.diff(s))[0]
    idx = np.r_[distinct, len(y) - 1]
    tpr = np.r_[0, tps[idx] / tps[-1]]
    fpr = np.r_[0, fps[idx] / fps[-1]]
    auc = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))  # trapezoidal rule
    return fpr, tpr, auc


# threshold-free view of how well each model separates the classes
def main():
    """ Plots one ROC curve per architecture and saves the figure.
        inputs: none (paths set at the top of the file)
        outputs: none (evaluation_charts/roc_curves.png)
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    for name, fname in MODELS.items():
        df = pd.read_csv(os.path.join(CSV_DIR, fname))
        fpr, tpr, auc = roc_curve(df["True_Label"].values, df["P_Malignant"].values)
        ax.plot(fpr, tpr, lw=2, color=COLORS[name], label=f"{name} (AUC = {auc * 100:.1f}%)")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")  # diagonal = random classifier
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title("ROC curves on the 247-scan held-out set")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "roc_curves.png"), dpi=300)
    plt.close(fig)
    print("Saved", os.path.join(OUT_DIR, "roc_curves.png"))


if __name__ == "__main__":
    main()
