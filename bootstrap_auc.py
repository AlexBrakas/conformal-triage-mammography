""" Bootstrap confidence intervals for ROC-AUC and paired AUC differences. """
import numpy as np
import pandas as pd
from generate_roc import roc_curve, MODELS, CSV_DIR

N_RESAMPLES = 2000  # bootstrap resamples reported in the evaluation
SEED = 0    #setting a random seed of 0 for now       

PAIRS = [
    ("Hybrid Dual-Topology", "Baseline CNN"),
    ("ViT-Only (DINOv2)", "Baseline CNN"),
    ("ViT-Only (DINOv2)", "Hybrid Dual-Topology"),
]


# loads each model's logged probabilities for the same held-out scans
def load_scores():
    """ Reads the telemetry CSVs and checks that they describe the same scans.
        inputs: none (paths from generate_roc.py)
        outputs: y_true (array), {model name: P_Malignant array}
    """
    frames = {name: pd.read_csv(f"{CSV_DIR}/{fname}") for name, fname in MODELS.items()}
    first = next(iter(frames.values()))
    for df in frames.values():
        # paired comparison is only valid if rows line up scan-for-scan
        assert (df["Scan_ID"] == first["Scan_ID"]).all()
        assert (df["True_Label"] == first["True_Label"]).all()
    y_true = first["True_Label"].values
    scores = {name: df["P_Malignant"].values for name, df in frames.items()}
    return y_true, scores


# resampling scans shows how much each AUC depends on this particular held-out set
def bootstrap(y_true, scores):
    """ Paired bootstrap: every model is scored on the same resampled scans.
        inputs: y_true, {model name: scores}
        outputs: {model name: array of resampled AUCs}
    """
    rng = np.random.default_rng(SEED)
    n = len(y_true)
    aucs = {name: [] for name in scores}
    while len(next(iter(aucs.values()))) < N_RESAMPLES:
        idx = rng.integers(0, n, n)
        if y_true[idx].min() == y_true[idx].max():
            continue  # AUC undefined when a resample holds only one class
        for name, s in scores.items():
            aucs[name].append(roc_curve(y_true[idx], s[idx])[2])
    return {name: np.array(v) for name, v in aucs.items()}


# point estimates are observed values; the bootstrap supplies only the intervals
def main():
    """ Prints each model's AUC and every pairwise difference with 95% percentile CIs.
        inputs: none
        outputs: none (printed)
    """
    y_true, scores = load_scores()
    observed = {name: roc_curve(y_true, s)[2] for name, s in scores.items()}
    boot = bootstrap(y_true, scores)

    print(f"ROC-AUC, {N_RESAMPLES} paired bootstrap resamples (seed {SEED})")
    for name in scores:
        lo, hi = np.percentile(boot[name], [2.5, 97.5])
        print(f"  {name}: {observed[name] * 100:.1f}% [{lo * 100:.1f}, {hi * 100:.1f}]")

    print("Paired differences (A - B)")
    for a, b in PAIRS:
        diff = boot[a] - boot[b]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        print(f"  {a} - {b}: {(observed[a] - observed[b]) * 100:+.1f} "
              f"[{lo * 100:+.1f}, {hi * 100:+.1f}]")


if __name__ == "__main__":
    main()
