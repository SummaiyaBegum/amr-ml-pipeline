r"""
src/evaluate.py

Post-training analysis:
  1. Renders a per-antibiotic AUROC bar chart -> results/figures/auroc_by_antibiotic.png
  2. Renders top-biomarker heatmap -> results/figures/top_biomarkers.png
  3. (If ground_truth.json exists) measures how well the model recovered the
     synthetic "true driver" genes -> results/ground_truth_recovery.json

Run after src/train.py:
    py -3 -I src/evaluate.py
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Defaults; overridden by CLI in main()
RESULTS = Path("results")
FIG = RESULTS / "figures"
DATA = Path("data/processed")


def plot_auroc(metrics):
    abx = list(metrics.keys())
    rf = [metrics[a]["RandomForest"]["auroc"]["mean"] for a in abx]
    rf_sd = [metrics[a]["RandomForest"]["auroc"]["std"] for a in abx]
    lr = [metrics[a]["LogisticRegression"]["auroc"]["mean"] for a in abx]
    lr_sd = [metrics[a]["LogisticRegression"]["auroc"]["std"] for a in abx]

    x = np.arange(len(abx)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w/2, rf, w, yerr=rf_sd, capsize=3, label="RandomForest", color="#4c78a8")
    ax.bar(x + w/2, lr, w, yerr=lr_sd, capsize=3, label="LogisticRegression", color="#f58518")
    ax.axhline(0.5, color="grey", ls="--", lw=1, label="Random baseline")
    ax.set_xticks(x); ax.set_xticklabels(abx, rotation=20, ha="right")
    ax.set_ylabel("AUROC (5-fold CV, mean ± SD)")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("AMR phenotype prediction from genomic biomarkers")
    ax.legend(loc="upper right")
    fig.tight_layout()
    out = FIG / "auroc_by_antibiotic.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"Saved {out}")


def plot_biomarkers(imp):
    """Heatmap of top-8 biomarker importances per antibiotic (RandomForest)."""
    abx = list(imp.keys())
    # collect union of top-8 across antibiotics
    top_genes = []
    for a in abx:
        for g in list(imp[a]["RandomForest"].keys())[:8]:
            if g not in top_genes:
                top_genes.append(g)
    mat = np.zeros((len(abx), len(top_genes)))
    for i, a in enumerate(abx):
        d = imp[a]["RandomForest"]
        for j, g in enumerate(top_genes):
            mat[i, j] = d.get(g, 0.0)
    # normalise rows so each antibiotic's drivers are comparable
    mat = mat / (mat.max(axis=1, keepdims=True) + 1e-9)

    fig, ax = plt.subplots(figsize=(max(8, len(top_genes)*0.55), 4.5))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(top_genes)))
    ax.set_xticklabels(top_genes, rotation=90, fontsize=7)
    ax.set_yticks(range(len(abx))); ax.set_yticklabels(abx)
    ax.set_title("Relative biomarker importance (row-normalised, RandomForest)")
    fig.colorbar(im, ax=ax, label="relative importance")
    fig.tight_layout()
    out = FIG / "top_biomarkers.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"Saved {out}")


def check_ground_truth(imp, data_dir):
    gt_path = data_dir / "ground_truth.json"
    if not gt_path.exists():
        print("No ground_truth.json (real data?) - skipping recovery check.")
        return
    gt = json.loads(gt_path.read_text())
    report = {}
    for abx, drivers in gt.items():
        if abx not in imp:
            continue
        ranked = list(imp[abx]["RandomForest"].keys())
        top8 = set(ranked[:8])
        recovered = [g for g in drivers if g in top8]
        report[abx] = {
            "true_drivers": drivers,
            "recovered_in_top8": recovered,
            "recall": round(len(recovered) / len(drivers), 3),
        }
    (RESULTS / "ground_truth_recovery.json").write_text(json.dumps(report, indent=2))
    print("\n=== Ground-truth driver recovery (top-8 features) ===")
    for abx, r in report.items():
        print(f"  {abx:15s} recall={r['recall']:.2f}  "
              f"recovered={r['recovered_in_top8']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--data-dir", default="data/processed")
    args = ap.parse_args()

    RESULTS = Path(args.results_dir)
    FIG = RESULTS / "figures"
    DATA = Path(args.data_dir)
    FIG.mkdir(parents=True, exist_ok=True)

    metrics = json.loads((RESULTS / "metrics.json").read_text())
    imp = json.loads((RESULTS / "feature_importance.json").read_text())
    plot_auroc(metrics)
    plot_biomarkers(imp)
    check_ground_truth(imp, DATA)


if __name__ == "__main__":
    main()
