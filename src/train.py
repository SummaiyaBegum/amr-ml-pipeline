r"""
src/train.py

Trains one interpretable, "off-the-shelf" classifier per antibiotic to predict
R/S phenotype from AMR-gene presence/absence features.

Design choices that mirror the KAUST project description:
  * stratified k-fold cross-validation reported per antibiotic (not a single
    train/test split) -- honest performance on small, imbalanced data.
  * RandomForest + LogisticRegression, both interpretable; we report AUROC,
    F1, sensitivity/specificity and the top-ranked biomarkers per antibiotic.
  * class_weight='balanced' to handle antibiotic-specific class imbalance.
  * models, metrics and feature importances are written to models/ and
    results/ so the whole thing is reproducible and deployable.

Run:
    py -3 -I src/train.py [--data-dir data/processed] [--out-dir results]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, recall_score, make_scorer


SEED = 7
N_SPLITS = 5


def load_dataset(data_dir):
    feats = pd.read_csv(Path(data_dir) / "features.csv")
    labels = pd.read_csv(Path(data_dir) / "labels.csv")
    df = feats.merge(labels, on="sample_id")
    gene_cols = [c for c in feats.columns if c != "sample_id"]
    abx_cols = [c for c in labels.columns if c != "sample_id"]
    return df, gene_cols, abx_cols


def build_models():
    """Two off-the-shelf, interpretable models."""
    return {
        "RandomForest": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=400, max_depth=None, min_samples_leaf=3,
                class_weight="balanced", random_state=SEED, n_jobs=-1)),
        ]),
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000, class_weight="balanced", C=1.0)),
        ]),
    }


def cv_metrics(model, X, y_bin):
    """Stratified CV; returns mean +/- std for several metrics."""
    scorers = {
        "auroc": "roc_auc",
        "f1": "f1",
        "sensitivity": make_scorer(recall_score, pos_label=1),
        "specificity": make_scorer(recall_score, pos_label=0),
        "accuracy": "accuracy",
    }
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    res = cross_validate(model, X, y_bin, cv=cv, scoring=scorers, n_jobs=-1)
    out = {}
    for m in scorers:
        vals = res[f"test_{m}"]
        out[m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return out


def get_importances(model, gene_cols):
    """Return dict gene->importance for either model type."""
    clf = model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        vals = clf.feature_importances_
    else:  # logistic regression: use |coefficient|
        vals = np.abs(clf.coef_[0])
    s = pd.Series(vals, index=gene_cols).sort_values(ascending=False)
    return s.to_dict()


def compute_shap_importance(model, X, gene_cols):
    """Compute mean absolute SHAP values for each feature for a fitted model."""
    clf = model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):  # tree-based
        explainer = shap.TreeExplainer(clf)
        shap_vals = explainer.shap_values(X)  # list of arrays (one per class)
        if len(shap_vals) >= 2:
            class_shap = shap_vals[1]  # binary classification: class 1 (resistant)
        else:
            # Fallback: use class 0 and add dimension
            class_shap = shap_vals[0][:, np.newaxis]  # (n_samples, 1)
    else:  # linear model
        explainer = shap.LinearExplainer(clf, X, feature_perturbation="interventional")
        shap_vals = explainer.shap_values(X)
        if len(shap_vals) >= 2:
            class_shap = shap_vals[1]
        else:
            class_shap = shap_vals[0][:, np.newaxis]  # fallback

    # Ensure class_shap is 2D: (n_samples, n_features)
    if class_shap.ndim == 1:
        class_shap = class_shap[:, np.newaxis]  # (n_samples, 1)

    # Compute mean absolute SHAP per feature (across samples)
    mean_abs = np.mean(np.abs(class_shap), axis=0)  # (n_features,)

    # Align with gene_cols length to avoid index errors
    n_features = len(gene_cols)
    if mean_abs.shape[0] < n_features:
        mean_abs = np.pad(mean_abs, (0, n_features - mean_abs.shape[0]), mode='constant')
    elif mean_abs.shape[0] > n_features:
        mean_abs = mean_abs[:n_features]

    imp_dict = {gene: float(mean_abs[i]) for i, gene in enumerate(gene_cols)}
    return imp_dict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--model-dir", default="models")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir); model_dir.mkdir(parents=True, exist_ok=True)

    df, gene_cols, abx_cols = load_dataset(args.data_dir)
    X = df[gene_cols].to_numpy()
    print(f"Dataset: {len(df)} genomes x {len(gene_cols)} features, "
          f"{len(abx_cols)} antibiotics\n")

    all_metrics = {}
    all_importances = {}
    shap_feature_importance = {}

    for abx in abx_cols:
        y_bin = (df[abx] == "R").astype(int).to_numpy()
        # skip antibiotics with too few of one class for CV
        if min(y_bin.sum(), (1 - y_bin).sum()) < N_SPLITS:
            print(f"[skip] {abx}: too few samples in one class")
            continue

        print(f"=== {abx}  (R={int(y_bin.sum())}/{len(y_bin)}) ===")
        all_metrics[abx] = {}
        all_importances[abx] = {}

        for name, model in build_models().items():
            metrics = cv_metrics(model, X, y_bin)
            all_metrics[abx][name] = metrics
            # fit on all data for deployment + importance
            model.fit(X, y_bin)
            joblib.dump(model, model_dir / f"{abx}_{name}.joblib")
            imp = get_importances(model, gene_cols)
            all_importances[abx][name] = imp
            # compute SHAP importance
            shap_imp = compute_shap_importance(model, X, gene_cols)
            shap_feature_importance.setdefault(abx, {})[name] = shap_imp

            auc = metrics["auroc"]["mean"]; auc_sd = metrics["auroc"]["std"]
            f1 = metrics["f1"]["mean"]
            print(f"  {name:18s} AUROC={auc:.3f}+/-{auc_sd:.3f}  F1={f1:.3f}")
            top = list(imp.items())[:5]
            print("    top biomarkers: " + ", ".join(f"{g}({v:.3f})" for g, v in top))

    # persist results
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    with open(out_dir / "feature_importance.json", "w", encoding="utf-8") as f:
        json.dump(all_importances, f, indent=2)
    with open(out_dir / "shap_feature_importance.json", "w", encoding="utf-8") as f:
        json.dump(shap_feature_importance, f, indent=2)

    # compact summary table (RandomForest)
    rows = []
    for abx, d in all_metrics.items():
        m = d["RandomForest"]
        rows.append({
            "antibiotic": abx,
            "auroc": round(m["auroc"]["mean"], 3),
            "auroc_sd": round(m["auroc"]["std"], 3),
            "f1": round(m["f1"]["mean"], 3),
            "sensitivity": round(m["sensitivity"]["mean"], 3),
            "specificity": round(m["specificity"]["mean"], 3),
        })
    summary = pd.DataFrame(rows).sort_values("auroc", ascending=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print("\n=== Summary (RandomForest) ===")
    print(summary.to_string(index=False))
    print(f"\nSaved models -> {model_dir}/, metrics -> {out_dir}/, SHAP importance -> {out_dir}/")

    # also write summary of SHAP top biomarkers per antibiotic
    for abx, models_shap in shap_feature_importance.items():
        print(f"\nSHAP top biomarkers for {abx}:")
        for name, imp_dict in models_shap.items():
            top = list(imp_dict.items())[:5]
            print(f"  {name}: " + ", ".join(f"{g}({v:.3f})" for g, v in top))

if __name__ == "__main__":
    main()