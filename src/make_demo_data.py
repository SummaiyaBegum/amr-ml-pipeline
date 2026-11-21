r"""
src/make_demo_data.py

Generates a realistic *synthetic* E. coli AMR dataset so the whole pipeline can
be demonstrated offline and reproducibly.

Schema mirrors what a real workflow produces from AMRFinderPlus / BV-BRC:
  - data/processed/features.csv : one row per genome, 1/0 columns = AMR gene
                                  presence (plus a few resistance SNPs).
  - data/processed/labels.csv   : one row per genome, one column per antibiotic
                                  with 'R' or 'S' phenotype.

The resistance label for each antibiotic is generated from a small set of
"true driver" genes via a logistic model plus label noise, WITH correlated
background carriage so results are not trivially perfect. The model has to
recover the drivers from the data -- exactly the task the KAUST project
describes. Ground-truth genes are written to data/processed/ground_truth.json
so you can measure how well the model recovers them.

NOTE: This is clearly-labelled demo data for method development only. Swap in
real genomes with src/fetch_data.py for a real study.

Run:
    py -3 -I src/make_demo_data.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 7
N_SAMPLES = 1200
OUT = Path("data/processed")

# --- Candidate AMR features (real E. coli determinants; presence/absence) ------
GENES = [
    # beta-lactams
    "blaTEM-1", "blaCTX-M-15", "blaCTX-M-14", "blaSHV-12",
    "blaOXA-1", "blaCMY-2", "blaNDM-1", "blaKPC-2",
    # aminoglycosides
    "aac(3)-IIa", "aac(6')-Ib", "ant(2'')-Ia", "aph(3')-Ia",
    # tetracyclines
    "tetA", "tetB", "tetM",
    # sulfonamides / trimethoprim
    "sul1", "sul2", "sul3", "dfrA1", "dfrA12", "dfrA17",
    # quinolone target mutations
    "gyrA_S83L", "gyrA_D87N", "parC_S80I",
    # phenicols
    "catA1", "floR", "cmlA1",
    # macrolides
    "mphA", "ermB",
    # colistin
    "mcr-1", "mcr-3",
    # fosfomycin
    "fosA3",
]

# Background carriage: genes that travel together on common mobile elements
# (integrons, plasmids). This creates colinearity the model must untangle.
BACKGROUND_GROUPS = [
    ["sul1", "dfrA12", "aac(6')-Ib", "catA1", "cmlA1"],   # class-1 integron
    ["blaTEM-1", "blaCTX-M-15", "blaOXA-1", "tetA"],       # common ESBL plasmid
    ["blaCTX-M-14", "floR", "sul2", "dfrA17", "mphA"],     # IncF / Asia lineage
]

# True drivers per antibiotic: {gene: logit weight}
GROUND_TRUTH = {
    "Ampicillin":   {"blaTEM-1": 1.8, "blaCTX-M-15": 2.2, "blaCTX-M-14": 2.0,
                     "blaSHV-12": 2.1, "blaOXA-1": 1.4, "blaNDM-1": 2.6, "blaKPC-2": 2.6},
    "Ciprofloxacin": {"gyrA_S83L": 2.4, "gyrA_D87N": 2.0, "parC_S80I": 1.3},
    "Tetracycline": {"tetA": 2.1, "tetB": 2.2, "tetM": 2.3},
    "Gentamicin":   {"aac(3)-IIa": 2.2, "aac(6')-Ib": 1.6, "ant(2'')-Ia": 1.4,
                     "aph(3')-Ia": 1.1, "blaNDM-1": 2.0, "blaKPC-2": 2.0},
    "TMP-SMX":      {"sul1": 1.9, "sul2": 1.8, "sul3": 1.6, "dfrA1": 1.7,
                     "dfrA12": 1.7, "dfrA17": 1.6},
    "Colistin":     {"mcr-1": 2.8, "mcr-3": 2.6},
}

# Baseline logit per antibiotic (background resistance rate before drivers)
BASELINE_RESISTANCE = {
    "Ampicillin": -1.1, "Ciprofloxacin": -1.6, "Tetracycline": -1.2,
    "Gentamicin": -1.8, "TMP-SMX": -1.0, "Colistin": -3.6,
}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def gen_features(rng):
    """Presence/absence matrix with realistic prevalence + correlated groups."""
    n = N_SAMPLES
    X = pd.DataFrame(0, index=range(n), columns=GENES, dtype=int)

    # Independent prevalence per gene (roughly reflecting real E. coli isolates)
    prevalence = {
        "blaTEM-1": 0.45, "blaCTX-M-15": 0.25, "blaCTX-M-14": 0.10,
        "blaSHV-12": 0.06, "blaOXA-1": 0.18, "blaCMY-2": 0.08,
        "blaNDM-1": 0.02, "blaKPC-2": 0.02,
        "aac(3)-IIa": 0.14, "aac(6')-Ib": 0.35, "ant(2'')-Ia": 0.12,
        "aph(3')-Ia": 0.22,
        "tetA": 0.32, "tetB": 0.18, "tetM": 0.10,
        "sul1": 0.42, "sul2": 0.30, "sul3": 0.05,
        "dfrA1": 0.10, "dfrA12": 0.15, "dfrA17": 0.20,
        "gyrA_S83L": 0.30, "gyrA_D87N": 0.12, "parC_S80I": 0.18,
        "catA1": 0.18, "floR": 0.16, "cmlA1": 0.08,
        "mphA": 0.22, "ermB": 0.09,
        "mcr-1": 0.03, "mcr-3": 0.02,
        "fosA3": 0.04,
    }
    for g in GENES:
        X[g] = (rng.random(n) < prevalence.get(g, 0.1)).astype(int)

    # Correlated groups: if the "anchor" gene present, partners more likely
    for group in BACKGROUND_GROUPS:
        anchor = group[0]
        carriers = X[anchor] == 1
        for partner in group[1:]:
            boost = carriers & (rng.random(n) < 0.75)
            X.loc[boost, partner] = 1
    return X


def gen_labels(X, rng):
    """Generate R/S phenotype per antibiotic from true drivers + noise."""
    labels = pd.DataFrame(index=X.index)
    for abx, weights in GROUND_TRUTH.items():
        logit = np.full(len(X), BASELINE_RESISTANCE.get(abx, -1.5))
        for gene, w in weights.items():
            logit += w * X[gene].to_numpy()
        p = _sigmoid(logit)
        # 8% label noise (measurement error / borderline MIC)
        p = np.clip(p + rng.normal(0, 0.05, size=len(p)), 0.0, 1.0)
        labels[abx] = np.where(rng.random(len(p)) < p, "R", "S")
    return labels


def main():
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    X = gen_features(rng)
    X.insert(0, "sample_id", [f"ECOLI_{i:05d}" for i in range(len(X))])
    y = gen_labels(X.drop(columns="sample_id"), rng)
    y.insert(0, "sample_id", X["sample_id"])

    X.to_csv(OUT / "features.csv", index=False)
    y.to_csv(OUT / "labels.csv", index=False)
    with open(OUT / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(GROUND_TRUTH, f, indent=2)

    print(f"Wrote {len(X)} genomes x {len(GENES)} genes -> {OUT/'features.csv'}")
    print(f"Wrote labels for {len(GROUND_TRUTH)} antibiotics -> {OUT/'labels.csv'}")
    print("\nResistance rate per antibiotic:")
    for abx in GROUND_TRUTH:
        rate = (y[abx] == "R").mean()
        print(f"  {abx:15s} {rate:.1%}")


if __name__ == "__main__":
    main()
