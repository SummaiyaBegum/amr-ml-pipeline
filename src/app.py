r"""
src/app.py

Minimal Streamlit demo that deploys the trained per-antibiotic classifiers.
Given a set of detected AMR genes, predicts resistant / susceptible
probability for each antibiotic with a trained model.

Run:
    py -3 -m streamlit run src/app.py

This is the "deploy the model as a data science solution" deliverable from the
KAUST project description, in its simplest defensible form.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_DIR = Path("models")
DATA = Path("data/processed")

st.set_page_config(page_title="AMR phenotype predictor", layout="wide")
st.title("Antimicrobial resistance phenotype predictor")
st.caption(
    "Predicts R/S phenotype from AMR-gene presence/absence. "
    "Demo models trained on synthetic E. coli-like data; swap in real genomes "
    "with src/fetch_data.py."
)

# Discover trained models (both RandomForest and LogisticRegression)
model_files = sorted(MODEL_DIR.glob("*_RandomForest.joblib") + MODEL_DIR.glob("*_LogisticRegression.joblib"))
if not model_files:
    st.error("No trained models found. Run `py -3 -I src/train.py` first.")
    st.stop()

antibiotics = [f.stem.replace("_RandomForest", "") for f in model_files]

# Feature ordering must match training - read gene columns from features.csv
feats = pd.read_csv(DATA / "features.csv", nrows=1)
gene_cols = [c for c in feats.columns if c != "sample_id"]

st.sidebar.header("Detected AMR genes")
st.sidebar.write("Toggle the genes found in your genome:")
selected = {}
for g in gene_cols:
    selected[g] = st.sidebar.checkbox(g, value=False)

x = np.array([[1 if selected[g] else 0 for g in gene_cols]])

st.subheader("Predicted resistance probabilities")
rows = []
for abx, f in zip(antibiotics, model_files):
    model = joblib.load(f)
    p = float(model.predict_proba(x)[0, 1])
    rows.append({"antibiotic": abx, "P(resistant)": round(p, 3),
                 "call": "R" if p >= 0.5 else "S"})
st.dataframe(pd.DataFrame(rows), use_container_width=True)

st.caption(
    "P(resistant) is the model's probability. "
    "Threshold 0.5 shown; tune per-antibiotic for clinical use."
)
