import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Load a sample model
model_path = Path("models/Ampicillin_RandomForest.joblib")
model = joblib.load(model_path)

# Load feature columns from features.csv
features_df = pd.read_csv("data/processed/features.csv", nrows=1)
gene_cols = [c for c in features_df.columns if c != "sample_id"]

# Create a sample input (all genes absent)
sample_input = np.array([[0] * len(gene_cols)])  # all zeros

# Predict
pred_proba = model.predict_proba(sample_input)[0, 1]  # probability of resistant
pred_call = "R" if pred_proba >= 0.5 else "S"

print(f"Ampicillin prediction: P(resistant) = {pred_proba:.3f}, call = {pred_call}")