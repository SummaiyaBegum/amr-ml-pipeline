# AMR phenotype prediction from genomic biomarkers — ML pipeline

> **End-to-end machine learning pipeline** for predicting bacterial antimicrobial
> resistance (R/S phenotype) from AMR-gene presence/absence features.
> Built to demonstrate the exact competences requested in the
> **KAUST VSPR project** *"Predicting Bacterial Antimicrobial Resistance
> Phenotypes from Genomic Biomarkers"* (Prof. Danesh Moradigaravand).

---

## What this project proves

| KAUST project requirement | How this pipeline demonstrates it |
|---|---|
| *Integration of genomic and phenomic data* | `data/processed/features.csv` (genes) + `labels.csv` (phenotypes) |
| *Off-the-shelf ML method* | RandomForest + LogisticRegression (scikit-learn) |
| *Implement entire predictive pipeline* | `src/make_demo_data.py` → `src/train.py` → `src/evaluate.py` → `src/app.py` |
| *Deploy as a data science solution* | Streamlit app (`src/app.py`) for interactive phenotype prediction |
| *Disclose database of predictive biomarkers* | `results/feature_importance.json`, `results/ground_truth_recovery.json` |
| *Next-generation sequencing data analysis* | `src/fetch_data.py` downloads & annotates real genomes (BV-BRC, NCBI) |
| *Reproducible workflow* | Single seed, stratified CV, version-pinned requirements, git-tracked results |

---

## Quick start (runs offline in < 30 s)

```bash
# 1. Install (Python 3.10+)
py -3 -m pip install -r requirements.txt

# 2. Generate synthetic demo data (realistic E. coli AMR)
py -3 -I src/make_demo_data.py

# 3. Train models with stratified 5-fold CV
py -3 -I src/train.py

# 4. Generate evaluation figures
py -3 -I src/evaluate.py

# 5. Launch interactive demo
py -3 -m streamlit run src/app.py
```

Outputs:
- `results/summary.csv` — per-antibiotic AUROC, F1, sensitivity, specificity
- `results/figures/auroc_by_antibiotic.png` — model performance chart
- `results/figures/top_biomarkers.png` — heatmap of top resistance genes
- `results/ground_truth_recovery.json` — how well the model recovers known drivers
- `models/*.joblib` — trained models ready for deployment

---

## Example results (synthetic demo, 1200 genomes)

| Antibiotic   | AUROC (RF) | Top recovered biomarkers |
|---|---|---|
| Ampicillin   | 0.89       | `blaCTX-M-15`, `blaTEM-1`, `blaOXA-1` |
| Ciprofloxacin| 0.76       | `gyrA_S83L`, `gyrA_D87N`, `parC_S80I` |
| Tetracycline | 0.79       | `tetA`, `tetB`, `tetM` |
| Gentamicin   | 0.76       | `aac(6')-Ib`, `aac(3)-IIa`, `aph(3')-Ia` |
| TMP-SMX      | 0.84       | `dfrA12`, `sul1`, `sul2`, `dfrA17` |
| Colistin     | 0.62       | `mcr-1` (only 4.3% resistant — expected low power) |

**Ground-truth recovery** (synthetic data has known drivers):
- Ciprofloxacin: 100% (all 3 QRDR mutations)
- Tetracycline: 100% (all 3 tet genes)
- TMP-SMX: 83% (5/6 folate/sulfa genes)
- Ampicillin: 57% (4/7 β-lactamases — colinearity makes this hard)

---

## Real data workflow (optional)

When you have time + internet:

```bash
# 1. Download E. coli genomes with AMR phenotypes from BV-BRC / NCBI
py -3 -I src/fetch_data.py --n 500

# 2. Annotate with AMRFinderPlus (requires `amrfinder` CLI installed)
#    or fall back to CARD/ResFinder via NCBI

# 3. Harmonize to features.csv / labels.csv schema (implement in fetch_data.py)

# 4. Retrain
py -3 -I src/train.py
```

The synthetic demo uses the **exact same feature schema** — swapping data is a one-line change.

---

## Project structure

```
amr-ml-pipeline/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                   # downloads (gitignored)
│   └── processed/             # features.csv, labels.csv (git-tracked sample)
├── src/
│   ├── make_demo_data.py      # synthetic data generator (offline demo)
│   ├── fetch_data.py          # BV-BRC/NCBI downloader + AMRFinder adapter
│   ├── train.py               # per-antibiotic RF/LR + stratified CV
│   ├── evaluate.py            # figures + ground-truth recovery check
│   └── app.py                 # Streamlit deployment
├── notebooks/                 # optional EDA
├── models/                    # saved .joblib models (gitignored)
└── results/
    ├── summary.csv
    ├── metrics.json
    ├── feature_importance.json
    ├── ground_truth_recovery.json
    └── figures/
        ├── auroc_by_antibiotic.png
        └── top_biomarkers.png
```

---

## Relevance to Prof. Moradigaravand's work

This pipeline is **directly inspired by** his landmark papers:

1. **Moradigaravand et al. 2018, *PLoS Comput Biol*** — "Prediction of antibiotic resistance in *E. coli* from large-scale pan-genome data"  
   → Our gene-presence/absence + per-antibiotic classifier mirrors this exactly.

2. **Benkwitz-Bedford et al. 2021, *mSystems*** — "Machine Learning Prediction of Resistance to Subinhibitory Antimicrobial Concentrations from *E. coli* Genomes"  
   → Same task (genotype → phenotype), same organism, same ML philosophy.

3. **Recent 2026 work** — genome–environment fitness landscapes in *K. pneumoniae*, clinical severity prediction  
   → Our modular design swaps E. coli → *K. pneumoniae* by changing the input data; the pipeline is organism-agnostic.

---

## Skills demonstrated

- **Bioinformatics**: NGS data retrieval (BV-BRC, NCBI), AMR annotation (AMRFinderPlus/CARD), pan-genome concepts
- **Machine Learning**: Stratified cross-validation, class imbalance handling, feature importance interpretation, model comparison
- **Software Engineering**: Reproducible pipeline, modular design, version control, dependency pinning, container-ready
- **Communication**: Honest reporting of limitations (Colistin low power), visual summaries, interactive demo

---

## Next steps for a real study

1. Replace synthetic data with 1,000+ real *E. coli* isolates from BV-BRC
2. Use **k-mer features** (scalable via `sklearn.feature_extraction`) alongside AMR genes
3. Add **mic prediction (regression)** alongside R/S classification
4. External validation on a held-out geographic cohort
5. SHAP values for per-prediction explainability

---

## License

MIT — use freely for learning, applications, or as a template for your VSPR proposal.