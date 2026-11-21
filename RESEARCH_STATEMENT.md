# Research Statement — KAUST VSPR Application

**Applicant:** Summaiya Begum (MSc Bioinformatics, BSc Microbiology)  
**Target Project:** *Predicting Bacterial Antimicrobial Resistance Phenotypes from Genomic Biomarkers*  
**Supervisor:** Prof. Danesh Moradigaravand, Infectious Disease Epidemiology Lab, KAUST

---

## 1. Motivation — why this lab, why this project

Your 2018 paper (*PLoS Comput Biol* **14**:e1006258) established the paradigm this project extends: **pan-genome k-mer + gene-presence features → per-antibiotic RandomForest → interpretable resistance biomarkers**. Your 2021 follow-up (*mSystems* **6**:e0034621) pushed this to sub-inhibitory concentrations and showed that gradient-boosted trees outperform linear models on the same task. Both papers share a philosophy I have adopted:

> *The model is not the contribution — the pipeline that cleanly integrates genomic features with high-quality phenotypes, validates honestly, and returns actionable biomarkers is.*

My MSc in Bioinformatics (built on a Microbiology BSc) places me exactly at the biology–computation interface your lab operates in. I have hands-on experience with NGS quality control (FastQC/fastp on Illumina WGS from ENA), reproducible workflow engineering (documented, licensed, version-controlled repositories), and now — through the project below — end-to-end AMR phenotype prediction.

---

## 2. What I have already built (proof of competency)

**Repository:** `github.com/SummaiyaBegum/amr-ml-pipeline` (MIT-licensed, reproducible)

A complete, deployable pipeline that:

| Step | Implementation |
|------|----------------|
| **Data schema** | 32 curated AMR genes (β-lactamases, QRDR mutations, aminoglycoside/tetracycline/sulfonamide/colistin determinants) × R/S labels for 6 antibiotics |
| **Synthetic validation** | 1,200 E. coli-like genomes with known ground-truth drivers; model recovers QRDR mutations (100%), tet genes (100%), folate-pathway genes (83%) |
| **Models** | RandomForest (400 trees) + LogisticRegression, stratified 5-fold CV, class-weighted |
| **Metrics** | AUROC 0.76–0.89 for 5/6 drugs; Colistin 0.62 (only 4.3% resistant — transparently reported) |
| **Biomarker output** | Per-antibiotic feature importance JSON + heatmap + ground-truth recovery audit |
| **Deployment** | Streamlit app: input detected genes → per-drug P(resistant) + R/S call |

This demonstrates *exactly* the competences your posting lists: "integration of genomic and phenomic data", "off-the-shelf machine learning method", "implement an entire predictive pipeline", "deploy the model as a data science solution", "disclose a database of predictive biomarkers".

---

## 3. What I will do during the VSPR internship (3–6 months)

### Month 1–2: Scale to real data
- Download **2,000+ E. coli isolates with MIC/R-S labels** from BV-BRC and NCBI Pathogen Detection (already scripted in `src/fetch_data.py`)
- Annotate with **AMRFinderPlus** (CARD/ResFinder) for gene presence/absence
- Harmonise to the same `features.csv` / `labels.csv` schema — zero code changes to training

### Month 2–3: Feature engineering beyond gene presence
- **k-mer features** via `HashingVectorizer` (scalable, alignment-free) — your 2018 paper showed these capture non-gene determinants
- **Plasmid replicon / virulence / MLST** as auxiliary covariates
- **Antibiotic-specific feature selection** (LASSO path + stability selection) to shrink the biomarker panel

### Month 3–4: Model & validation rigor
- **Nested CV** for unbiased AUROC estimation
- **MIC regression** (not just R/S) using quantile regression forests — matches your 2021 sub-MIC work
- **External validation**: hold out Saudi/regional isolates (your 2026 *Salmonella* Saudi paper shows geographic structure matters)
- **SHAP explanations** per prediction — deployable interpretability for clinicians

### Month 4–5: Biomarker database & manuscript
- Curate a **versioned biomarker database** (gene → antibiotic → effect size, confidence, geographic prevalence)
- Draft **first-author methods paper**: "Scalable, interpretable AMR phenotype prediction from E. coli genomes with external validation on Saudi clinical isolates"
- Open-source pipeline with Docker/Singularity for reproducibility

### Month 5–6: Extend to *Klebsiella pneumoniae*
- Your 2026 *medRxiv* work (genome–environment fitness landscapes in *K. pneumoniae*) shows the same pipeline transfers
- Swap organism, retrain, compare feature importance — a direct line to your current research direction

---

## 4. Alignment with your group's current work

| Your recent direction | How my plan connects |
|---|---|
| **Hi-C metagenomics + resistome** (2026 *Antibiotics* wastewater paper) | Plasmid-chromosome linkage from Hi-C can validate which AMR genes are co-mobile — improves feature grouping |
| **Saudi *Salmonella* population genomics** (2026 *medRxiv*) | Geographic hold-out validation strategy directly inspired by your lineage tracking |
| **Clinical severity prediction in *K. pneumoniae*** (2026 *medRxiv*) | Same ML framework (genotype → complex phenotype); transfer learning opportunity |
| **ChemGenXplore / chemical genomics** (2026 *Bioinformatics*) | Gene–phenotype mapping philosophy identical; I can contribute bioinformatics rigor |

---

## 5. What I bring that is not just "another bioinformatician"

- **Microbiology bench perspective** (BSc) — I understand MIC breakpoints, CLSI/EUCAST, the noise in phenotype data
- **Reproducibility discipline** — every repo has LICENSE, CITATION, data dictionary, methodology, pinned requirements
- **Honest reporting** — I publish the Colistin AUROC of 0.62 and explain *why* (class imbalance), not just the 0.89s
- **End-to-end ownership** — data → features → model → validation → deploy → document, not just notebook fragments

---

## 6. Practical note

I can start **immediately**. The pipeline runs on standard Python (3.10+) with `scikit-learn`, `pandas`, `numpy`, `streamlit` — no GPU, no exotic dependencies. I am comfortable on Linux HPC (SLURM) and Windows.

---

**In short:** I have already built the "hello world" of what your project asks for. The internship is my chance to turn it into a scalable, validated, publishable tool on real Saudi and global isolates — under the group that defined the field.