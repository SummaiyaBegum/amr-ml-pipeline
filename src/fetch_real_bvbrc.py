r"""
src/fetch_real_bvbrc.py

Download REAL E. coli genomes + AMR genes + AMR phenotypes from BV-BRC and
harmonise to the same features.csv / labels.csv schema as the synthetic demo,
so src/train.py works unchanged.

Strategy:
  1. Pull ALL genome_amr phenotypes in ONE large request (limit=20000 works)
  2. Extract unique genome_ids that have at least one lab R/S measurement
  3. For those genome_ids, pull sp_gene (Antibiotic Resistance genes) using
     in(genome_id,...) filters in chunks
  4. Harmonise to features/labels schema

Endpoints:
    /genome_amr/?eq(taxon_id,562)&limit(20000)
    /sp_gene/?in(genome_id,(id1,id2,...))&eq(property,Antibiotic Resistance)&limit(N)

Outputs (data/processed_real/):
    features.csv : sample_id + one 1/0 column per AMR gene
    labels.csv   : sample_id + one R/S column per antibiotic
    provenance.json : exactly what was pulled, when, and how many

Run:
    py -3 -I src/fetch_real_bvbrc.py --phenotype-limit 20000
Then train on it:
    py -3 -I src/train.py --data-dir data/processed_real --out-dir results_real \
        --model-dir models_real
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

BASE = "https://www.bv-brc.org/api"
HEADERS = {"Accept": "application/json", "User-Agent": "amr-ml-pipeline/0.1"}
TAXON_ECOLI = 562

# Canonical antibiotics -> accepted raw names from BV-BRC
CANON = {
    "Ampicillin": {"ampicillin"},
    "Ciprofloxacin": {"ciprofloxacin"},
    "Tetracycline": {"tetracycline"},
    "Gentamicin": {"gentamicin"},
    "Trimethoprim-sulfamethoxazole": {
        "trimethoprim/sulfamethoxazole", "trimethoprim-sulfamethoxazole",
        "sulfamethoxazole/trimethoprim", "co-trimoxazole",
    },
    "Ceftriaxone": {"ceftriaxone"},
    "Meropenem": {"meropenem"},
    "Colistin": {"colistin", "polymyxin b", "polymyxin"},
    "Cefotaxime": {"cefotaxime"},
    "Cefepime": {"cefepime"},
    "Amikacin": {"amikacin"},
    "Tobramycin": {"tobramycin"},
    "Aztreonam": {"aztreonam"},
    "Ertapenem": {"ertapenem"},
    "Imipenem": {"imipenem"},
}


def _get(path, retries=3):
    url = f"{BASE}/{path}"
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=180)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 502, 503, 504):
                time.sleep(5 * (attempt + 1))
                continue
            raise RuntimeError(f"{path} -> {r.status_code}: {r.text[:300]}")
        except (requests.RequestException, ValueError) as e:
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))
    return []


def fetch_all_phenotypes(limit):
    """Single large request for all phenotypes."""
    print(f"Fetching all lab phenotypes (limit={limit})...")
    path = f"genome_amr/?eq(taxon_id,{TAXON_ECOLI})&limit({limit})"
    rows = _get(path)
    if not rows:
        raise RuntimeError("Empty phenotype response")

    canon_lookup = {}
    for canon, aliases in CANON.items():
        for a in aliases:
            canon_lookup[a.lower()] = canon

    keep = []
    for x in rows:
        abx_raw = (x.get("antibiotic") or "").lower().strip()
        abx = canon_lookup.get(abx_raw)
        phen = (x.get("resistant_phenotype") or "").lower()
        if not abx:
            continue
        if phen.startswith("resist"):
            keep.append((x["genome_id"], abx, "R"))
        elif phen.startswith("suscept"):
            keep.append((x["genome_id"], abx, "S"))
        elif phen.startswith("intermed"):
            keep.append((x["genome_id"], abx, "I"))

    df = pd.DataFrame(keep, columns=["genome_id", "antibiotic", "phenotype"])
    if df.empty:
        return df

    # collapse duplicates: R dominates, then S, ignore I
    df = df[df.phenotype != "I"]
    df = df.groupby(["genome_id", "antibiotic"], as_index=False).agg(
        phenotype=("phenotype", lambda s: "R" if "R" in set(s) else "S"))

    print(f"  {len(rows)} raw phenotype records -> {len(df)} usable, "
          f"{df['genome_id'].nunique()} genomes with phenotypes")
    return df


def fetch_genes_for_genomes(genome_ids, chunk_size=300):
    """Pull Antibiotic Resistance genes for a specific set of genome_ids."""
    print(f"Fetching AMR genes for {len(genome_ids)} genomes (chunk={chunk_size})...")
    genome_ids = list(dict.fromkeys(genome_ids))  # preserve order, dedup
    all_rows = []

    for i in range(0, len(genome_ids), chunk_size):
        chunk = genome_ids[i:i+chunk_size]
        ids_str = ",".join(chunk)
        # NOTE: server-side `eq(property,...)` with a space silently returns 0 rows,
        # so we pull all specialty genes and filter to Antibiotic Resistance locally.
        path = f"sp_gene/?in(genome_id,({ids_str}))&limit(20000)"
        rows = _get(path)
        n_amr = 0
        for x in rows:
            if x.get("property") == "Antibiotic Resistance" and x.get("gene"):
                all_rows.append((x["genome_id"], x["gene"]))
                n_amr += 1
        print(f"  chunk {i//chunk_size + 1}/{ (len(genome_ids)-1)//chunk_size + 1 }: "
              f"{len(chunk)} genomes -> {len(rows)} rows, {n_amr} AMR gene hits")
        time.sleep(0.5)  # be nice to the API

    df = pd.DataFrame(all_rows, columns=["genome_id", "gene"]).drop_duplicates()
    print(f"  total: {len(df)} unique genome/gene pairs, "
          f"{df['genome_id'].nunique()} genomes, {df['gene'].nunique()} genes")
    return df


def harmonise(genes_df, pheno_df, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # genomes present in BOTH genes and phenotypes
    common = set(genes_df.genome_id) & set(pheno_df.genome_id)
    print(f"\nGenomes with both genes and phenotypes: {len(common)}")
    if len(common) < 20:
        print("  WARNING: very few overlapping genomes; try a larger --phenotype-limit.")

    genes_df = genes_df[genes_df.genome_id.isin(common)]
    pheno_df = pheno_df[pheno_df.genome_id.isin(common)]

    # features: pivot gene presence -> 1/0
    feats = (genes_df.assign(val=1)
             .pivot_table(index="genome_id", columns="gene", values="val",
                          aggfunc="max", fill_value=0)
             .reset_index().rename(columns={"genome_id": "sample_id"}))
    feats.columns = ["sample_id"] + [str(c) for c in feats.columns[1:]]

    # drop genes present in <2% or >98% of genomes (uninformative)
    gene_cols = [c for c in feats.columns if c != "sample_id"]
    frac = feats[gene_cols].mean()
    keep = frac[(frac >= 0.02) & (frac <= 0.98)].index.tolist()
    dropped = len(gene_cols) - len(keep)
    feats = feats[["sample_id"] + keep]

    # labels: pivot to one column per antibiotic
    labels = (pheno_df.pivot_table(index="genome_id", columns="antibiotic",
                                   values="phenotype", aggfunc="first")
              .reset_index().rename(columns={"genome_id": "sample_id"}))

    # keep only antibiotics with enough of both classes
    abx_cols = [c for c in labels.columns if c != "sample_id"]
    good_abx = []
    for a in abx_cols:
        vc = labels[a].value_counts()
        if vc.get("R", 0) >= 10 and vc.get("S", 0) >= 10:
            good_abx.append(a)
    labels = labels[["sample_id"] + good_abx]

    feats.to_csv(out_dir / "features.csv", index=False)
    labels.to_csv(out_dir / "labels.csv", index=False)

    prov = {
        "source": "BV-BRC (bv-brc.org)",
        "taxon_id": TAXON_ECOLI,
        "organism": "Escherichia coli",
        "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "genomes": int(feats.shape[0]),
        "genes": int(len(keep)),
        "genes_dropped_uninformative": int(dropped),
        "antibiotics": good_abx,
        "class_balance": {
            a: labels[a].value_counts().to_dict() for a in good_abx
        },
    }
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2))
    print(f"\nWrote {out_dir/'features.csv'}  ({feats.shape[0]} genomes x {len(keep)} genes)")
    print(f"Wrote {out_dir/'labels.csv'}    ({len(good_abx)} antibiotics)")
    print("Class balance:")
    for a in good_abx:
        print(f"  {a:35s} {prov['class_balance'][a]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phenotype-limit", type=int, default=20000,
                    help="max phenotype records to request (single BV-BRC request)")
    ap.add_argument("--out", default="data/processed_real")
    args = ap.parse_args()

    pheno = fetch_all_phenotypes(args.phenotype_limit)
    if pheno.empty:
        raise SystemExit("No phenotype data returned; check network/BV-BRC availability.")
    genome_ids = pheno["genome_id"].unique().tolist()
    genes = fetch_genes_for_genomes(genome_ids)
    if genes.empty:
        raise SystemExit("No AMR genes returned for those genomes; try larger phenotype-limit.")
    harmonise(genes, pheno, args.out)


if __name__ == "__main__":
    main()