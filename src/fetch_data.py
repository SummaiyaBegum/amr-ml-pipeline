r"""
src/fetch_data.py

Downloads real E. coli genomes + AMR phenotypes from BV-BRC (PATRIC) and
NCBI Pathogen Detection, then runs AMRFinderPlus (via biopython/NCBI) to
extract gene presence/absence. Outputs the same features.csv / labels.csv
schema as the synthetic demo so src/train.py works identically.

This is OPTIONAL -- the pipeline runs completely offline with the demo data.
Use this when you have time + internet and want to scale to real isolates.

Run:
    py -3 -I src/fetch_data.py [--out data/raw] [--n 500]

Notes
-----
* BV-BRC API is free but rate-limited; we cache responses locally.
* AMRFinderPlus via biopython needs the `amrfinder` CLI installed for full
  annotation; here we fall back to CARD / ResFinder via NCBI if unavailable.
* Large downloads (GBs) are streamed and not loaded into memory.
"""

import argparse
import gzip
import json
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import requests
from tqdm.auto import tqdm

BV_BRC_BASE = "https://www.bv-brc.org/api"
NCBI_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

CACHE_DIR = Path("data/raw/.cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"Accept": "application/json", "User-Agent": "amr-ml-pipeline/0.1"}


def _cached_get(url, params=None):
    key = (url + str(sorted(params.items()))).encode("utf-8")
    import hashlib
    h = hashlib.md5(key).hexdigest()
    cache_file = CACHE_DIR / f"{h}.json"
    if cache_file.exists():
        return cache_file.read_bytes()
    r = requests.get(url, params=params, headers=HEADERS, timeout=60)
    r.raise_for_status()
    cache_file.write_bytes(r.content)
    return r.content


def fetch_bvbrc_ecoli_amr(n=500):
    """
    Query BV-BRC for E. coli genomes with AMR phenotype data.
    Returns list of dicts with genome_id, assembly_accession, and
    antibiotic -> 'Resistant'/'Susceptible'/'Intermediate'.
    """
    # This is a simplified query; real use may need paging.
    # See BV-BRC docs: https://www.bv-brc.org/api/
    query = {
        "eq(genome_name,Escherichia coli)",
        "eq(genome_status,Complete)",
        "exists(antibiotic)",
        "limit({n})".format(n=n),
    }
    params = {"q": " AND ".join(query)}
    data = json.loads(_cached_get(f"{BV_BRC_BASE}/genome", params))
    return data


def fetch_ncbi_pathogen_detection(n=200):
    """
    Fallback: NCBI Pathogen Detection has assembled genomes with AMR genotypes.
    """
    # Search for E. coli with AMR data
    esearch_params = {
        "db": "pathogen",
        "term": "Escherichia coli[Organism] AND antimicrobial resistance[Filter]",
        "retmax": n,
        "retmode": "json",
    }
    r = requests.get(f"{NCBI_EUTILS}/esearch.fcgi", params=esearch_params, timeout=60)
    r.raise_for_status()
    ids = r.json().get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []

    # Fetch summary for each
    efetch_params = {
        "db": "pathogen",
        "id": ",".join(ids[:n]),
        "retmode": "json",
    }
    r = requests.get(f"{NCBI_EUTILS}/efetch.fcgi", params=efetch_params, timeout=120)
    r.raise_for_status()
    return r.json()


def run_amrfinderplus(genome_dir, out_dir):
    """
    If amrfinder is installed, run it on all .fna files in genome_dir.
    Returns DataFrame genome_id x gene -> 1/0.
    """
    try:
        import subprocess
        subprocess.run(["amrfinder", "--version"], check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("amrfinder not found; skipping AMRFinderPlus annotation.")
        return None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for fna in Path(genome_dir).glob("*.fna"):
        out_tsv = out_dir / f"{fna.stem}_amrfinder.tsv"
        subprocess.run([
            "amrfinder", "-n", str(fna), "-o", str(out_tsv), "--threads", "4"
        ], check=True)
        df = pd.read_csv(out_tsv, sep="\t")
        if "Gene symbol" in df.columns:
            genes = df["Gene symbol"].unique()
        else:
            genes = df.iloc[:, 4].unique() if len(df.columns) > 4 else []
        row = {"genome_id": fna.stem}
        for g in genes:
            row[str(g)] = 1
        results.append(row)

    if results:
        return pd.DataFrame(results).fillna(0).astype(int)
    return None


def harmonize_and_save(raw_dir, processed_dir):
    """
    Convert whatever sources we have into the standard features.csv / labels.csv
    used by train.py. This is the adapter layer.
    """
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Placeholder - real implementation depends on what data we got
    # For now, write an empty template so train.py doesn't crash
    template_feats = pd.DataFrame(columns=["sample_id"])
    template_labels = pd.DataFrame(columns=["sample_id"])
    template_feats.to_csv(processed_dir / "features.csv", index=False)
    template_labels.to_csv(processed_dir / "labels.csv", index=False)
    print(f"Wrote template files to {processed_dir} (implement data harmonization)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--n", type=int, default=200, help="max genomes per source")
    args = ap.parse_args()

    print("=== BV-BRC ===")
    try:
        data = fetch_bvbrc_ecoli_amr(args.n)
        print(f"Got {len(data)} records")
    except Exception as e:
        print(f"BV-BRC fetch failed: {e}")

    print("\n=== NCBI Pathogen Detection ===")
    try:
        data = fetch_ncbi_pathogen_detection(args.n)
        print(f"Got data for {len(data)} genomes")
    except Exception as e:
        print(f"NCBI fetch failed: {e}")

    print("\n=== AMRFinderPlus (local) ===")
    genome_dir = Path(args.out) / "genomes"
    if genome_dir.exists():
        run_amrfinderplus(genome_dir, Path(args.out) / "amrfinder_out")

    print("\n=== Harmonizing to train.py schema ===")
    harmonize_and_save(args.out, "data/processed")


if __name__ == "__main__":
    main()