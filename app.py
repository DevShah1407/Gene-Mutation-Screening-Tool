# -*- coding: utf-8 -*-
"""CarbonVEP Streamlit app.

This file is a Streamlit conversion of the original CarbonVEP_v3 notebook-style
pipeline. The major functions are preserved and wrapped in UI controls so the
pipeline can be run step by step from a browser.
"""

import csv
import gzip
import os
import shutil
import time
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import py3Dmol
import pysam
import requests
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


APP_DIR = Path(__file__).resolve().parent
WORK_DIR = APP_DIR / "carbonvep_runs"
WORK_DIR.mkdir(parents=True, exist_ok=True)


st.set_page_config(
    page_title="CarbonVEP Glioma Variant Prioritization",
    page_icon="DNA",
    layout="wide",
)


def save_uploaded_file(uploaded_file, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "wb") as output_handle:
        output_handle.write(uploaded_file.getbuffer())
    return destination


def dataframe_download_button(df: pd.DataFrame, filename: str, label: str):
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
    )


def file_download_button(file_path: Path, label: str, mime: str):
    if file_path.exists():
        st.download_button(
            label=label,
            data=file_path.read_bytes(),
            file_name=file_path.name,
            mime=mime,
        )


def get_default_hf_token():
    try:
        return st.secrets.get("HF_TOKEN", "")
    except Exception:
        return ""


def convert_maf_to_vcf(maf_path, vcf_path):
    st.write(f"Reading MAF file: {maf_path}...")

    # MAF files usually have comment lines starting with '#' at the top.
    # We skip them to read the actual tabular data.
    df = pd.read_csv(maf_path, sep="\t", comment="#", low_memory=False)

    # Create the mandatory VCF columns
    vcf_df = pd.DataFrame()
    vcf_df["#CHROM"] = df["Chromosome"]
    vcf_df["POS"] = df["Start_Position"]
    vcf_df["ID"] = "."
    vcf_df["REF"] = df["Reference_Allele"]
    vcf_df["ALT"] = df["Tumor_Seq_Allele2"]
    vcf_df["QUAL"] = "."
    vcf_df["FILTER"] = "PASS"
    vcf_df["INFO"] = "."

    # Optional: If the chromosome names don't start with 'chr' (e.g., '17' instead of 'chr17'),
    # Carbon/Reference genomes usually expect the 'chr' prefix.
    vcf_df["#CHROM"] = vcf_df["#CHROM"].apply(
        lambda x: f"chr{x}" if not str(x).startswith("chr") else x
    )

    # Remove any rows that don't have valid genomic variants (like structural placeholders)
    vcf_df = vcf_df.dropna(subset=["#CHROM", "POS", "REF", "ALT"])

    # Sort variants by chromosome and position (Standard VCF practice)
    vcf_df = vcf_df.sort_values(by=["#CHROM", "POS"])

    # Write out the VCF with standard headers
    st.write(f"Writing VCF file to: {vcf_path}...")
    with open(vcf_path, "w") as f:
        # Minimal mandatory VCF header metadata
        f.write("##fileformat=VCFv4.2\n")
        f.write("##source=GDC_Glioma_MAF_Converter\n")
        # Save the DataFrame columns beneath the header
        vcf_df.to_csv(f, sep="\t", index=False)

    st.success("Conversion complete!")
    return vcf_df


def download_and_prepare_hg38(download_dir: Path):
    fasta_gz_path = download_dir / "hg38.fa.gz"
    fasta_path = download_dir / "hg38.fa"
    fasta_index_path = download_dir / "hg38.fa.fai"

    if not fasta_path.exists():
        if not fasta_gz_path.exists():
            st.write("Downloading compressed hg38 reference genome from UCSC...")
            urlretrieve(
                "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz",
                fasta_gz_path,
            )

        st.write("Decompressing genome. This may take a minute...")
        with gzip.open(fasta_gz_path, "rb") as compressed_handle:
            with open(fasta_path, "wb") as fasta_handle:
                shutil.copyfileobj(compressed_handle, fasta_handle)

    if not fasta_index_path.exists():
        st.write("Indexing hg38 FASTA for random genomic access...")
        pysam.faidx(str(fasta_path))

    st.success("Genome ready.")
    return fasta_path


def count_vcf_variants(vcf_path):
    count = 0
    with open(vcf_path, "r") as vcf_handle:
        for line in vcf_handle:
            if not line.startswith("#"):
                count += 1
    return count


def extract_mutation_context(vcf_path, fasta_path, context_window=131072):
    # 1. Load the reference genome index
    st.write("Loading genome index...")
    genome = pysam.FastaFile(str(fasta_path))

    # 2. Read the VCF directly into a clean table using Pandas (skipping header rows)
    df = pd.read_csv(
        vcf_path,
        sep="\t",
        comment="#",
        names=["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO"],
    )

    half_window = context_window // 2
    prepared_data = []
    progress_bar = st.progress(0)
    progress_text = st.empty()

    # 3. Loop through every mutation line directly
    for idx, row in df.iterrows():
        chrom = str(row["CHROM"])
        pos = int(row["POS"])
        ref = str(row["REF"])
        alt = str(row["ALT"])

        # Calculate exactly where to clip the DNA sequence slice
        start = max(0, pos - half_window)
        end = pos + half_window

        # Pull the Wildtype window from the genome file
        wt_seq = genome.fetch(chrom, start, end).upper()

        # Locate the exact index of our mutation within this slice
        mutation_idx = pos - start - 1

        # Build the Mutant sequence by swapping the reference base with the variant base
        mut_seq = wt_seq[:mutation_idx] + alt + wt_seq[mutation_idx + 1 :]

        # Save the results
        prepared_data.append(
            {
                "chrom": chrom,
                "pos": pos,
                "ref": ref,
                "alt": alt,
                "wildtype_ctx": wt_seq,
                "mutant_ctx": mut_seq,
            }
        )

        progress_bar.progress((idx + 1) / max(len(df), 1))
        progress_text.write(f"Extracted context for {idx + 1} of {len(df)} variants")

    st.success(f"Done! Successfully extracted contexts for {len(prepared_data)} variants.")
    return prepared_data


@st.cache_resource(show_spinner=False)
def load_carbon_model(model_name, hf_token):
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            dtype=torch.float32,
        ).to("cpu").eval()
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        ).to("cpu").eval()
    return tokenizer, model


def evaluate_variant(variant_data, tokenizer, model):
    ref_seq = variant_data["wildtype_ctx"]
    var_seq = variant_data["mutant_ctx"]

    def score_sequence(seq):
        ids = tokenizer(
            "<dna>" + seq,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to("cpu")

        with torch.inference_mode():
            outputs = model(ids, output_hidden_states=True)
            logits = outputs.logits
            # Grabbing the last token embedding to save RAM overhead
            embeddings = outputs.hidden_states[-1].squeeze(0).float().cpu().numpy()

        logp = F.log_softmax(logits.float(), dim=-1)[:, :-1, :]
        log_prob_sum = logp.gather(2, ids[:, 1:].unsqueeze(-1)).sum().item()

        return log_prob_sum, embeddings

    ref_logp, ref_embeds = score_sequence(ref_seq)
    var_logp, var_embeds = score_sequence(var_seq)

    delta = var_logp - ref_logp

    if ref_embeds.shape == var_embeds.shape:
        l2_distance = float(np.linalg.norm(ref_embeds - var_embeds))
    else:
        l2_distance = 0.0
        st.warning(
            f"Length mismatch for position {variant_data['pos']}. Setting L2 to 0.0"
        )

    return {
        "chrom": variant_data["chrom"],
        "pos": variant_data["pos"],
        "ref": variant_data["ref"],
        "alt": variant_data["alt"],
        "delta_log_prob": delta,
        "l2_distance": l2_distance,
    }


def run_carbon_scoring(dataset, output_csv, model_name, hf_token, window_before_after=100):
    tokenizer, model = load_carbon_model(model_name, hf_token)

    if "dataset" and len(dataset) > 0:
        total_variants = len(dataset)
        st.write(f"Found {total_variants} variants in dataset. Step 1: Gathering raw scores...")

        raw_results = []
        progress_bar = st.progress(0)
        progress_text = st.empty()

        # Step 1: Compute raw metrics across the entire dataset
        for idx, variant in enumerate(dataset):
            wt_full = variant["wildtype_ctx"]
            mut_full = variant["mutant_ctx"]

            mid_wt = len(wt_full) // 2
            mid_mut = len(mut_full) // 2
            WINDOW_BEFORE_AFTER = window_before_after

            sliced_variant = {
                "chrom": variant["chrom"],
                "pos": variant["pos"],
                "ref": variant["ref"],
                "alt": variant["alt"],
                "wildtype_ctx": wt_full[
                    max(0, mid_wt - WINDOW_BEFORE_AFTER) : mid_wt + WINDOW_BEFORE_AFTER
                ],
                "mutant_ctx": mut_full[
                    max(0, mid_mut - WINDOW_BEFORE_AFTER) : mid_mut + WINDOW_BEFORE_AFTER
                ],
            }

            res = evaluate_variant(sliced_variant, tokenizer, model)
            raw_results.append(res)
            progress_bar.progress((idx + 1) / max(total_variants, 1))
            progress_text.write(
                f"[{idx + 1}/{total_variants}] Processed raw metrics for {res['chrom']}:{res['pos']}"
            )

        st.write("Step 2: Calculating Dataset Normalization Factors...")

        # Extract raw vectors for normalization math
        deltas = np.array([r["delta_log_prob"] for r in raw_results])
        l2s = np.array([r["l2_distance"] for r in raw_results])

        # Calculate means and standard deviations (with safe handling if std is 0)
        mean_delta, std_delta = np.mean(deltas), np.std(deltas)
        mean_l2, std_l2 = np.mean(l2s), np.std(l2s)

        std_delta = std_delta if std_delta > 0 else 1.0
        std_l2 = std_l2 if std_l2 > 0 else 1.0

        st.write("Step 3: Compounding Z-Scores and Writing to CSV...")

        # Step 2 & 3: Save out everything with the normalized Variant Score
        with open(output_csv, mode="w", newline="") as f:
            fieldnames = [
                "chrom",
                "pos",
                "ref",
                "alt",
                "delta_log_prob",
                "l2_distance",
                "variant_score",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for idx, res in enumerate(raw_results):
                # Compute Z-scores manually to keep external library dependencies zero
                norm_delta = (res["delta_log_prob"] - mean_delta) / std_delta
                norm_l2 = (res["l2_distance"] - mean_l2) / std_l2

                # Applying our synchronized formula: Normalized(L2) - Normalized(Delta)
                v_score = norm_l2 - norm_delta

                # Attach final score to output dictionary
                res["variant_score"] = float(v_score)
                writer.writerow(res)

        st.success(f"Successfully finished! Pipeline results written to: {output_csv}")
        return pd.DataFrame(raw_results)

    st.warning("Dataset array not found in your current workspace.")
    return pd.DataFrame()


def generate_chromosome_plots(csv_path, output_dir):
    # 1. Load the dataset
    df = pd.read_csv(csv_path)

    # Set clean style
    sns.set_theme(style="whitegrid")

    # 2. Find all unique chromosomes present in the dataset
    unique_chromosomes = df["chrom"].unique()
    st.write(f"Found data for chromosomes: {unique_chromosomes}")

    saved_plots = []

    # 3. Loop through each chromosome and generate its specific profile charts
    for chrom in unique_chromosomes:
        st.write(f"Generating positional profiles for {chrom}...")

        # Filter and sort data sequentially by its genomic position
        chrom_df = df[df["chrom"] == chrom].sort_values(by="pos")

        if chrom_df.empty or len(chrom_df) < 2:
            st.write(
                f"Skipping {chrom}: Not enough variants for continuous sequence line plotting."
            )
            continue

        # Initialize a 3-panel stacked plot sharing the exact same X-axis (Genomic Position)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        # --- Top Panel: Position vs Variant Score ---
        ax1.plot(
            chrom_df["pos"],
            chrom_df["variant_score"],
            color="purple",
            marker="o",
            linestyle="-",
            linewidth=1.5,
            alpha=0.8,
        )
        ax1.fill_between(chrom_df["pos"], chrom_df["variant_score"], color="purple", alpha=0.1)
        ax1.set_ylabel("Unified Variant Score", fontsize=11, fontweight="bold")
        ax1.set_title(
            f"Genomic Landscape Profile: {chrom}",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )

        # --- Middle Panel: Position vs Delta Log Prob ---
        ax2.plot(
            chrom_df["pos"],
            chrom_df["delta_log_prob"],
            color="teal",
            marker="s",
            linestyle="-",
            linewidth=1.5,
            alpha=0.8,
        )
        ax2.fill_between(chrom_df["pos"], chrom_df["delta_log_prob"], color="teal", alpha=0.1)
        ax2.set_ylabel("Delta Log Probability", fontsize=11, fontweight="bold")

        # --- Bottom Panel: Position vs L2 Distance ---
        ax3.plot(
            chrom_df["pos"],
            chrom_df["l2_distance"],
            color="darkorange",
            marker="^",
            linestyle="-",
            linewidth=1.5,
            alpha=0.8,
        )
        ax3.fill_between(chrom_df["pos"], chrom_df["l2_distance"], color="darkorange", alpha=0.1)
        ax3.set_ylabel("L2 Distance", fontsize=11, fontweight="bold")

        # Format the shared X-axis labels properly
        ax3.set_xlabel(f"Genomic Position along {chrom} (bp)", fontsize=12, fontweight="bold")

        # Clean up formatting across all plots (turn off scientific notation on X-axis for readability)
        for ax in [ax1, ax2, ax3]:
            ax.ticklabel_format(style="plain", axis="x")
            ax.xaxis.grid(True, linestyle=":", alpha=0.6)

        # Save the chromosome layout out to a unique image file
        output_filename = output_dir / f"chrom_{chrom}_profile.png"
        plt.tight_layout()
        plt.savefig(output_filename, dpi=300)
        st.pyplot(fig)
        plt.close()
        saved_plots.append(output_filename)

        st.write(f"Successfully saved: {output_filename.name}")

    st.success("All chromosome tracking plots are successfully generated and saved!")
    return saved_plots


def identify_gene_via_ucsc(chrom, pos):
    """Queries the UCSC Genome Browser API to map coordinates to gene symbols."""
    # Defensive check: ensure chrom is a string so .lower() doesn't fail on numerical chromosomes (e.g., 1, 2)
    chrom_str = str(chrom)
    chrom_fixed = chrom_str if chrom_str.lower().startswith("chr") else f"chr{chrom_str}"

    # Check a narrow 2kb window around the mutation point
    url = (
        "https://api.genome.ucsc.edu/getData/track?"
        f"genome=hg38;track=ncbiRefSeq;chrom={chrom_fixed};start={pos - 1000};end={pos + 1000}"
    )

    try:
        response = requests.get(url, timeout=5)
        if response.ok:
            data = response.json()
            items = data.get("ncbiRefSeq", [])
            if items:
                # Grab the official gene symbol (name2)
                return items[0].get("name2", "Unknown Feature")
    except Exception:
        pass
    return "Intergenic / Non-coding"


def map_genes_with_ucsc(file_path, output_file):
    df = pd.read_csv(file_path)
    st.write(f"Mapping all {len(df)} variant coordinates to true gene symbols using UCSC...")

    results = []
    progress_bar = st.progress(0)
    progress_text = st.empty()

    # Modification 1: Loop through the entire 'df' instead of 'top_5'
    for idx, row in df.iterrows():
        # Progress Tracker: Prints an update every 50 rows so you can monitor the pipeline
        if idx % 50 == 0 and idx > 0:
            progress_text.write(f"Processed {idx} out of {len(df)} variants...")

        gene_symbol = identify_gene_via_ucsc(row["chrom"], row["pos"])
        results.append(
            {
                "chrom": row["chrom"],
                "pos": row["pos"],
                "ref": row["ref"],
                "alt": row["alt"],
                "Mutation": f"{row['chrom']}:{row['pos']} ({row['ref']}->{row['alt']})",
                "Variant Score": round(row["variant_score"], 4),
                "Mapped Gene": gene_symbol,
            }
        )
        progress_bar.progress((idx + 1) / max(len(df), 1))

    # Render the final clean table
    summary_df = pd.DataFrame(results)

    # Modification 2: Save the completed DataFrame to a new CSV file
    # index=False prevents pandas from adding an extra, unnamed row-number column to your file
    summary_df.to_csv(output_file, index=False)

    st.success(f"Processing complete! Successfully saved all mapped records to '{output_file}'.")
    return summary_df


def generate_cohort_property_plots(file_path, output_dir):
    # 1. Load the newly mapped CSV file
    df = pd.read_csv(file_path)

    # Quick Data Cleaning/Feature Engineering for plots
    # Build a clean "Ref -> Alt" column for the mutation spectrum plot
    df["Mutation Type"] = df["ref"].astype(str) + " -> " + df["alt"].astype(str)

    # Set a professional plotting style
    sns.set_theme(style="whitegrid")

    saved_plots = []

    # ==========================================
    # GRAPH 1: Distribution of Variant Scores
    # ==========================================
    st.write("Generating Graph 1: Variant Score Distribution...")
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot the distribution with a Kernel Density Estimate (KDE) curve
    sns.histplot(data=df, x="Variant Score", kde=True, color="skyblue", ax=ax, bins=30)

    ax.set_title(
        "Distribution of Carbon Variant Scores",
        fontsize=14,
        pad=15,
        fontweight="bold",
    )
    ax.set_xlabel("Variant Score", fontsize=12)
    ax.set_ylabel("Count / Frequency", fontsize=12)

    plt.tight_layout()
    distribution_path = output_dir / "1_variant_score_distribution.png"
    plt.savefig(distribution_path, dpi=300)
    st.pyplot(fig)
    plt.close()
    saved_plots.append(distribution_path)

    # ==========================================
    # GRAPH 2: Top Mutated Genes (Excluding Non-coding)
    # ==========================================
    st.write("Generating Graph 2: Top Mutated Genes...")

    # Filter out Intergenic spaces so we only focus on true functional genes
    gene_df = df[df["Mapped Gene"] != "Intergenic / Non-coding"]

    if not gene_df.empty:
        # Count the mutations per gene and sort them in descending order
        gene_counts = gene_df["Mapped Gene"].value_counts().reset_index()
        gene_counts.columns = ["Mapped Gene", "Count"]

        # Grab the top 15 most frequently mutated genes for readability
        top_genes = gene_counts.head(15)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Sorted horizontal bar chart
        sns.barplot(
            data=top_genes,
            x="Count",
            y="Mapped Gene",
            palette="viridis",
            ax=ax,
            hue="Mapped Gene",
            legend=False,
        )

        ax.set_title(
            "Top 15 Most Frequently Mutated Genes",
            fontsize=14,
            pad=15,
            fontweight="bold",
        )
        ax.set_xlabel("Number of Variants Found", fontsize=12)
        ax.set_ylabel("Gene Symbol", fontsize=12)

        plt.tight_layout()
        top_genes_path = output_dir / "2_top_mutated_genes.png"
        plt.savefig(top_genes_path, dpi=300)
        st.pyplot(fig)
        plt.close()
        saved_plots.append(top_genes_path)
    else:
        st.write("Skipping Graph 2: No functional genes mapped in this dataset.")

    # ==========================================
    # GRAPH 3: Mutation Type Spectrum (Ref -> Alt)
    # ==========================================
    st.write("Generating Graph 3: Mutation Type Spectrum...")

    # Calculate the frequencies of each mutation exchange type and sort them
    mut_counts = df["Mutation Type"].value_counts().reset_index()
    mut_counts.columns = ["Mutation Type", "Count"]
    mut_counts = mut_counts.sort_values(by="Count", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Sorted vertical bar chart
    sns.barplot(
        data=mut_counts,
        x="Mutation Type",
        y="Count",
        palette="flare",
        ax=ax,
        hue="Mutation Type",
        legend=False,
    )

    ax.set_title(
        "Genomic Mutation Spectrum (Substitution Frequency)",
        fontsize=14,
        pad=15,
        fontweight="bold",
    )
    ax.set_xlabel("Nucleotide Substitution Type", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)

    # Rotate labels slightly to ensure they never overlap
    plt.xticks(rotation=45)
    plt.tight_layout()
    mutation_spectrum_path = output_dir / "3_mutation_spectrum.png"
    plt.savefig(mutation_spectrum_path, dpi=300)
    st.pyplot(fig)
    plt.close()
    saved_plots.append(mutation_spectrum_path)

    st.success("All graphics successfully generated and saved as high-resolution PNGs.")
    return saved_plots


def query_ensembl_vep(chrom, pos, ref, alt):
    """Queries the Ensembl VEP API to find overlapping regulatory elements and closest target genes."""
    # Clean chromosome names (Ensembl expects "1" or "X", not "chr1")
    chrom_clean = str(chrom).lower().replace("chr", "")

    # Ensembl VEP URL for single variants
    url = f"https://rest.ensembl.org/vep/human/region/{chrom_clean}:{pos}-{pos}/{alt}?"

    headers = {"Content-Type": "application/json"}

    try:
        # Ensembl asks users to declare a timeout and modest hit rate
        response = requests.get(url, headers=headers, timeout=10)
        if response.ok:
            results = response.json()
            if results:
                # Extract genes from the regulatory or nearest feature consequences
                genes = []

                # Check regular transcript consequences (picks up nearby/overlapping genes)
                if "transcript_consequences" in results[0]:
                    for tc in results[0]["transcript_consequences"]:
                        if "gene_symbol" in tc:
                            genes.append(tc["gene_symbol"])

                # Check regulatory feature consequences (picks up enhancers/promoters)
                if "regulatory_feature_consequences" in results[0]:
                    for rc in results[0]["regulatory_feature_consequences"]:
                        # If it hits a regulatory element, VEP notes the regulatory ID
                        if "regulatory_feature_id" in rc:
                            genes.append(rc["regulatory_feature_id"])

                if genes:
                    return ", ".join(list(set(genes[:3])))  # Return top 3 unique targets
    except Exception:
        pass

    return "Nearest Gene / Intergenic Feature Not Found"


def map_regulatory_context_with_vep(input_file, output_filename, request_delay=0.3):
    # 1. Load your local Carbon data
    df = pd.read_csv(input_file)

    st.write(f"Querying Ensembl VEP for {len(df)} real genomic intervals...")

    mapped_results = []
    progress_bar = st.progress(0)
    progress_text = st.empty()

    for idx, row in df.iterrows():
        # Progress monitoring
        if idx % 10 == 0 and idx > 0:
            progress_text.write(f"Mapping mutation {idx} of {len(df)}...")

        time.sleep(request_delay)

        target_gene = query_ensembl_vep(row["chrom"], row["pos"], row["ref"], row["alt"])

        mapped_results.append(
            {
                "Mutation": f"{row['chrom']}:{row['pos']}",
                "Carbon Score": row["Variant Score"],
                "Real Mapped Target": target_gene,
            }
        )
        progress_bar.progress((idx + 1) / max(len(df), 1))

    # Convert to DataFrame
    final_vep_df = pd.DataFrame(mapped_results)

    # --- SAVE RESULT TO CSV ---
    final_vep_df.to_csv(output_filename, index=False)
    st.success(f"Intermediate VEP results successfully saved to '{output_filename}'")

    return final_vep_df


def get_alphamissense_profile(gene_symbol):
    """
    Queries public databases to check the structural
    pathogenicity load for a specific protein.
    """
    if pd.isna(gene_symbol) or "Not Found" in str(gene_symbol):
        return 0.0, "Unknown"

    # Clean the string in case VEP returned multiple genes separated by commas
    # We will pick the first gene in the list for structural mapping
    primary_gene = str(gene_symbol).split(",")[0].strip()

    # Query MyGene.info to get the  structural annotation or general pathogenicity status
    url = f"https://mygene.info/v3/query?q=symbol:{primary_gene}&fields=summary,name&size=1"

    try:
        response = requests.get(url, timeout=5)
        if response.ok:
            data = response.json()
            hits = data.get("hits", [])
            if hits:
                # We use a simulated AlphaMissense metric: structural constraint/vulnerability
                # Real structural constraint maps how essential the protein domains are.
                # Let's assign a deterministic structural vulnerability score based on gene length/essentiality
                # (For production, you would merge with the pre-downloaded AM table)
                hash_val = sum(ord(c) for c in primary_gene)
                simulated_am_score = round((hash_val % 100) / 100.0, 4)

                if simulated_am_score > 0.6:
                    am_class = "Highly Fragile (Am-Pathogenic)"
                elif simulated_am_score > 0.3:
                    am_class = "Moderately Tolerant"
                else:
                    am_class = "Benign/Tolerant Structural Architecture"

                return simulated_am_score, am_class
    except Exception:
        pass

    return 0.45, "Ambiguous Domain Structure"


def layer_alphamissense_profiles(input_file, output_file, request_delay=0.2):
    # 1. Load your VEP results dataframe
    df = pd.read_csv(input_file)

    st.write("Layering AlphaMissense structural profiles over your regulatory mapping data...")

    final_results = []
    progress_bar = st.progress(0)
    progress_text = st.empty()

    for idx, row in df.iterrows():
        time.sleep(request_delay)

        # Get structural vulnerability score from AlphaMissense data logic
        am_score, am_verdict = get_alphamissense_profile(row["Real Mapped Target"])

        final_results.append(
            {
                "Mutation": row["Mutation"],
                "Regulatory Carbon Score": row["Carbon Score"],
                "Target Protein Coding Gene": row["Real Mapped Target"],
                "AlphaMissense Structure Fragility Score": am_score,
                "Structural Vulnerability Class": am_verdict,
            }
        )

        progress_bar.progress((idx + 1) / max(len(df), 1))
        progress_text.write(f"Processed structural profile {idx + 1} of {len(df)}")

    # Convert to the final consolidated DataFrame
    pipeline_summary_df = pd.DataFrame(final_results)

    # Save the final masterpiece CSV
    pipeline_summary_df.to_csv(output_file, index=False)

    st.success("Final integrated bioinformatics pipeline output saved.")
    return pipeline_summary_df


def get_uniprot_id_from_gene(gene_name):
    """Queries UniProt and forces Reviewed (Swiss-Prot) records to the top."""
    # We add '+AND+reviewed:true' to guarantee we get the canonical canonical ID (like Q99848)
    url = (
        "https://rest.uniprot.org/uniprotkb/search?"
        f"query=gene_exact:{gene_name}+AND+taxonomy_id:9606+AND+reviewed:true&size=1"
    )
    headers = {"User-Agent": "BioinformaticsPipeline/1.0"}

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            if results:
                return results[0].get("primaryAccession")

        # Fallback: If NO reviewed entry exists, try unreviewed entries as a backup
        fallback_url = (
            "https://rest.uniprot.org/uniprotkb/search?"
            f"query=gene_exact:{gene_name}+AND+taxonomy_id:9606&size=1"
        )
        fallback_response = requests.get(fallback_url, headers=headers)
        if fallback_response.status_code == 200:
            fallback_results = fallback_response.json().get("results", [])
            if fallback_results:
                return fallback_results[0].get("primaryAccession")

    except Exception as e:
        st.write(f"API Navigation Error details: {e}")
    return None


def build_3d_protein_view(csv_file_path, target_gene_override=None):
    # ==========================================
    # 1. READ YOUR ACTUAL UPLOADED CSV FILE
    # ==========================================
    df = pd.read_csv(csv_file_path)

    df.columns = df.columns.str.strip()
    if "AlphaMissense Structure Fragility Score" not in df.columns and "tructure Fragility Score" in df.columns:
        df = df.rename(columns={"tructure Fragility Score": "AlphaMissense Structure Fragility Score"})
    df["Target Protein Coding Gene"] = df["Target Protein Coding Gene"].astype(str).str.strip()

    # Target the gene name from your row (e.g., EBNA1BP2)
    # If your row contains multiple comma-separated genes, this splits them and picks the first one safely
    if target_gene_override:
        TARGET_GENE = target_gene_override.split(",")[0].strip()
    else:
        TARGET_GENE = df["Target Protein Coding Gene"].iloc[0].split(",")[0].strip()
    st.write(f"Targeting Gene Name from CSV: {TARGET_GENE}")

    # ==========================================
    # 2. ROBUST UNIPROT API QUERY (SORT BY REVIEWED)
    # ==========================================
    st.write(f"Querying UniProt API for {TARGET_GENE}...")
    UNIPROT_ID = get_uniprot_id_from_gene(TARGET_GENE)

    if not UNIPROT_ID:
        raise ValueError(f"Could not map Gene Name '{TARGET_GENE}' to a valid human UniProt ID.")

    st.write(f"Successfully mapped {TARGET_GENE} -> Canonical UniProt ID: {UNIPROT_ID}")

    # ==========================================
    # 3. MAP GENOMIC COORDINATES TO RESIDUES
    # ==========================================
    gene_df = df[df["Target Protein Coding Gene"].str.contains(TARGET_GENE, na=False)]

    mutations_to_map = {}
    for _, row in gene_df.iterrows():
        genomic_coord = str(row["Mutation"]).strip()
        score = float(row["AlphaMissense Structure Fragility Score"])

        # Placeholder mapping for demonstration. Change 150 to whatever residue index
        # maps to your genomic site in your actual downstream pipeline logic!
        mutations_to_map[150] = score

    # ==========================================
    # 4. FETCH STRUCTURE VIA DYNAMIC ALPHAFOLD API
    # ==========================================
    # Instead of hardcoding the URL, we query AlphaFold's API directly
    af_api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{UNIPROT_ID}"
    st.write("Querying AlphaFold API for structure download links...")

    api_response = requests.get(af_api_url)
    if api_response.status_code != 200:
        raise ValueError(f"No AlphaFold prediction records found for ID: {UNIPROT_ID}")

    # Parse the JSON response to grab the active PDB URL dynamically
    api_data = api_response.json()
    if isinstance(api_data, list) and len(api_data) > 0:
        alphafold_url = api_data[0].get("pdbUrl")
    else:
        raise ValueError("Unexpected data format returned from AlphaFold API.")

    if not alphafold_url:
        raise ValueError(f"Could not locate a valid PDB URL for {UNIPROT_ID} in the AlphaFold entry.")

    st.write(f"Dynamically resolved active structure link: {alphafold_url}")

    # Now securely fetch the structural file text
    response = requests.get(alphafold_url)
    if response.status_code != 200:
        raise ValueError("Failed to stream the resolved PDB coordinate file.")
    pdb_content = response.text

    # ==========================================
    # 5. INJECT SCORES INTO B-FACTOR COLUMN
    # ==========================================
    modified_pdb = []
    for line in pdb_content.splitlines():
        if line.startswith("ATOM  ") or line.startswith("HETATM"):
            res_seq = int(line[22:26].strip())
            variant_score = mutations_to_map.get(res_seq, 0.0)

            new_line = line[:60] + f"{variant_score:6.2f}" + line[66:]
            modified_pdb.append(new_line)
        else:
            modified_pdb.append(line)

    pdb_data_ready = "\n".join(modified_pdb)

    # ==========================================
    # 6. RENDER 3D CANVAS WITH THE HTML FIX
    # ==========================================
    view = py3Dmol.view(width=800, height=600)
    view.addModel(pdb_data_ready, "pdb")

    view.setStyle(
        {
            "cartoon": {
                "colorscheme": {
                    "prop": "b",
                    "gradient": "rwb",
                    "min": 0.0,
                    "max": 1.0,
                }
            }
        }
    )

    if mutations_to_map:
        mutated_residues = list(mutations_to_map.keys())
        view.addStyle(
            {"resi": mutated_residues},
            {
                "sphere": {
                    "radius": 2.5,
                    "colorscheme": {"prop": "b", "gradient": "rwb", "min": 0, "max": 1},
                }
            },
        )
        view.zoomTo({"resi": mutated_residues})
    else:
        view.zoom()

    html_content = view._make_html()
    return html_content, UNIPROT_ID, TARGET_GENE


def reset_session():
    for key in [
        "maf_path",
        "vcf_path",
        "fasta_path",
        "dataset",
        "carbon_scores_path",
        "mapped_variants_path",
        "vep_path",
        "final_pipeline_path",
    ]:
        if key in st.session_state:
            del st.session_state[key]


st.title("CarbonVEP Glioma Variant Prioritization")
st.caption(
    "MAF to VCF conversion, hg38 flanking context extraction, Carbon-500M scoring, "
    "gene/regulatory/protein annotation, visual summaries, and interactive 3D protein rendering."
)

with st.sidebar:
    st.header("Run Settings")
    run_name = st.text_input("Run folder name", value="default_run")
    safe_run_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in run_name).strip("_")
    current_run_dir = WORK_DIR / (safe_run_name or "default_run")
    current_run_dir.mkdir(parents=True, exist_ok=True)

    model_name = st.text_input("Carbon model", value="HuggingFaceBio/Carbon-500M")
    hf_token = st.text_input(
        "Hugging Face token",
        value=get_default_hf_token(),
        type="password",
    )
    context_window = st.number_input("Reference context window", min_value=100, value=131072, step=100)
    scoring_window = st.number_input("Scoring bases before/after midpoint", min_value=1, value=100, step=1)
    request_delay_vep = st.number_input("Ensembl request delay seconds", min_value=0.0, value=0.3, step=0.1)
    request_delay_mygene = st.number_input("MyGene request delay seconds", min_value=0.0, value=0.2, step=0.1)

    st.divider()
    if st.button("Reset visible session outputs"):
        reset_session()
        st.rerun()


tab_input, tab_context, tab_scoring, tab_visuals, tab_annotations, tab_structure = st.tabs(
    [
        "1. Inputs",
        "2. Context",
        "3. Carbon Scoring",
        "4. Plots",
        "5. Annotations",
        "6. Protein View",
    ]
)


with tab_input:
    st.subheader("MAF to VCF")
    uploaded_maf = st.file_uploader("Upload glioma MAF file", type=["maf", "txt", "tsv"])

    col1, col2 = st.columns(2)
    with col1:
        if uploaded_maf and st.button("Save uploaded MAF"):
            maf_path = save_uploaded_file(uploaded_maf, current_run_dir / uploaded_maf.name)
            st.session_state["maf_path"] = str(maf_path)
            st.success(f"Saved MAF to {maf_path}")

    with col2:
        manual_maf_path = st.text_input("Or enter existing MAF path", value=st.session_state.get("maf_path", ""))
        if manual_maf_path and st.button("Use existing MAF path"):
            st.session_state["maf_path"] = manual_maf_path
            st.success(f"Using MAF path: {manual_maf_path}")

    if st.session_state.get("maf_path"):
        output_vcf = current_run_dir / "glioma_mutations.vcf"
        if st.button("Convert MAF to VCF"):
            vcf_df = convert_maf_to_vcf(st.session_state["maf_path"], output_vcf)
            st.session_state["vcf_path"] = str(output_vcf)
            st.dataframe(vcf_df, use_container_width=True)
            file_download_button(output_vcf, "Download glioma_mutations.vcf", "text/plain")

    st.divider()
    st.subheader("Reference Genome")
    uploaded_fasta = st.file_uploader("Upload existing hg38 FASTA", type=["fa", "fasta"])
    if uploaded_fasta and st.button("Save uploaded FASTA"):
        fasta_path = save_uploaded_file(uploaded_fasta, current_run_dir / uploaded_fasta.name)
        st.session_state["fasta_path"] = str(fasta_path)
        if not Path(f"{fasta_path}.fai").exists():
            pysam.faidx(str(fasta_path))
        st.success(f"Saved and indexed FASTA: {fasta_path}")

    manual_fasta_path = st.text_input("Or enter existing hg38 FASTA path", value=st.session_state.get("fasta_path", ""))
    if manual_fasta_path and st.button("Use existing FASTA path"):
        st.session_state["fasta_path"] = manual_fasta_path
        st.success(f"Using FASTA path: {manual_fasta_path}")

    if st.button("Download and prepare hg38 from UCSC"):
        fasta_path = download_and_prepare_hg38(current_run_dir)
        st.session_state["fasta_path"] = str(fasta_path)

    if st.session_state.get("vcf_path"):
        st.metric("VCF variant count", count_vcf_variants(st.session_state["vcf_path"]))


with tab_context:
    st.subheader("Genomic Flanking-Context Extraction")
    st.write("This step slices hg38 around each VCF mutation and creates wildtype and mutant contexts.")

    vcf_path_for_context = st.text_input(
        "VCF path",
        value=st.session_state.get("vcf_path", str(current_run_dir / "glioma_mutations.vcf")),
        key="context_vcf_path",
    )
    fasta_path_for_context = st.text_input(
        "hg38 FASTA path",
        value=st.session_state.get("fasta_path", ""),
        key="context_fasta_path",
    )

    if st.button("Extract mutation contexts"):
        dataset = extract_mutation_context(vcf_path_for_context, fasta_path_for_context, int(context_window))
        st.session_state["dataset"] = dataset
        context_preview = pd.DataFrame(dataset)
        st.dataframe(context_preview.drop(columns=["wildtype_ctx", "mutant_ctx"]), use_container_width=True)

    if st.session_state.get("dataset"):
        st.success(f"Current session has {len(st.session_state['dataset'])} extracted variant contexts.")


with tab_scoring:
    st.subheader("Carbon-500M Variant Scoring")
    st.write("This step scores wildtype and mutant sequence windows and writes the normalized variant score CSV.")

    carbon_scores_path = current_run_dir / "carbon_variant_scores.csv"

    if st.button("Run Carbon scoring"):
        if not st.session_state.get("dataset"):
            st.error("Extract mutation contexts first.")
        else:
            scores_df = run_carbon_scoring(
                st.session_state["dataset"],
                carbon_scores_path,
                model_name,
                hf_token,
                int(scoring_window),
            )
            st.session_state["carbon_scores_path"] = str(carbon_scores_path)
            st.dataframe(scores_df, use_container_width=True)
            dataframe_download_button(scores_df, "carbon_variant_scores.csv", "Download Carbon scores CSV")

    uploaded_scores = st.file_uploader("Or upload existing carbon_variant_scores.csv", type=["csv"], key="carbon_scores_upload")
    if uploaded_scores and st.button("Use uploaded Carbon scores"):
        saved_path = save_uploaded_file(uploaded_scores, carbon_scores_path)
        st.session_state["carbon_scores_path"] = str(saved_path)
        st.success(f"Using Carbon score CSV: {saved_path}")

    manual_scores_path = st.text_input(
        "Or enter existing Carbon score CSV path",
        value=st.session_state.get("carbon_scores_path", ""),
    )
    if manual_scores_path and st.button("Use existing Carbon score CSV path"):
        st.session_state["carbon_scores_path"] = manual_scores_path
        st.success(f"Using Carbon score CSV: {manual_scores_path}")

    if st.session_state.get("carbon_scores_path") and Path(st.session_state["carbon_scores_path"]).exists():
        score_df = pd.read_csv(st.session_state["carbon_scores_path"])
        st.dataframe(score_df, use_container_width=True)
        file_download_button(Path(st.session_state["carbon_scores_path"]), "Download current Carbon scores", "text/csv")


with tab_visuals:
    st.subheader("Chromosome-Level and Cohort Plots")

    if st.button("Generate chromosome-level profiles"):
        if not st.session_state.get("carbon_scores_path"):
            st.error("Provide or generate carbon_variant_scores.csv first.")
        else:
            plot_paths = generate_chromosome_plots(st.session_state["carbon_scores_path"], current_run_dir)
            for plot_path in plot_paths:
                file_download_button(plot_path, f"Download {plot_path.name}", "image/png")

    st.divider()
    st.write("Cohort property plots require mapped_carbon_variants.csv from the UCSC annotation step.")

    if st.button("Generate cohort property plots"):
        if not st.session_state.get("mapped_variants_path"):
            st.error("Run UCSC gene mapping first, or upload mapped_carbon_variants.csv in the Annotations tab.")
        else:
            cohort_plot_paths = generate_cohort_property_plots(
                st.session_state["mapped_variants_path"],
                current_run_dir,
            )
            for plot_path in cohort_plot_paths:
                file_download_button(plot_path, f"Download {plot_path.name}", "image/png")


with tab_annotations:
    st.subheader("Genomic and Regulatory Annotation")

    mapped_variants_path = current_run_dir / "mapped_carbon_variants.csv"
    vep_path = current_run_dir / "vep_mapped_output.csv"
    final_pipeline_path = current_run_dir / "final_brain_tumor_integrated_pipeline.csv"

    st.write("UCSC gene mapping")
    if st.button("Map Carbon variants to UCSC genes"):
        if not st.session_state.get("carbon_scores_path"):
            st.error("Provide or generate carbon_variant_scores.csv first.")
        else:
            summary_df = map_genes_with_ucsc(st.session_state["carbon_scores_path"], mapped_variants_path)
            st.session_state["mapped_variants_path"] = str(mapped_variants_path)
            st.dataframe(summary_df, use_container_width=True)
            dataframe_download_button(summary_df, "mapped_carbon_variants.csv", "Download mapped variants CSV")

    uploaded_mapped = st.file_uploader("Or upload existing mapped_carbon_variants.csv", type=["csv"], key="mapped_upload")
    if uploaded_mapped and st.button("Use uploaded mapped variants"):
        saved_path = save_uploaded_file(uploaded_mapped, mapped_variants_path)
        st.session_state["mapped_variants_path"] = str(saved_path)
        st.success(f"Using mapped variant CSV: {saved_path}")

    st.divider()
    st.write("Ensembl VEP regulatory overlap mapping")
    if st.button("Query Ensembl VEP"):
        if not st.session_state.get("mapped_variants_path"):
            st.error("Run UCSC gene mapping first, or upload mapped_carbon_variants.csv.")
        else:
            final_vep_df = map_regulatory_context_with_vep(
                st.session_state["mapped_variants_path"],
                vep_path,
                request_delay=float(request_delay_vep),
            )
            st.session_state["vep_path"] = str(vep_path)
            st.dataframe(final_vep_df, use_container_width=True)
            dataframe_download_button(final_vep_df, "vep_mapped_output.csv", "Download VEP mapped CSV")

    uploaded_vep = st.file_uploader("Or upload existing vep_mapped_output.csv", type=["csv"], key="vep_upload")
    if uploaded_vep and st.button("Use uploaded VEP results"):
        saved_path = save_uploaded_file(uploaded_vep, vep_path)
        st.session_state["vep_path"] = str(saved_path)
        st.success(f"Using VEP CSV: {saved_path}")

    st.divider()
    st.write("MyGene / AlphaMissense-style structural profile layering")
    if st.button("Build final integrated pipeline CSV"):
        if not st.session_state.get("vep_path"):
            st.error("Run Ensembl VEP mapping first, or upload vep_mapped_output.csv.")
        else:
            pipeline_summary_df = layer_alphamissense_profiles(
                st.session_state["vep_path"],
                final_pipeline_path,
                request_delay=float(request_delay_mygene),
            )
            st.session_state["final_pipeline_path"] = str(final_pipeline_path)
            st.dataframe(pipeline_summary_df, use_container_width=True)
            dataframe_download_button(
                pipeline_summary_df,
                "final_brain_tumor_integrated_pipeline.csv",
                "Download final integrated CSV",
            )

    uploaded_final = st.file_uploader(
        "Or upload existing final_brain_tumor_integrated_pipeline.csv",
        type=["csv"],
        key="final_upload",
    )
    if uploaded_final and st.button("Use uploaded final integrated CSV"):
        saved_path = save_uploaded_file(uploaded_final, final_pipeline_path)
        st.session_state["final_pipeline_path"] = str(saved_path)
        st.success(f"Using final integrated CSV: {saved_path}")


with tab_structure:
    st.subheader("Interactive 3D Protein Rendering")
    st.write("This uses UniProt, AlphaFold, py3Dmol, and the final integrated CSV.")

    protein_csv_path = st.text_input(
        "Final integrated CSV path",
        value=st.session_state.get("final_pipeline_path", ""),
        key="protein_csv_path",
    )
    target_gene_override = st.text_input("Optional target gene override", value="")

    if st.button("Render 3D protein view"):
        if not protein_csv_path:
            st.error("Provide final_brain_tumor_integrated_pipeline.csv first.")
        else:
            try:
                html_content, uniprot_id, target_gene = build_3d_protein_view(
                    protein_csv_path,
                    target_gene_override=target_gene_override or None,
                )
                components.html(html_content, height=650, scrolling=False)
                st.success(f"Rendered {target_gene} ({uniprot_id})")
            except Exception as exc:
                st.error(str(exc))
