# -*- coding: utf-8 -*-
"""Faithful Streamlit wrapper for CarbonVEP_v4.ipynb.

The notebook is the source of truth. This app changes only the interface:
one MAF upload and one button execute the same disk-based pipeline.
"""

import csv
import gzip
import os
import shutil
import tempfile
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlretrieve

APP_DIR = Path(__file__).resolve().parent


def resolve_writable_run_dir():
    """Pick a writable run directory on local machines and Streamlit Cloud."""
    candidates = []
    env_run_dir = os.environ.get("CARBONVEP_RUN_DIR")
    if env_run_dir:
        candidates.append(Path(env_run_dir))
    candidates.extend(
        [
            APP_DIR / "carbonvep_run",
            Path.cwd() / "carbonvep_run",
            Path(tempfile.gettempdir()) / "carbonvep_run",
        ]
    )

    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write_test"
            probe.write_text("ok")
            probe.unlink(missing_ok=True)
            return candidate
        except OSError:
            continue

    raise PermissionError(
        "CarbonVEP could not create a writable run directory. "
        "Set CARBONVEP_RUN_DIR to a writable path."
    )


RUN_DIR = resolve_writable_run_dir()
MPL_CACHE_DIR = RUN_DIR / "matplotlib_cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

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
from Bio.PDB import MMCIFIO, MMCIFParser, PDBList, Select
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "HuggingFaceBio/Carbon-500M"
PREVIEW_ROWS = 25
GOOGLE_SHEETS_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
GOOGLE_SHEETS_HEADER = [
    "Timestamp",
    "Full Name",
    "Email",
    "Institution",
    "Role / Grade",
    "Session ID",
    "Status",
]

MAF_PATH = RUN_DIR / "input.maf"
VCF_PATH = RUN_DIR / "glioma_mutations.vcf"
FASTA_GZ_PATH = RUN_DIR / "hg38.fa.gz"
FASTA_PATH = RUN_DIR / "hg38.fa"
CARBON_CSV = RUN_DIR / "carbon_variant_scores.csv"
MAPPED_CSV = RUN_DIR / "mapped_carbon_variants.csv"
VEP_CSV = RUN_DIR / "vep_mapped_output.csv"
PDB_DIR = RUN_DIR / "pdb_files"
PDB_DIR.mkdir(exist_ok=True)
ZIP_PATH = RUN_DIR / "carbonvep_outputs.zip"


st.set_page_config(page_title="CarbonVEP", page_icon="DNA", layout="wide")


def app_log(message):
    st.session_state.setdefault("log_messages", []).append(str(message))


def show_log():
    with st.expander("Run Log", expanded=False):
        st.text("\n".join(st.session_state.get("log_messages", [])))


def preview_dataframe(path, label):
    if Path(path).exists():
        df = pd.read_csv(path)
        st.write(f"{label}: `{Path(path).name}` ({len(df)} rows x {len(df.columns)} columns)")
        st.dataframe(df.head(PREVIEW_ROWS), use_container_width=True)
        return df
    return None


def get_hf_token():
    try:
        token = st.secrets.get("HF_TOKEN", "")
    except Exception:
        token = ""
    return token or os.environ.get("HF_TOKEN", "")


def normalize_service_account_info(credentials_info):
    credentials_info = dict(credentials_info)
    required_fields = [
        "type",
        "project_id",
        "private_key_id",
        "private_key",
        "client_email",
        "client_id",
        "auth_uri",
        "token_uri",
        "auth_provider_x509_cert_url",
        "client_x509_cert_url",
    ]

    for field in required_fields:
        if field not in credentials_info or str(credentials_info.get(field, "")).strip() == "":
            raise RuntimeError(f"Google service account secret is missing required field: {field}")

    private_key = str(credentials_info["private_key"]).strip()
    if (private_key.startswith('"') and private_key.endswith('"')) or (
        private_key.startswith("'") and private_key.endswith("'")
    ):
        private_key = private_key[1:-1].strip()

    private_key = private_key.replace("\r\n", "\n").replace("\r", "\n")
    private_key = private_key.replace("\\n", "\n")
    private_key = private_key.strip()
    private_key = private_key.rstrip("\n") + "\n"

    key_body = private_key.strip()
    starts_correctly = key_body.startswith("-----BEGIN PRIVATE KEY-----")
    ends_correctly = key_body.endswith("-----END PRIVATE KEY-----")
    literal_newlines_remain = "\\n" in private_key

    st.write(
        {
            "private_key_starts_correctly": starts_correctly,
            "private_key_ends_correctly": ends_correctly,
            "private_key_first_30": private_key[:30],
            "private_key_last_30": private_key[-30:],
            "private_key_length": len(private_key),
        }
    )

    if not starts_correctly:
        raise RuntimeError("Google service account private_key must begin with -----BEGIN PRIVATE KEY-----")
    if not ends_correctly:
        raise RuntimeError("Google service account private_key must end with -----END PRIVATE KEY-----")
    if literal_newlines_remain:
        raise RuntimeError("Google service account private_key still contains literal '\\n' text after normalization.")

    credentials_info["private_key"] = private_key
    return credentials_info


def get_google_sheet():
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Google Sheets logging dependencies are missing. Add 'gspread' and "
            "'google-auth' to the requirements.txt file at the root of your "
            "Streamlit Cloud GitHub repo, then redeploy."
        ) from exc

    if "gcp_service_account" not in st.secrets:
        raise RuntimeError("Missing st.secrets['gcp_service_account'].")

    credentials_info = normalize_service_account_info(st.secrets["gcp_service_account"])
    private_key = credentials_info["private_key"]
    key_body = private_key.strip()
    starts_correctly = key_body.startswith("-----BEGIN PRIVATE KEY-----")
    ends_correctly = key_body.endswith("-----END PRIVATE KEY-----")
    literal_newlines_remain = "\\n" in private_key

    try:
        credentials = Credentials.from_service_account_info(credentials_info, scopes=GOOGLE_SHEETS_SCOPES)
    except Exception as exc:
        raise RuntimeError(
            "Failed to create Google service account credentials. "
            f"Original error: {exc}. "
            f"BEGIN marker found: {starts_correctly}. "
            f"END marker found: {ends_correctly}. "
            f"Literal '\\n' remains: {literal_newlines_remain}. "
            f"Private key length: {len(private_key)}."
        ) from exc
    client = gspread.authorize(credentials)

    sheet_id = st.secrets.get("google_sheet_id", "")
    sheet_name = st.secrets.get("google_sheet_name", "")
    if sheet_id:
        spreadsheet = client.open_by_key(sheet_id)
    elif sheet_name:
        spreadsheet = client.open(sheet_name)
    else:
        raise RuntimeError("Set st.secrets['google_sheet_id'] or st.secrets['google_sheet_name'].")

    worksheet = spreadsheet.sheet1
    if not worksheet.row_values(1):
        worksheet.append_row(GOOGLE_SHEETS_HEADER, value_input_option="USER_ENTERED")
    return worksheet


def log_user_login(full_name, email, institution, role_grade, session_id, status="Login"):
    worksheet = get_google_sheet()
    worksheet.append_row(
        [
            datetime.now(timezone.utc).isoformat(),
            full_name,
            email,
            institution,
            role_grade,
            session_id,
            status,
        ],
        value_input_option="USER_ENTERED",
    )


def is_logged_in():
    return bool(st.session_state.get("user_checkin"))


def ensure_session_id():
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    return st.session_state["session_id"]


def ensure_login():
    ensure_session_id()
    if is_logged_in():
        return True

    st.title("User Check-In")
    st.caption("Please enter your details before using CarbonVEP. This check-in is logged for usage tracking.")

    with st.form("user_checkin_form", clear_on_submit=False):
        full_name = st.text_input("Full name")
        email = st.text_input("Email address")
        institution = st.text_input("Institution / organization")
        role_grade = st.text_input("Optional role or grade")
        submitted = st.form_submit_button("Continue", type="primary")

    if submitted:
        if not full_name.strip() or not email.strip() or not institution.strip():
            st.error("Full name, email address, and institution are required.")
            st.stop()

        user_data = {
            "full_name": full_name.strip(),
            "email": email.strip(),
            "institution": institution.strip(),
            "role_grade": role_grade.strip(),
            "session_id": st.session_state["session_id"],
        }

        try:
            log_user_login(
                user_data["full_name"],
                user_data["email"],
                user_data["institution"],
                user_data["role_grade"],
                user_data["session_id"],
                status="Login",
            )
        except Exception as exc:
            st.error(f"Could not log check-in to Google Sheets: {exc}")
            st.stop()

        st.session_state["user_checkin"] = user_data
        st.rerun()

    st.stop()


def save_uploaded_maf(uploaded_file):
    with open(MAF_PATH, "wb") as handle:
        handle.write(uploaded_file.getbuffer())
    return MAF_PATH


def convert_maf_to_vcf(maf_path, vcf_path):
    app_log(f"Reading MAF file: {maf_path}...")
    df = pd.read_csv(maf_path, sep="\t", comment="#", low_memory=False)

    vcf_df = pd.DataFrame()
    vcf_df["#CHROM"] = df["Chromosome"]
    vcf_df["POS"] = df["Start_Position"]
    vcf_df["ID"] = "."
    vcf_df["REF"] = df["Reference_Allele"]
    vcf_df["ALT"] = df["Tumor_Seq_Allele2"]
    vcf_df["QUAL"] = "."
    vcf_df["FILTER"] = "PASS"
    vcf_df["INFO"] = "."

    vcf_df["#CHROM"] = vcf_df["#CHROM"].apply(
        lambda x: f"chr{x}" if not str(x).startswith("chr") else x
    )
    vcf_df = vcf_df.dropna(subset=["#CHROM", "POS", "REF", "ALT"])
    vcf_df = vcf_df.sort_values(by=["#CHROM", "POS"])

    app_log(f"Writing VCF file to: {vcf_path}...")
    with open(vcf_path, "w") as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write("##source=GDC_Glioma_MAF_Converter\n")
        vcf_df.to_csv(f, sep="\t", index=False)

    app_log("Conversion complete!")
    return vcf_df


def download_and_prepare_hg38():
    if not FASTA_PATH.exists():
        if not FASTA_GZ_PATH.exists():
            app_log("Downloading hg38.fa.gz from UCSC...")
            urlretrieve(
                "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz",
                FASTA_GZ_PATH,
            )

        app_log("Decompressing genome...")
        with gzip.open(FASTA_GZ_PATH, "rb") as compressed:
            with open(FASTA_PATH, "wb") as fasta:
                shutil.copyfileobj(compressed, fasta)

    if not Path(f"{FASTA_PATH}.fai").exists():
        app_log("Indexing genome with pysam...")
        pysam.faidx(str(FASTA_PATH))

    app_log("Genome ready!")
    return FASTA_PATH


def extract_mutation_context(vcf_path, fasta_path, context_window=131072):
    app_log("Loading genome index...")
    genome = pysam.FastaFile(str(fasta_path))
    df = pd.read_csv(
        vcf_path,
        sep="\t",
        comment="#",
        names=["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO"],
    )

    half_window = context_window // 2
    prepared_data = []
    for _, row in df.iterrows():
        chrom = str(row["CHROM"])
        pos = int(row["POS"])
        ref = str(row["REF"])
        alt = str(row["ALT"])
        start = max(0, pos - half_window)
        end = pos + half_window
        wt_seq = genome.fetch(chrom, start, end).upper()
        mutation_idx = pos - start - 1
        mut_seq = wt_seq[:mutation_idx] + alt + wt_seq[mutation_idx + 1 :]
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

    app_log(f"Done! Successfully extracted contexts for {len(prepared_data)} variants.")
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
        ids = tokenizer("<dna>" + seq, return_tensors="pt", add_special_tokens=False).input_ids.to("cpu")
        with torch.inference_mode():
            outputs = model(ids, output_hidden_states=True)
            logits = outputs.logits
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
        app_log(f" -> Warning: Length mismatch for position {variant_data['pos']}. Setting L2 to 0.0")

    return {
        "chrom": variant_data["chrom"],
        "pos": variant_data["pos"],
        "ref": variant_data["ref"],
        "alt": variant_data["alt"],
        "delta_log_prob": delta,
        "l2_distance": l2_distance,
    }


def run_carbon_inference(dataset, output_csv):
    tokenizer, model = load_carbon_model(MODEL_NAME, get_hf_token())

    if dataset and len(dataset) > 0:
        total_variants = len(dataset)
        app_log(f"Found {total_variants} variants in dataset. Step 1: Gathering raw scores...")
        raw_results = []

        for idx, variant in enumerate(dataset):
            wt_full = variant["wildtype_ctx"]
            mut_full = variant["mutant_ctx"]
            mid_wt = len(wt_full) // 2
            mid_mut = len(mut_full) // 2
            WINDOW_BEFORE_AFTER = 100
            sliced_variant = {
                "chrom": variant["chrom"],
                "pos": variant["pos"],
                "ref": variant["ref"],
                "alt": variant["alt"],
                "wildtype_ctx": wt_full[max(0, mid_wt - WINDOW_BEFORE_AFTER) : mid_wt + WINDOW_BEFORE_AFTER],
                "mutant_ctx": mut_full[max(0, mid_mut - WINDOW_BEFORE_AFTER) : mid_mut + WINDOW_BEFORE_AFTER],
            }
            res = evaluate_variant(sliced_variant, tokenizer, model)
            raw_results.append(res)
            app_log(f"[{idx + 1}/{total_variants}] Processed raw metrics for {res['chrom']}:{res['pos']}")

        app_log("Step 2: Calculating Dataset Normalization Factors...")
        deltas = np.array([r["delta_log_prob"] for r in raw_results])
        l2s = np.array([r["l2_distance"] for r in raw_results])
        mean_delta, std_delta = np.mean(deltas), np.std(deltas)
        mean_l2, std_l2 = np.mean(l2s), np.std(l2s)
        std_delta = std_delta if std_delta > 0 else 1.0
        std_l2 = std_l2 if std_l2 > 0 else 1.0

        app_log("Step 3: Compounding Z-Scores and Writing to CSV...")
        with open(output_csv, mode="w", newline="") as f:
            fieldnames = ["chrom", "pos", "ref", "alt", "delta_log_prob", "l2_distance", "variant_score"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for res in raw_results:
                norm_delta = (res["delta_log_prob"] - mean_delta) / std_delta
                norm_l2 = (res["l2_distance"] - mean_l2) / std_l2
                v_score = norm_l2 - norm_delta
                res["variant_score"] = float(v_score)
                writer.writerow(res)

        app_log(f"Successfully finished! Pipeline results written to: {output_csv}")
        return pd.DataFrame(raw_results)

    app_log("Dataset array not found in your current workspace.")
    return pd.DataFrame()


def generate_chromosome_plots(csv_path):
    df = pd.read_csv(csv_path)
    sns.set_theme(style="whitegrid")
    unique_chromosomes = df["chrom"].unique()
    app_log(f"Found data for chromosomes: {unique_chromosomes}")

    for chrom in unique_chromosomes:
        app_log(f"Generating positional profiles for {chrom}...")
        chrom_df = df[df["chrom"] == chrom].sort_values(by="pos")
        if chrom_df.empty or len(chrom_df) < 2:
            app_log(f" -> Skipping {chrom}: Not enough variants for continuous sequence line plotting.")
            continue

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        ax1.plot(chrom_df["pos"], chrom_df["variant_score"], color="purple", marker="o", linestyle="-", linewidth=1.5, alpha=0.8)
        ax1.fill_between(chrom_df["pos"], chrom_df["variant_score"], color="purple", alpha=0.1)
        ax1.set_ylabel("Unified Variant Score", fontsize=11, fontweight="bold")
        ax1.set_title(f"Genomic Landscape Profile: {chrom}", fontsize=14, fontweight="bold", pad=15)
        ax2.plot(chrom_df["pos"], chrom_df["delta_log_prob"], color="teal", marker="s", linestyle="-", linewidth=1.5, alpha=0.8)
        ax2.fill_between(chrom_df["pos"], chrom_df["delta_log_prob"], color="teal", alpha=0.1)
        ax2.set_ylabel(r"$\Delta$ Log Probability", fontsize=11, fontweight="bold")
        ax3.plot(chrom_df["pos"], chrom_df["l2_distance"], color="darkorange", marker="^", linestyle="-", linewidth=1.5, alpha=0.8)
        ax3.fill_between(chrom_df["pos"], chrom_df["l2_distance"], color="darkorange", alpha=0.1)
        ax3.set_ylabel(r"$L2$ Distance", fontsize=11, fontweight="bold")
        ax3.set_xlabel(f"Genomic Position along {chrom} (bp)", fontsize=12, fontweight="bold")
        for ax in [ax1, ax2, ax3]:
            ax.ticklabel_format(style="plain", axis="x")
            ax.xaxis.grid(True, linestyle=":", alpha=0.6)
        output_filename = RUN_DIR / f"chrom_{chrom}_profile.png"
        plt.tight_layout()
        plt.savefig(output_filename, dpi=300)
        plt.close()
        app_log(f" -> Successfully saved: {output_filename.name}")

    app_log("All chromosome tracking plots are successfully generated and saved!")


def identify_gene_via_ucsc(chrom, pos):
    chrom_str = str(chrom)
    chrom_fixed = chrom_str if chrom_str.lower().startswith("chr") else f"chr{chrom_str}"
    url = f"https://api.genome.ucsc.edu/getData/track?genome=hg38;track=ncbiRefSeq;chrom={chrom_fixed};start={pos-1000};end={pos+1000}"
    try:
        response = requests.get(url, timeout=5)
        if response.ok:
            data = response.json()
            items = data.get("ncbiRefSeq", [])
            if items:
                return items[0].get("name2", "Unknown Feature")
    except Exception:
        pass
    return "Intergenic / Non-coding"


def map_carbon_variants_with_ucsc(file_path, output_file):
    df = pd.read_csv(file_path)
    app_log(f"Mapping all {len(df)} variant coordinates to true gene symbols using UCSC...")
    results = []
    for idx, row in df.iterrows():
        if idx % 50 == 0 and idx > 0:
            app_log(f" -> Processed {idx} out of {len(df)} variants...")
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
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(output_file, index=False)
    app_log(f"Processing complete! Successfully saved all mapped records to '{output_file}'.")
    return summary_df


def generate_cohort_property_plots(file_path):
    df = pd.read_csv(file_path)
    df["Mutation Type"] = df["ref"].astype(str) + " ➔ " + df["alt"].astype(str)
    sns.set_theme(style="whitegrid")

    app_log("Generating Graph 1: Variant Score Distribution...")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x="Variant Score", kde=True, color="skyblue", ax=ax, bins=30)
    ax.set_title("Distribution of Carbon Variant Scores", fontsize=14, pad=15, fontweight="bold")
    ax.set_xlabel("Variant Score", fontsize=12)
    ax.set_ylabel("Count / Frequency", fontsize=12)
    plt.tight_layout()
    plt.savefig(RUN_DIR / "1_variant_score_distribution.png", dpi=300)
    plt.close()

    app_log("Generating Graph 2: Top Mutated Genes...")
    gene_df = df[df["Mapped Gene"] != "Intergenic / Non-coding"]
    if not gene_df.empty:
        gene_counts = gene_df["Mapped Gene"].value_counts().reset_index()
        gene_counts.columns = ["Mapped Gene", "Count"]
        top_genes = gene_counts.head(15)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=top_genes, x="Count", y="Mapped Gene", palette="viridis", ax=ax, hue="Mapped Gene", legend=False)
        ax.set_title("Top 15 Most Frequently Mutated Genes", fontsize=14, pad=15, fontweight="bold")
        ax.set_xlabel("Number of Variants Found", fontsize=12)
        ax.set_ylabel("Gene Symbol", fontsize=12)
        plt.tight_layout()
        plt.savefig(RUN_DIR / "2_top_mutated_genes.png", dpi=300)
        plt.close()
    else:
        app_log("Skipping Graph 2: No functional genes mapped in this dataset.")

    app_log("Generating Graph 3: Mutation Type Spectrum...")
    mut_counts = df["Mutation Type"].value_counts().reset_index()
    mut_counts.columns = ["Mutation Type", "Count"]
    mut_counts = mut_counts.sort_values(by="Count", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=mut_counts, x="Mutation Type", y="Count", palette="flare", ax=ax, hue="Mutation Type", legend=False)
    ax.set_title("Genomic Mutation Spectrum (Substitution Frequency)", fontsize=14, pad=15, fontweight="bold")
    ax.set_xlabel("Nucleotide Substitution Type", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(RUN_DIR / "3_mutation_spectrum.png", dpi=300)
    plt.close()
    app_log("All graphics successfully generated and saved as high-resolution PNGs in your directory!")


def query_vep_grch37(chrom, pos, ref, alt, carbon_score):
    chrom_clean = str(chrom).lower().replace("chr", "")
    url = f"https://rest.ensembl.org/vep/human/region/{chrom_clean}:{pos}-{pos}/{alt}?"
    headers = {"Content-Type": "application/json"}
    output_row = {
        "Mutation": f"chr{chrom_clean}:{pos}",
        "Real Mapped Target": "Intergenic / Non-coding",
        "Ensembl_Transcript": "N/A",
        "Protein_Position": "N/A",
        "Amino_Acid_Mutation": "N/A",
        "Carbon Score": carbon_score,
    }
    try:
        response = requests.get(url, headers=headers, timeout=12)
        if response.ok and response.json():
            results = response.json()
            tc_list = results[0].get("transcript_consequences", [])
            selected_tc = None
            for tc in tc_list:
                if "missense_variant" in tc.get("consequence_terms", []):
                    selected_tc = tc
                    break
            if not selected_tc and tc_list:
                selected_tc = tc_list[0]
            if selected_tc:
                gene_symbol = selected_tc.get("gene_symbol", "Unknown")
                transcript_id = selected_tc.get("transcript_id", "N/A")
                protein_pos = selected_tc.get("protein_start", "N/A")
                amino_acids = selected_tc.get("amino_acids", "")
                if "/" in amino_acids and protein_pos != "N/A":
                    ref_aa, alt_aa = amino_acids.split("/")
                    variant_str = f"{ref_aa}{protein_pos}{alt_aa}"
                else:
                    variant_str = "N/A"
                output_row["Real Mapped Target"] = gene_symbol
                output_row["Ensembl_Transcript"] = transcript_id
                output_row["Protein_Position"] = protein_pos
                output_row["Amino_Acid_Mutation"] = variant_str
            elif "regulatory_feature_consequences" in results[0]:
                rc_list = results[0]["regulatory_feature_consequences"]
                reg_ids = [rc["regulatory_feature_id"] for rc in rc_list if "regulatory_feature_id" in rc]
                if reg_ids:
                    output_row["Real Mapped Target"] = ", ".join(reg_ids[:2])
    except Exception as e:
        app_log(f"Error resolving mutation track at chr{chrom_clean}:{pos} -> {e}")
    return output_row


def run_vep_mapping(input_carbon_csv, output_vep_csv):
    app_log(f"Loading raw Carbon dataset: '{input_carbon_csv}'...")
    df = pd.read_csv(input_carbon_csv)
    app_log(f"Processing {len(df)} variants via Ensembl GRCh37 REST nodes (0.3s polite delay)...")
    mapped_rows = []
    for idx, row in df.iterrows():
        time.sleep(0.3)
        v_score = float(row.get("Variant Score", row.get("variant_score", 0.0)))
        chrom = row.get("chrom", row.get("Chromosome"))
        pos = row.get("pos", row.get("Position"))
        ref = row.get("ref", row.get("Reference"))
        alt = row.get("alt", row.get("Alternative"))
        enriched_data = query_vep_grch37(chrom, pos, ref, alt, v_score)
        mapped_rows.append(enriched_data)
        if idx % 10 == 0 and idx > 0:
            app_log(f" -> Checkpoint: Completed parsing index entry {idx} of {len(df)}")
    ordered_cols = ["Mutation", "Real Mapped Target", "Ensembl_Transcript", "Protein_Position", "Amino_Acid_Mutation", "Carbon Score"]
    final_df = pd.DataFrame(mapped_rows)[ordered_cols]
    final_df.to_csv(output_vep_csv, index=False)
    app_log(f"[SUCCESS] Pipeline clean database layer written directly to -> '{output_vep_csv}'")
    return final_df


def get_uniprot_id(gene_name, organism_id="9606"):
    API_URL = "https://rest.uniprot.org/idmapping"
    payload = {"from": "Gene_Name", "to": "UniProtKB", "ids": gene_name, "taxId": organism_id}
    submit_res = requests.post(f"{API_URL}/run", data=payload)
    if submit_res.status_code != 200:
        return None
    job_id = submit_res.json().get("jobId")
    if not job_id:
        return None
    while True:
        status_res = requests.get(f"{API_URL}/status/{job_id}")
        if status_res.status_code == 200:
            status_data = status_res.json()
            if "results" in status_data or status_data.get("jobStatus") == "FINISHED":
                break
        elif status_res.history:
            break
        time.sleep(2)
    results_res = requests.get(f"{API_URL}/results/{job_id}")
    if results_res.status_code != 200:
        return None
    results = results_res.json()
    if isinstance(results, dict) and "results" in results:
        results_list = results["results"]
        if results_list and isinstance(results_list, list):
            first_entry = results_list[0]
            if isinstance(first_entry, dict) and "to" in first_entry:
                to_field = first_entry["to"]
                if isinstance(to_field, dict):
                    return to_field.get("primaryAccession")
                elif isinstance(to_field, str):
                    return to_field
    return None


def get_pdb_ids(uniprot_id):
    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    query = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                "operator": "exact_match",
                "value": uniprot_id,
            },
        },
        "return_type": "entry",
    }
    response = requests.post(url, json=query)
    if response.status_code == 200:
        results = response.json()
        return [item["identifier"] for item in results.get("result_set", [])]
    app_log(f"Error {response.status_code}: {response.text}")
    return []


def retrieve_pdb_mmcif(pdb_id):
    pdbl = PDBList()
    expected = PDB_DIR / f"{pdb_id.lower()}.cif"
    if expected.exists():
        app_log(f"Using local mmCIF: {expected}")
        return str(expected)
    filename = pdbl.retrieve_pdb_file(pdb_id, pdir=str(PDB_DIR), file_format="mmCif")
    app_log(f"Downloaded to: {filename}")
    return filename


def inject_carbon_score_safely(cif_folder, cif_filename, target_residue, carbon_score):
    cif_path = os.path.join(cif_folder, cif_filename)
    output_filename = cif_filename.lower().replace(".cif", "_mapped.cif")
    output_path = os.path.join(cif_folder, output_filename)
    if not os.path.exists(cif_path):
        app_log(f"Structural file not found at: {cif_path}")
        return None
    try:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("protein_target", cif_path)
        modified_atoms = 0
        found_residues_in_file = set()
        found_chains = set()
        for model in structure:
            for chain in model:
                found_chains.add(chain.id)
                for residue in chain:
                    res_num = residue.id[1]
                    found_residues_in_file.add(res_num)
                    if res_num == target_residue:
                        for atom in residue:
                            atom.set_bfactor(float(carbon_score))
                            modified_atoms += 1
        if modified_atoms > 0:
            io = MMCIFIO()
            io.set_structure(structure)
            io.save(output_path)
            app_log(f"Success! Mapped Carbon Score ({carbon_score}) to residue {target_residue} across {modified_atoms} atoms.")
            app_log(f"Saved -> {output_path}")
            return output_path
        app_log(f"Failed to find specific residue {target_residue} inside '{cif_filename}'.")
        sorted_res = sorted(list(found_residues_in_file))
        app_log(f"Available chains in this file: {list(found_chains)}")
        if sorted_res:
            app_log(f"Structural sequence number range present in file: {sorted_res[0]} to {sorted_res[-1]}")
        else:
            app_log("No valid residues parsed from this structural layout.")
    except Exception as e:
        app_log(f"Error parsing structural file arrays: {e}")
    return None


class VariantEnvironmentSelect(Select):
    def __init__(self, valid_chains):
        self.valid_chains = valid_chains

    def accept_chain(self, chain):
        return 1 if chain.id in self.valid_chains else 0


def get_structure_candidates(vep_df):
    candidates = vep_df[(vep_df["Real Mapped Target"].astype(str) != "Intergenic / Non-coding") & (vep_df["Protein_Position"].astype(str) != "N/A")].copy()
    if candidates.empty:
        raise ValueError("No coding VEP row with a Protein_Position is available for structure mapping.")
    candidates["abs_score"] = candidates["Carbon Score"].astype(float).abs()
    return candidates.sort_values("abs_score", ascending=False)


def create_variant_slice(input_cif, output_slice_cif, target_residue):
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("protein", input_cif)
    target_key = (" ", int(target_residue), " ")
    chains_to_keep = []
    for model in structure:
        for chain in model:
            if chain.has_id(target_key):
                chains_to_keep.append(chain.id)
    app_log(f"Filtering complex down to variant-bearing chains: {chains_to_keep}")
    io = MMCIFIO()
    io.set_structure(structure)
    io.save(output_slice_cif, VariantEnvironmentSelect(chains_to_keep))
    app_log(f"Created lightweight variant slice: {output_slice_cif}")
    return chains_to_keep


def render_variant_slice(output_slice_cif, target_residue, amino_acid_mutation, carbon_score):
    with open(output_slice_cif, "r") as f:
        slice_data = f.read()
    view = py3Dmol.view(width=800, height=600)
    view.addModel(slice_data, "cif")
    view.setStyle({}, {"cartoon": {"color": "#CECECE", "opacity": 0.8}})
    mutation_selection = {"resi": [int(target_residue)]}
    view.addStyle(mutation_selection, {"sphere": {"color": "#FF007F", "radius": 3.0}})
    view.addLabel(
        f"Mutation Site: {amino_acid_mutation}\nCarbon Score: {carbon_score}",
        {"fontColor": "white", "backgroundColor": "#111111", "backgroundOpacity": 0.9, "fontSize": 14},
        mutation_selection,
    )
    view.zoomTo(mutation_selection)
    return view._make_html()


def run_structure_mapping(vep_csv):
    vep_df = pd.read_csv(vep_csv)
    candidates = get_structure_candidates(vep_df)
    skipped = []

    for _, row in candidates.iterrows():
        gene_name = str(row["Real Mapped Target"])
        target_residue = int(row["Protein_Position"])
        carbon_score = float(row["Carbon Score"])
        amino_acid_mutation = str(row["Amino_Acid_Mutation"])

        uniprot_id = get_uniprot_id(gene_name)
        if not uniprot_id:
            skipped.append(f"{gene_name}: no UniProt ID")
            app_log(f"Skipping {gene_name}: could not resolve UniProt ID.")
            continue

        app_log(f"UniProt ID for {gene_name}: {uniprot_id}")
        pdb_ids = get_pdb_ids(uniprot_id)
        app_log(f"PDB IDs for {uniprot_id}: {pdb_ids}")
        if not pdb_ids:
            skipped.append(f"{gene_name} ({uniprot_id}): no PDB IDs")
            app_log(f"Skipping {gene_name}: no PDB IDs found for UniProt ID {uniprot_id}.")
            continue

        for pdb_id in pdb_ids:
            cif_path = Path(retrieve_pdb_mmcif(pdb_id))
            mapped_file_path = inject_carbon_score_safely(str(cif_path.parent), cif_path.name, target_residue, carbon_score)
            if not mapped_file_path:
                skipped.append(f"{gene_name} ({pdb_id}): residue {target_residue} not found")
                continue

            output_slice_cif = PDB_DIR / f"{pdb_id.lower()}_variant_slice.cif"
            create_variant_slice(mapped_file_path, str(output_slice_cif), target_residue)
            html = render_variant_slice(output_slice_cif, target_residue, amino_acid_mutation, carbon_score)
            return {
                "skipped": False,
                "gene_name": gene_name,
                "uniprot_id": uniprot_id,
                "pdb_id": pdb_id,
                "target_residue": target_residue,
                "carbon_score": carbon_score,
                "amino_acid_mutation": amino_acid_mutation,
                "mapped_file": mapped_file_path,
                "slice_file": str(output_slice_cif),
                "html": html,
            }

    message = "No structure could be mapped for coding VEP rows with available residue positions."
    app_log(message)
    return {"skipped": True, "message": message, "skipped_details": skipped, "html": None}


def collect_output_files():
    patterns = [
        "glioma_mutations.vcf",
        "carbon_variant_scores.csv",
        "mapped_carbon_variants.csv",
        "vep_mapped_output.csv",
        "chrom_*_profile.png",
        "1_variant_score_distribution.png",
        "2_top_mutated_genes.png",
        "3_mutation_spectrum.png",
        "pdb_files/*.cif",
    ]
    files = []
    for pattern in patterns:
        files.extend(RUN_DIR.glob(pattern))
    return [p for p in files if p.exists() and p.is_file()]


def create_outputs_zip():
    files = collect_output_files()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, arcname=str(path.relative_to(RUN_DIR)))
    return ZIP_PATH


def run_full_pipeline(uploaded_file, progress_bar, status_box):
    st.session_state["log_messages"] = []
    save_uploaded_maf(uploaded_file)
    if st.session_state.get("user_checkin"):
        user = st.session_state["user_checkin"]
        try:
            log_user_login(
                user["full_name"],
                user["email"],
                user["institution"],
                user["role_grade"],
                user["session_id"],
                status=f"Analysis started: {uploaded_file.name}",
            )
        except Exception as exc:
            raise RuntimeError(f"Could not log analysis submission to Google Sheets: {exc}") from exc

    stages = [
        "Upload complete",
        "MAF conversion",
        "Genome preparation",
        "Context extraction",
        "Carbon inference",
        "Chromosome plots",
        "UCSC annotation",
        "Cohort plots",
        "VEP mapping",
        "Protein structure mapping",
        "Output package",
        "Complete",
    ]

    def update(stage_index, message):
        progress_bar.progress(stage_index / (len(stages) - 1), text=message)
        status_box.write(message)
        app_log(message)

    update(0, "Upload complete")
    update(1, "Converting MAF to VCF")
    convert_maf_to_vcf(MAF_PATH, VCF_PATH)
    update(2, "Preparing hg38 reference genome")
    download_and_prepare_hg38()
    update(3, "Extracting mutation contexts")
    dataset = extract_mutation_context(VCF_PATH, FASTA_PATH)
    update(4, "Running Carbon-500M inference")
    run_carbon_inference(dataset, CARBON_CSV)
    update(5, "Generating chromosome profile plots")
    generate_chromosome_plots(CARBON_CSV)
    update(6, "Mapping variants with UCSC")
    map_carbon_variants_with_ucsc(CARBON_CSV, MAPPED_CSV)
    update(7, "Generating cohort summary plots")
    generate_cohort_property_plots(MAPPED_CSV)
    update(8, "Running Ensembl VEP mapping")
    run_vep_mapping(MAPPED_CSV, VEP_CSV)
    update(9, "Mapping Carbon score into protein structure")
    structure_result = run_structure_mapping(VEP_CSV)
    update(10, "Creating output ZIP")
    create_outputs_zip()
    update(11, "Complete")
    return structure_result


ensure_login()

st.title("CarbonVEP")
st.caption("One-click Streamlit interface for the original CarbonVEP_v4 notebook pipeline.")

uploaded_maf = st.file_uploader("Upload glioma MAF file", type=["maf", "txt", "tsv"])
run_clicked = st.button("Start CarbonVEP Analysis", type="primary", disabled=uploaded_maf is None)

if run_clicked and uploaded_maf is not None:
    progress_bar = st.progress(0, text="Starting")
    status_box = st.empty()
    try:
        with st.spinner("Running full CarbonVEP pipeline..."):
            structure_result = run_full_pipeline(uploaded_maf, progress_bar, status_box)
        st.session_state["structure_result"] = structure_result
        st.success("CarbonVEP analysis complete.")
    except Exception as exc:
        st.error(f"Pipeline failed: {exc}")
        app_log(f"Pipeline failed: {exc}")

show_log()

if CARBON_CSV.exists() or MAPPED_CSV.exists() or VEP_CSV.exists():
    st.header("Results Dashboard")
    summary_cols = st.columns(4)
    if VCF_PATH.exists():
        with open(VCF_PATH) as handle:
            variant_count = sum(1 for line in handle if not line.startswith("#"))
        summary_cols[0].metric("VCF variants", variant_count)
    if CARBON_CSV.exists():
        carbon_df = pd.read_csv(CARBON_CSV)
        summary_cols[1].metric("Scored variants", len(carbon_df))
        summary_cols[2].metric("Max Carbon score", f"{carbon_df['variant_score'].max():.4f}")
        summary_cols[3].metric("Min Carbon score", f"{carbon_df['variant_score'].min():.4f}")

    csv_tabs = st.tabs(["Carbon Scores", "Mapped Variants", "VEP Matrix"])
    with csv_tabs[0]:
        preview_dataframe(CARBON_CSV, "Carbon score preview")
    with csv_tabs[1]:
        preview_dataframe(MAPPED_CSV, "Mapped variant preview")
    with csv_tabs[2]:
        preview_dataframe(VEP_CSV, "VEP preview")

    st.subheader("Generated Plots")
    chromosome_plots = sorted(RUN_DIR.glob("chrom_*_profile.png"))
    cohort_plots = [
        RUN_DIR / "1_variant_score_distribution.png",
        RUN_DIR / "2_top_mutated_genes.png",
        RUN_DIR / "3_mutation_spectrum.png",
    ]

    if chromosome_plots:
        chrom_labels = {plot.name.replace("chrom_", "").replace("_profile.png", ""): plot for plot in chromosome_plots}
        selected_chrom = st.selectbox("Chromosome profile", list(chrom_labels.keys()))
        st.image(str(chrom_labels[selected_chrom]), caption=chrom_labels[selected_chrom].name, width=850)

    visible_cohort_plots = [plot for plot in cohort_plots if plot.exists()]
    if visible_cohort_plots:
        with st.expander("Cohort summary plots", expanded=True):
            for plot_path in visible_cohort_plots:
                st.image(str(plot_path), caption=plot_path.name, width=760)

    structure_result = st.session_state.get("structure_result")
    if structure_result:
        st.subheader("Interactive Protein Viewer")
        if structure_result.get("skipped"):
            st.warning(structure_result.get("message", "Protein structure mapping was skipped."))
            skipped_details = structure_result.get("skipped_details", [])
            if skipped_details:
                with st.expander("Structure mapping details"):
                    st.write(skipped_details)
        else:
            components.html(structure_result["html"], height=650, scrolling=False)
            st.write(f"Mapped structure: `{structure_result['mapped_file']}`")
            st.write(f"Variant slice: `{structure_result['slice_file']}`")

    if ZIP_PATH.exists():
        st.download_button(
            "Download all CarbonVEP outputs as ZIP",
            ZIP_PATH.read_bytes(),
            file_name=ZIP_PATH.name,
            mime="application/zip",
        )
