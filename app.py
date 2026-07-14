# -*- coding: utf-8 -*-
"""Faithful Streamlit wrapper for CarbonVEP_v4.ipynb.

The notebook is the source of truth. This app changes only the interface:
one MAF upload and one button execute the same disk-based pipeline.
"""

import csv
import gc
import html
import os
import tempfile
import time
import uuid
import zipfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

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
HF_CACHE_DIR = RUN_DIR / "hf_cache"
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_DIR))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components


MODEL_NAME = "HuggingFaceBio/Carbon-500M"
CARBON_MODEL_REVISION = os.environ.get("CARBONVEP_MODEL_REVISION", "106e36ff51b5dfbfe0b078ad18ad37a6956c5714")
PREVIEW_ROWS = 25
IMAGE_PREVIEW_WIDTH = 520
MAX_STRUCTURE_CANDIDATES = 8
MAX_PDB_IDS_PER_UNIPROT = 8
MAX_CIF_BYTES_FOR_MAPPING = int(os.environ.get("CARBONVEP_MAX_CIF_BYTES", str(120 * 1024 * 1024)))
MAX_SLICE_BYTES_FOR_RENDERING = int(os.environ.get("CARBONVEP_MAX_SLICE_BYTES", str(35 * 1024 * 1024)))
MAX_VIEWER_HTML_CHARS = int(os.environ.get("CARBONVEP_MAX_VIEWER_HTML_CHARS", "50000000"))
PROTEIN_VIEWER_WIDTH = 800
PROTEIN_VIEWER_HEIGHT = 600
MAX_INLINE_DOWNLOAD_BYTES = 50 * 1024 * 1024
VEP_REQUEST_DELAY_SECONDS = float(os.environ.get("CARBONVEP_VEP_DELAY_SECONDS", "0.1"))
ALPHAFOLD_MODEL_VERSIONS = ("v6", "v5", "v4", "v3", "v2", "v1")
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
FASTA_PATH = RUN_DIR / "hg38.fa"
CARBON_CSV = RUN_DIR / "carbon_variant_scores.csv"
MAPPED_CSV = RUN_DIR / "mapped_carbon_variants.csv"
VEP_CSV = RUN_DIR / "vep_mapped_output.csv"
PDB_DIR = RUN_DIR / "pdb_files"
PDB_DIR.mkdir(exist_ok=True)
ZIP_PATH = RUN_DIR / "carbonvep_outputs.zip"


st.set_page_config(page_title="CarbonVEP", page_icon="DNA", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --carbon-bg: #f6f8fb;
        --carbon-surface: #ffffff;
        --carbon-surface-soft: #f9fbfd;
        --carbon-border: #d9e1ec;
        --carbon-border-strong: #c5d0df;
        --carbon-text: #111827;
        --carbon-muted: #475569;
        --carbon-heading: #0f172a;
        --carbon-accent: #2563eb;
        --carbon-accent-dark: #1e40af;
        --carbon-accent-soft: #eff6ff;
        --carbon-good-bg: #ecfdf5;
        --carbon-good-text: #065f46;
        --carbon-warn-bg: #fffbeb;
        --carbon-warn-text: #92400e;
        --carbon-error-bg: #fef2f2;
        --carbon-error-text: #991b1b;
    }
    html, body, [data-testid="stAppViewContainer"] {
        background: var(--carbon-bg);
        color: var(--carbon-text);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 3.2rem;
        max-width: 1280px;
    }
    h1 {
        color: var(--carbon-heading);
        font-size: clamp(2rem, 4vw, 3rem);
        font-weight: 800;
        letter-spacing: 0;
        margin-bottom: 0.15rem;
    }
    h2, h3 {
        color: var(--carbon-heading);
        letter-spacing: 0;
    }
    p, label, span, div {
        color: var(--carbon-text);
    }
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stCaptionContainer"],
    .stCaptionContainer {
        color: var(--carbon-muted);
        line-height: 1.55;
    }
    label,
    [data-testid="stWidgetLabel"] p {
        color: var(--carbon-heading);
        font-weight: 650;
    }
    [data-testid="stSidebar"], [data-testid="stHeader"] {
        background: var(--carbon-bg);
    }
    section[data-testid="stSidebar"] {
        border-right: 1px solid var(--carbon-border);
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, #ffffff 0%, #f9fbff 100%);
        border: 1px solid var(--carbon-border);
        border-radius: 12px;
        padding: 1rem 1.05rem;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
    }
    div[data-testid="stMetricValue"] {
        color: var(--carbon-heading);
        font-weight: 800;
    }
    div[data-testid="stMetricLabel"] p {
        color: var(--carbon-muted);
        font-weight: 600;
        font-size: 0.9rem;
    }
    .carbon-card {
        border: 1px solid var(--carbon-border);
        border-radius: 12px;
        padding: 1.05rem;
        background: var(--carbon-surface);
        margin-bottom: 1rem;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
    }
    .carbon-live-panel {
        border: 1px solid var(--carbon-border);
        border-radius: 14px;
        padding: 1.1rem;
        background: var(--carbon-surface);
        margin: 1.1rem 0;
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.07);
    }
    .carbon-section-title {
        font-size: 1.08rem;
        font-weight: 700;
        color: var(--carbon-heading);
        margin: 1rem 0 0.45rem;
        letter-spacing: 0;
    }
    .carbon-muted {
        color: var(--carbon-muted);
        font-size: 0.92rem;
    }
    .carbon-status-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 0.65rem;
        margin: 0.85rem 0 1rem;
    }
    .carbon-status-card {
        border: 1px solid var(--carbon-border);
        border-radius: 12px;
        background: var(--carbon-surface-soft);
        padding: 0.72rem 0.8rem;
    }
    .carbon-status-label {
        display: block;
        color: var(--carbon-heading);
        font-size: 0.88rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }
    .carbon-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        border-radius: 999px;
        padding: 0.28rem 0.62rem;
        font-size: 0.78rem;
        line-height: 1;
        font-weight: 800;
        letter-spacing: 0;
        white-space: nowrap;
        border: 1px solid transparent;
    }
    .carbon-badge-completed {
        background: #16a34a;
        color: #ffffff;
        border-color: #15803d;
    }
    .carbon-badge-running {
        background: #f59e0b;
        color: #111827;
        border-color: #d97706;
    }
    .carbon-badge-queued {
        background: #2563eb;
        color: #ffffff;
        border-color: #1d4ed8;
    }
    .carbon-badge-not-started {
        background: #e5e7eb;
        color: #374151;
        border-color: #d1d5db;
    }
    .carbon-badge-failed {
        background: #dc2626;
        color: #ffffff;
        border-color: #b91c1c;
    }
    .carbon-badge-running::before {
        content: "";
        width: 0.55rem;
        height: 0.55rem;
        border: 2px solid rgba(17, 24, 39, 0.35);
        border-top-color: #111827;
        border-radius: 999px;
        animation: carbon-spin 0.9s linear infinite;
    }
    @keyframes carbon-spin {
        to { transform: rotate(360deg); }
    }
    .carbon-interpretation-card {
        border: 1px solid var(--carbon-border);
        border-radius: 14px;
        background: var(--carbon-surface);
        padding: 1.1rem 1.15rem;
        margin: 0.85rem 0;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
    }
    .carbon-interpretation-card ul {
        margin-top: 0.35rem;
        padding-left: 1.2rem;
    }
    .carbon-interpretation-card li {
        margin-bottom: 0.35rem;
        color: var(--carbon-text);
    }
    .carbon-disclaimer {
        border-left: 4px solid #2563eb;
        background: #eff6ff;
        border-radius: 10px;
        padding: 0.85rem 1rem;
        color: #1e3a8a;
        font-weight: 650;
        margin-top: 1rem;
    }
    .carbon-assistant-response {
        border: 1px solid var(--carbon-border);
        border-radius: 12px;
        background: var(--carbon-surface);
        padding: 0.95rem 1rem;
        margin-top: 0.75rem;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.04);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.3rem;
        border-bottom: 1px solid var(--carbon-border);
        padding-top: 0.35rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: #eef3f9;
        border: 1px solid transparent;
        border-radius: 10px 10px 0 0;
        color: var(--carbon-heading);
        font-weight: 650;
        padding: 0.65rem 1rem;
        min-height: 2.75rem;
    }
    .stTabs [data-baseweb="tab"] p {
        color: var(--carbon-heading);
    }
    .stTabs [aria-selected="true"] {
        background: var(--carbon-surface);
        color: var(--carbon-accent-dark);
        border: 1px solid var(--carbon-border);
        border-bottom-color: var(--carbon-surface);
    }
    .stTabs [aria-selected="true"] p {
        color: var(--carbon-accent-dark);
    }
    div[data-testid="stExpander"] {
        background: var(--carbon-surface);
        border: 1px solid var(--carbon-border);
        border-radius: 12px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
        overflow: hidden;
    }
    div[data-testid="stExpander"] details summary p {
        color: var(--carbon-heading);
        font-weight: 650;
    }
    div[data-testid="stAlert"] {
        border-radius: 10px;
        border: 1px solid var(--carbon-border);
        color: var(--carbon-text);
    }
    div[data-testid="stAlert"] * {
        color: var(--carbon-text);
    }
    div[data-testid="stAlert"][kind="success"],
    div[data-testid="stAlert"][data-baseweb="notification"][kind="success"] {
        background: var(--carbon-good-bg);
    }
    div[data-testid="stAlert"][kind="warning"],
    div[data-testid="stAlert"][data-baseweb="notification"][kind="warning"] {
        background: var(--carbon-warn-bg);
    }
    div[data-testid="stAlert"][kind="error"],
    div[data-testid="stAlert"][data-baseweb="notification"][kind="error"] {
        background: var(--carbon-error-bg);
    }
    div[data-testid="stDataFrame"],
    div[data-testid="stTable"] {
        border: 1px solid var(--carbon-border);
        border-radius: 12px;
        overflow: hidden;
        background: var(--carbon-surface);
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.04);
    }
    div[data-testid="stDataFrame"] * {
        color: var(--carbon-text);
    }
    div[data-testid="stFileUploader"] {
        background: var(--carbon-surface);
        border: 1px solid var(--carbon-border);
        border-radius: 12px;
        padding: 0.8rem;
    }
    div[data-testid="stFileUploader"] section {
        background: var(--carbon-surface-soft);
        border-color: var(--carbon-border-strong);
        border-radius: 10px;
    }
    div[data-testid="stFileUploader"] * {
        color: var(--carbon-text);
    }
    div[data-testid="stSelectbox"] > div,
    div[data-baseweb="select"] > div,
    input,
    textarea {
        background: var(--carbon-surface);
        color: var(--carbon-text);
        border-color: var(--carbon-border);
    }
    div[data-baseweb="select"] * {
        color: var(--carbon-text);
    }
    .stButton > button {
        border-radius: 10px;
        font-weight: 700;
        border: 1px solid var(--carbon-border-strong);
        box-shadow: 0 3px 10px rgba(15, 23, 42, 0.08);
    }
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="baseButton-primary"] {
        background: var(--carbon-accent);
        border-color: var(--carbon-accent);
        color: #ffffff;
    }
    .stButton > button[kind="primary"] *,
    .stButton > button[data-testid="baseButton-primary"] * {
        color: #ffffff;
    }
    .stDownloadButton > button {
        border-radius: 10px;
        border: 1px solid var(--carbon-border-strong);
        font-weight: 700;
    }
    div[data-testid="stProgress"] div {
        color: var(--carbon-heading);
    }
    div[data-testid="stImage"] img {
        border-radius: 10px;
        border: 1px solid var(--carbon-border);
        background: var(--carbon-surface);
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
    }
    code {
        color: #1e3a8a;
        background: #eff6ff;
        border-radius: 6px;
        padding: 0.1rem 0.25rem;
    }
    hr {
        border-color: var(--carbon-border);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def app_log(message):
    st.session_state.setdefault("log_messages", []).append(str(message))


@contextmanager
def timed_stage(stage_name):
    start = time.perf_counter()
    app_log(f"[TIMER] {stage_name} started.")
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        app_log(f"[TIMER] {stage_name} completed in {duration:.2f}s.")


def show_log():
    with st.expander("Run Log", expanded=False):
        st.text("\n".join(st.session_state.get("log_messages", [])))


def preview_dataframe(path, label):
    if Path(path).exists():
        df = read_csv_safely(path, label)
        if df is None:
            return None
        st.write(f"{label}: `{Path(path).name}` ({len(df)} rows x {len(df.columns)} columns)")
        st.dataframe(df.head(PREVIEW_ROWS), width="stretch")
        return df
    return None


def read_csv_safely(path, label):
    try:
        if Path(path).exists():
            return pd.read_csv(path)
    except Exception as exc:
        message = f"Could not read {label} at {Path(path).name}: {exc}"
        app_log(message)
        st.warning(message)
    return None


def request_with_retries(method, url, *, max_attempts=3, backoff_seconds=0.75, retry_statuses=None, **kwargs):
    retry_statuses = retry_statuses or {429, 500, 502, 503, 504}
    last_error = None
    method = str(method).upper()
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.request(method, url, **kwargs)
            if response.status_code in retry_statuses and attempt < max_attempts:
                wait_time = backoff_seconds * (2 ** (attempt - 1))
                app_log(f"{method} {url} returned HTTP {response.status_code}; retrying in {wait_time:.2f}s.")
                time.sleep(wait_time)
                continue
            response.raise_for_status()
            return response
        except Exception as exc:
            last_error = exc
            if attempt >= max_attempts:
                break
            wait_time = backoff_seconds * (2 ** (attempt - 1))
            app_log(f"{method} {url} failed on attempt {attempt}/{max_attempts}: {exc}; retrying in {wait_time:.2f}s.")
            time.sleep(wait_time)
    raise last_error


def request_json_with_retries(method, url, **kwargs):
    response = request_with_retries(method, url, **kwargs)
    return response.json()


def status_badge_html(status, text=None):
    labels = {
        "completed": "✓ Completed",
        "running": "Processing...",
        "queued": "Queued",
        "not-started": "Not Started",
        "failed": "Failed",
    }
    safe_status = status if status in labels else "not-started"
    safe_text = html.escape(text or labels[safe_status])
    return f'<span class="carbon-badge carbon-badge-{safe_status}">{safe_text}</span>'


def render_status_badge(status, text=None):
    st.markdown(status_badge_html(status, text), unsafe_allow_html=True)


def get_pipeline_stage_statuses(structure_result=None):
    structure_result = structure_result or st.session_state.get("structure_result")
    chromosome_plot_ready = any(RUN_DIR.glob("chrom_*_profile.png"))
    cohort_plot_ready = all(
        (RUN_DIR / filename).exists()
        for filename in ["1_variant_score_distribution.png", "2_top_mutated_genes.png", "3_mutation_spectrum.png"]
    )
    reference_ready = bool(st.session_state.get("reference_context_ready")) or CARBON_CSV.exists()
    current_message = str(st.session_state.get("current_pipeline_message", "")).lower()

    stages = [
        ("MAF Uploaded", MAF_PATH.exists(), "upload"),
        ("MAF to VCF", VCF_PATH.exists(), "vcf"),
        ("hg38 Reference Access", reference_ready, "genome"),
        ("Context Extraction", CARBON_CSV.exists(), "context"),
        ("Carbon Scoring", CARBON_CSV.exists(), "carbon"),
        ("Chromosome Plots", chromosome_plot_ready, "chromosome"),
        ("UCSC Annotation", MAPPED_CSV.exists(), "ucsc"),
        ("Cohort Plots", cohort_plot_ready, "cohort"),
        ("VEP Annotation", VEP_CSV.exists(), "vep"),
        ("Protein Mapping", bool(structure_result), "protein"),
        ("Output Package", ZIP_PATH.exists(), "zip"),
    ]

    running_map = {
        "upload": ["upload"],
        "vcf": ["vcf", "maf to vcf", "converting maf"],
        "genome": ["genome", "hg38", "reference"],
        "context": ["context"],
        "carbon": ["carbon", "inference"],
        "chromosome": ["chromosome"],
        "ucsc": ["ucsc"],
        "cohort": ["cohort"],
        "vep": ["vep"],
        "protein": ["protein", "structure"],
        "zip": ["zip", "package"],
    }

    rendered = []
    first_unfinished_seen = False
    for label, is_done, key in stages:
        if key == "protein" and structure_result and structure_result.get("skipped"):
            status = "failed"
            badge_text = "Not Mapped"
        elif is_done:
            status = "completed"
            badge_text = "✓ Completed"
        elif any(term in current_message for term in running_map.get(key, [])):
            status = "running"
            badge_text = "Processing..."
        elif first_unfinished_seen:
            status = "queued"
            badge_text = "Queued"
        else:
            status = "not-started"
            badge_text = "Not Started"
            first_unfinished_seen = True
        if status in {"not-started", "running"}:
            first_unfinished_seen = True
        rendered.append((label, status, badge_text))
    return rendered


def render_pipeline_status_badges(structure_result=None):
    cards = []
    for label, status, badge_text in get_pipeline_stage_statuses(structure_result):
        cards.append(
            '<div class="carbon-status-card">'
            f'<span class="carbon-status-label">{html.escape(label)}</span>'
            f'{status_badge_html(status, badge_text)}'
            "</div>"
        )
    st.markdown('<div class="carbon-status-grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)


def get_score_profile_label(carbon_df):
    if carbon_df is None or carbon_df.empty or "variant_score" not in carbon_df.columns:
        return "not available", "Carbon score output is not available yet."
    scores = pd.to_numeric(carbon_df["variant_score"], errors="coerce").dropna()
    if scores.empty:
        return "not available", "Carbon scores could not be summarized because no numeric scores were available."

    abs_scores = scores.abs()
    median_abs = float(abs_scores.median())
    high_fraction = float((abs_scores >= 2.0).mean())
    moderate_fraction = float(((abs_scores >= 1.0) & (abs_scores < 2.0)).mean())

    if high_fraction >= 0.35 or median_abs >= 2.0:
        profile = "mostly high"
    elif moderate_fraction + high_fraction >= 0.35 or median_abs >= 1.0:
        profile = "moderate"
    else:
        profile = "mostly low"

    summary = (
        f"The score distribution appears {profile}. The median absolute variant score is {median_abs:.3f}, "
        f"with {high_fraction:.0%} of scored variants at or above an absolute score of 2.0."
    )
    return profile, summary


def format_top_items(series, max_items=5):
    if series is None or series.empty:
        return "none available"
    return ", ".join(f"{idx} ({int(val)})" for idx, val in series.head(max_items).items())


def build_clinical_interpretation(carbon_df, mapped_df, vep_df, structure_result):
    profile, score_summary = get_score_profile_label(carbon_df)
    total_variants = len(carbon_df) if carbon_df is not None else 0

    top_gene_text = "No mapped gene summary is available yet."
    highest_gene_text = "No highest-score gene summary is available yet."
    ttn_note = ""
    if mapped_df is not None and not mapped_df.empty and "Mapped Gene" in mapped_df.columns:
        gene_df = mapped_df[mapped_df["Mapped Gene"].astype(str) != "Intergenic / Non-coding"]
        if not gene_df.empty:
            top_counts = gene_df["Mapped Gene"].value_counts()
            top_gene_text = f"The most frequently mapped genes were {format_top_items(top_counts)}."
            if "Variant Score" in gene_df.columns:
                score_gene_df = gene_df.copy()
                score_gene_df["abs_variant_score"] = pd.to_numeric(score_gene_df["Variant Score"], errors="coerce").abs()
                score_gene_df = score_gene_df.dropna(subset=["abs_variant_score"]).sort_values("abs_variant_score", ascending=False)
                if not score_gene_df.empty:
                    highest_examples = [
                        f"{row['Mapped Gene']} ({row['Mutation']}, score {float(row['Variant Score']):.4f})"
                        for _, row in score_gene_df.head(3).iterrows()
                    ]
                    highest_gene_text = "The variants with the highest absolute Carbon scores mapped to " + "; ".join(highest_examples) + "."
            if "TTN" in set(gene_df["Mapped Gene"].astype(str)):
                ttn_note = (
                    " Several variants were identified in TTN. TTN is a very large structural gene, "
                    "and variants are commonly observed in sequencing datasets. Additional evidence is required "
                    "before assigning any clinical significance."
                )

    mutation_text = "Mutation spectrum and chromosome distribution are not available yet."
    if mapped_df is not None and not mapped_df.empty:
        pieces = []
        if {"ref", "alt"}.issubset(mapped_df.columns):
            mut_counts = (mapped_df["ref"].astype(str) + " -> " + mapped_df["alt"].astype(str)).value_counts()
            pieces.append(f"The most common mutation types were {format_top_items(mut_counts, max_items=4)}.")
        if "chrom" in mapped_df.columns:
            chrom_counts = mapped_df["chrom"].astype(str).value_counts()
            pieces.append(f"The most represented chromosomes were {format_top_items(chrom_counts, max_items=4)}.")
            if len(mapped_df) > 0 and not chrom_counts.empty:
                top_fraction = float(chrom_counts.iloc[0] / len(mapped_df))
                if top_fraction >= 0.4:
                    pieces.append(
                        f"Variants are concentrated on {chrom_counts.index[0]} in this dataset, "
                        "which may reflect the uploaded cohort or sequencing context rather than disease significance."
                    )
                else:
                    pieces.append("The variants appear distributed across multiple chromosomes in this output.")
        if pieces:
            mutation_text = " ".join(pieces)

    protein_text = "Protein structure mapping has not produced a mapped viewer for this run."
    if structure_result:
        if structure_result.get("skipped"):
            protein_text = structure_result.get("message", "Protein structure mapping was skipped for this run.")
        else:
            protein_text = (
                f"Protein mapping produced a structure view for {structure_result.get('gene_name', 'the selected gene')} "
                f"at residue {structure_result.get('target_residue', 'N/A')} using "
                f"{structure_result.get('structure_source', 'structure')} {structure_result.get('pdb_id', '')}."
            )

    return {
        "profile": profile,
        "carbon": (
            f"{score_summary} Carbon scores estimate the predicted functional impact of genomic variants using "
            "a language model trained on genomic sequences. Higher absolute scores may indicate variants that "
            "warrant further investigation but are not by themselves evidence of pathogenicity."
        ),
        "genes": f"{top_gene_text} {highest_gene_text}{ttn_note}",
        "mutation": mutation_text,
        "protein": protein_text,
        "total_variants": total_variants,
    }


def render_clinical_interpretation(carbon_df, mapped_df, vep_df, structure_result):
    summary = build_clinical_interpretation(carbon_df, mapped_df, vep_df, structure_result)
    st.markdown('<div class="carbon-section-title">Clinical Interpretation</div>', unsafe_allow_html=True)
    st.caption("This section summarizes the generated outputs in plain English. It is not diagnostic.")

    st.markdown(
        f"""
        <div class="carbon-interpretation-card">
            <h4>Carbon Score Summary</h4>
            <p>{html.escape(summary["carbon"])}</p>
        </div>
        <div class="carbon-interpretation-card">
            <h4>Gene Summary</h4>
            <p>{html.escape(summary["genes"])}</p>
        </div>
        <div class="carbon-interpretation-card">
            <h4>Mutation Distribution</h4>
            <p>{html.escape(summary["mutation"])}</p>
        </div>
        <div class="carbon-interpretation-card">
            <h4>Protein Mapping Summary</h4>
            <p>{html.escape(summary["protein"])}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="carbon-section-title">Possible Next Steps</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="carbon-interpretation-card">
            <ul>
                <li>Manual review of variants with higher absolute Carbon scores may be useful.</li>
                <li>Comparison with ClinVar could be considered for known clinical annotations.</li>
                <li>Comparison with COSMIC could be considered for cancer-associated variant context.</li>
                <li>Reviewing gene- and variant-specific literature may help place findings in context.</li>
                <li>Discussion with a clinical geneticist may be useful when interpreting findings for clinical settings.</li>
                <li>Experimental validation could be considered for variants selected for follow-up research.</li>
            </ul>
        </div>
        <div class="carbon-disclaimer">
            This report is intended for research and educational purposes only. Carbon scores estimate predicted functional impact and should not be interpreted as evidence of pathogenicity or used alone for clinical decision-making.
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_report_context(carbon_df, mapped_df, vep_df, structure_result):
    context = build_clinical_interpretation(carbon_df, mapped_df, vep_df, structure_result)
    context["has_carbon"] = carbon_df is not None and not carbon_df.empty
    context["has_mapped"] = mapped_df is not None and not mapped_df.empty
    context["has_vep"] = vep_df is not None and not vep_df.empty
    return context


def answer_assistant_question(question, carbon_df, mapped_df, vep_df, structure_result):
    q = str(question or "").strip()
    q_lower = q.lower()
    context = build_report_context(carbon_df, mapped_df, vep_df, structure_result)
    disclaimer = (
        "\n\nThis app is for research and educational use only. It cannot diagnose disease or determine pathogenicity."
    )

    diagnosis_terms = ["do i have", "diagnose", "cancer", "tumor", "pathogenic", "pathogenicity", "disease"]
    if any(term in q_lower for term in diagnosis_terms):
        return (
            "I cannot diagnose disease or determine whether a variant is pathogenic. CarbonVEP summarizes predicted "
            "functional impact and annotations that may help guide research follow-up, but clinical interpretation "
            "requires additional validated evidence and qualified clinical review."
            + disclaimer
        )

    if "carbon score" in q_lower or "variant score" in q_lower:
        return (
            f"Carbon scores estimate predicted functional impact from sequence-model scoring. In this report, the score profile is {context['profile']}. "
            "Higher absolute scores can help prioritize variants for follow-up, but they are not evidence of disease by themselves."
            + disclaimer
        )
    if "delta" in q_lower or "log probability" in q_lower:
        return (
            "Delta Log Probability compares how the model scores the mutant sequence versus the wildtype sequence. "
            "It is one component used by the pipeline before normalization into the final variant score."
            + disclaimer
        )
    if "l2" in q_lower or "distance" in q_lower:
        return (
            "L2 Distance measures how different the model's internal sequence embeddings are between wildtype and mutant contexts. "
            "The pipeline combines this with Delta Log Probability during variant-score normalization."
            + disclaimer
        )
    if "maf" in q_lower:
        return "A MAF file is a Mutation Annotation Format table. In this app it is the uploaded mutation table that gets converted into a VCF-like file before scoring." + disclaimer
    if "vcf" in q_lower:
        return "A VCF-like table stores chromosome, position, reference allele, and alternate allele information. CarbonVEP writes this as glioma_mutations.vcf before extracting sequence context." + disclaimer
    if "vep" in q_lower:
        return "VEP means Variant Effect Predictor. In this app it maps variants to transcripts, protein positions, amino acid changes, and coding or regulatory context when available." + disclaimer
    if "coordinate" in q_lower or "chromosome" in q_lower:
        return "Genome coordinates describe where a variant is located, usually by chromosome and base-pair position. The chromosome plots show where scored variants fall along each chromosome." + disclaimer
    if "residue" in q_lower:
        return "A protein residue is an amino-acid position in a protein sequence. CarbonVEP uses VEP protein positions when trying to map a variant score onto a 3D structure." + disclaimer
    if "uniprot" in q_lower:
        return "UniProt is a protein database. The app uses UniProt accessions to search for available protein structures connected to mapped genes." + disclaimer
    if "pdb" in q_lower:
        return "PDB refers to the Protein Data Bank, a database of experimentally determined protein structures. The app tries matching PDB structures before using an AlphaFold fallback." + disclaimer
    if "alphafold" in q_lower:
        return "AlphaFold provides predicted protein structures. In this app it can be used as a fallback structure source when PDB entries are unavailable or do not contain the target residue." + disclaimer
    if "missense" in q_lower:
        return "A missense variant changes one amino acid in a protein. VEP may report missense consequences and protein positions when the variant affects a coding transcript." + disclaimer
    if "frameshift" in q_lower:
        return "A frameshift variant changes the reading frame of a coding sequence. It can substantially alter downstream protein sequence, but this app does not diagnose clinical significance." + disclaimer
    if "mutation type" in q_lower or "spectrum" in q_lower:
        return context["mutation"] + disclaimer
    if "plot" in q_lower or "graph" in q_lower:
        return (
            "The plots summarize the same pipeline outputs visually: score distributions, frequently mapped genes, mutation spectrum, "
            "and chromosome-level positional profiles. They help users inspect patterns, not diagnose disease."
            + disclaimer
        )
    if "ttn" in q_lower:
        return (
            "TTN is a very large structural gene. Variants in TTN are commonly observed in sequencing datasets, so additional evidence is needed before assigning any clinical meaning."
            + disclaimer
        )
    if "protein" in q_lower or "structure" in q_lower:
        return context["protein"] + " Structure mapping is intended as a visualization aid and does not establish pathogenicity." + disclaimer

    return (
        "I can explain CarbonVEP terminology and this report's generated outputs in plain English. "
        "For this run, " + context["carbon"] + " " + context["genes"]
        + disclaimer
    )


def render_ai_assistant(carbon_df, mapped_df, vep_df, structure_result):
    st.markdown('<div class="carbon-section-title">AI Assistant</div>', unsafe_allow_html=True)
    st.caption("Ask plain-English questions about the report, terminology, plots, or pipeline outputs. The assistant does not diagnose disease.")

    examples = [
        "What is Carbon Score?",
        "What does this graph show?",
        "What is a missense mutation?",
        "What is UniProt?",
    ]
    st.caption("Example questions: " + " | ".join(examples))

    if "assistant_messages" not in st.session_state:
        st.session_state["assistant_messages"] = []

    for message in st.session_state["assistant_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask about CarbonVEP outputs or terminology")
    if question:
        st.session_state["assistant_messages"].append({"role": "user", "content": question})
        answer = answer_assistant_question(question, carbon_df, mapped_df, vep_df, structure_result)
        st.session_state["assistant_messages"].append({"role": "assistant", "content": answer})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            st.markdown(answer)


def render_live_outputs(live_box, stage_label="", structure_result=None):
    if live_box is None:
        return

    with live_box.container():
        st.markdown('<div class="carbon-live-panel">', unsafe_allow_html=True)
        st.markdown('<div class="carbon-section-title">Live Results</div>', unsafe_allow_html=True)
        if stage_label:
            st.caption(f"Latest completed step: {stage_label}")
        render_pipeline_status_badges(structure_result)

        if CARBON_CSV.exists():
            carbon_preview = read_csv_safely(CARBON_CSV, "live Carbon score preview")
            if carbon_preview is not None:
                st.markdown('<div class="carbon-section-title">Carbon Scores Available</div>', unsafe_allow_html=True)
                st.dataframe(carbon_preview.head(10), width="stretch")

        live_cols = st.columns(2)
        if MAPPED_CSV.exists():
            mapped_preview = read_csv_safely(MAPPED_CSV, "live mapped variant preview")
            if mapped_preview is not None:
                with live_cols[0]:
                    st.markdown('<div class="carbon-section-title">Mapped Variants Available</div>', unsafe_allow_html=True)
                    st.dataframe(mapped_preview.head(8), width="stretch")

        if VEP_CSV.exists():
            vep_preview = read_csv_safely(VEP_CSV, "live VEP preview")
            if vep_preview is not None:
                with live_cols[1]:
                    st.markdown('<div class="carbon-section-title">VEP Output Available</div>', unsafe_allow_html=True)
                    st.dataframe(vep_preview.head(8), width="stretch")

        live_plots = [
            RUN_DIR / "1_variant_score_distribution.png",
            RUN_DIR / "2_top_mutated_genes.png",
            RUN_DIR / "3_mutation_spectrum.png",
        ]
        visible_live_plots = [plot for plot in live_plots if plot.exists()]
        if visible_live_plots:
            st.markdown('<div class="carbon-section-title">Plots Available</div>', unsafe_allow_html=True)
            plot_cols = st.columns(min(3, len(visible_live_plots)))
            for idx, plot_path in enumerate(visible_live_plots[:3]):
                with plot_cols[idx % len(plot_cols)]:
                    st.image(str(plot_path), caption=plot_path.name, width="stretch")

        if structure_result:
            if structure_result.get("skipped"):
                st.warning(structure_result.get("message", "Protein structure mapping was skipped."))
            elif is_valid_structure_html(structure_result.get("html")):
                st.success(
                    f"Protein viewer ready for {structure_result.get('gene_name', 'selected gene')} "
                    f"residue {structure_result.get('target_residue', 'unknown')}."
                )
        st.markdown("</div>", unsafe_allow_html=True)


def chromosome_sort_key(chromosome):
    value = str(chromosome).replace("chr", "").replace("CHR", "")
    if value.isdigit():
        return (0, int(value))
    if value.upper() == "X":
        return (1, 23)
    if value.upper() == "Y":
        return (1, 24)
    if value.upper() in {"M", "MT"}:
        return (1, 25)
    return (2, value)


def get_chromosome_options(df):
    if df is None or "chrom" not in df.columns:
        return []
    values = [str(chrom) for chrom in df["chrom"].dropna().unique()]
    return sorted(values, key=chromosome_sort_key)


def filter_dataframe_by_chromosome(df, selected_chromosome):
    if df is None or selected_chromosome == "All chromosomes" or "chrom" not in df.columns:
        return df
    return df[df["chrom"].astype(str) == str(selected_chromosome)]


def is_valid_structure_html(html):
    if not isinstance(html, str):
        return False
    html = html.strip()
    if not html or len(html) > MAX_VIEWER_HTML_CHARS:
        return False
    return "<div" in html and ("<script" in html or "3Dmol" in html)


def get_file_size(path):
    try:
        return Path(path).stat().st_size
    except OSError:
        return None


def format_bytes(size):
    if size is None:
        return "unknown size"
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def make_skipped_structure_result(message, skipped_details=None):
    return {
        "skipped": True,
        "message": message,
        "skipped_details": skipped_details or [],
        "html": None,
    }


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
            app_log(f"Google Sheets check-in failed: {exc}")
            st.error("Could not log check-in to Google Sheets. Please verify the Streamlit secrets and sheet sharing, then try again.")
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
    st.session_state["reference_context_ready"] = True
    app_log("hg38 reference access ready through the cached UCSC sequence API.")
    return FASTA_PATH


@st.cache_data(show_spinner=False, ttl=86400)
def fetch_hg38_sequence(chrom, start, end):
    chrom_str = str(chrom)
    chrom_fixed = chrom_str if chrom_str.lower().startswith("chr") else f"chr{chrom_str}"
    start = max(0, int(start))
    end = max(start + 1, int(end))
    url = "https://api.genome.ucsc.edu/getData/sequence"
    params = {"genome": "hg38", "chrom": chrom_fixed, "start": start, "end": end}
    data = request_json_with_retries("GET", url, params=params, timeout=20)
    dna = str(data.get("dna", "")).upper()
    if not dna:
        raise RuntimeError(f"UCSC returned no hg38 sequence for {chrom_fixed}:{start}-{end}.")
    return dna


def extract_mutation_context(vcf_path, fasta_path, context_window=131072):
    app_log("Fetching hg38 mutation contexts from UCSC sequence API...")
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
        try:
            wt_seq = fetch_hg38_sequence(chrom, start, end)
        except Exception as exc:
            app_log(f"Skipping {chrom}:{pos} because hg38 context could not be retrieved: {exc}")
            continue
        mutation_idx = pos - start - 1
        if mutation_idx < 0 or mutation_idx >= len(wt_seq):
            app_log(
                f"Skipping {chrom}:{pos} because the mutation index {mutation_idx} "
                f"is outside the fetched context length {len(wt_seq)}."
            )
            continue
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
    if not prepared_data:
        raise RuntimeError("No hg38 mutation contexts could be extracted. Check UCSC availability and input coordinates.")
    return prepared_data


@st.cache_resource(show_spinner=False)
def load_carbon_model(model_name, hf_token):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    torch.set_num_threads(max(1, int(os.environ.get("CARBONVEP_TORCH_THREADS", "1"))))
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    model_load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=str(HF_CACHE_DIR),
        revision=CARBON_MODEL_REVISION,
        token=hf_token or None,
    )
    requested_dtype = os.environ.get("CARBONVEP_MODEL_DTYPE", "bfloat16").strip().lower()
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    model_dtype = dtype_map.get(requested_dtype, torch.bfloat16)
    model_kwargs = {
        "trust_remote_code": True,
        "cache_dir": str(HF_CACHE_DIR),
        "revision": CARBON_MODEL_REVISION,
        "token": hf_token or None,
        "dtype": model_dtype,
        "low_cpu_mem_usage": True,
        "use_safetensors": True,
    }
    app_log(f"Loading Carbon model revision={CARBON_MODEL_REVISION} with dtype={model_dtype} and low-memory CPU settings.")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    except TypeError:
        model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
        model_kwargs.pop("low_cpu_mem_usage", None)
        model_kwargs.pop("use_safetensors", None)
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model = model.to("cpu").eval()
    app_log(f"[TIMER] Carbon model load completed in {time.perf_counter() - model_load_start:.2f}s.")
    return tokenizer, model


def release_carbon_model_memory():
    try:
        load_carbon_model.clear()
        app_log("Released cached Carbon model before late-stage annotation and structure mapping.")
    except Exception as exc:
        app_log(f"Carbon model cache release skipped: {exc}")
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


def evaluate_variant(variant_data, tokenizer, model):
    import torch
    import torch.nn.functional as F

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


def run_carbon_inference(dataset, output_csv, progress_callback=None):
    tokenizer, model = load_carbon_model(MODEL_NAME, get_hf_token())

    if dataset and len(dataset) > 0:
        total_variants = len(dataset)
        app_log(f"Found {total_variants} variants in dataset. Step 1: Gathering raw scores...")
        raw_results = []
        inference_start = time.perf_counter()

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
            elapsed = max(time.perf_counter() - inference_start, 1e-9)
            throughput = (idx + 1) / elapsed
            app_log(f"[{idx + 1}/{total_variants}] Processed raw metrics for {res['chrom']}:{res['pos']} ({throughput:.2f} variants/s)")
            if progress_callback and (idx == 0 or idx + 1 == total_variants or (idx + 1) % 5 == 0):
                progress_callback(idx + 1, total_variants, res["chrom"], res["pos"], throughput)

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
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = pd.read_csv(csv_path)
    sns.set_theme(
        style="whitegrid",
        rc={
            "axes.facecolor": "#ffffff",
            "figure.facecolor": "#ffffff",
            "axes.edgecolor": "#d9e1ec",
            "grid.color": "#e5eaf2",
            "text.color": "#111827",
            "axes.labelcolor": "#111827",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
        },
    )
    unique_chromosomes = df["chrom"].unique()
    app_log(f"Found data for chromosomes: {unique_chromosomes}")

    for chrom in unique_chromosomes:
        app_log(f"Generating positional profiles for {chrom}...")
        chrom_df = df[df["chrom"] == chrom].sort_values(by="pos")
        if chrom_df.empty or len(chrom_df) < 2:
            app_log(f" -> Skipping {chrom}: Not enough variants for continuous sequence line plotting.")
            continue

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        fig.patch.set_facecolor("#ffffff")
        ax1.plot(chrom_df["pos"], chrom_df["variant_score"], color="#2563eb", marker="o", linestyle="-", linewidth=1.7, alpha=0.9)
        ax1.fill_between(chrom_df["pos"], chrom_df["variant_score"], color="#2563eb", alpha=0.10)
        ax1.set_ylabel("Unified Variant Score", fontsize=11, fontweight="bold")
        ax1.set_title(f"Genomic Landscape Profile: {chrom}", fontsize=14, fontweight="bold", pad=15)
        ax2.plot(chrom_df["pos"], chrom_df["delta_log_prob"], color="#0f766e", marker="s", linestyle="-", linewidth=1.7, alpha=0.9)
        ax2.fill_between(chrom_df["pos"], chrom_df["delta_log_prob"], color="#0f766e", alpha=0.10)
        ax2.set_ylabel(r"$\Delta$ Log Probability", fontsize=11, fontweight="bold")
        ax3.plot(chrom_df["pos"], chrom_df["l2_distance"], color="#b45309", marker="^", linestyle="-", linewidth=1.7, alpha=0.9)
        ax3.fill_between(chrom_df["pos"], chrom_df["l2_distance"], color="#b45309", alpha=0.10)
        ax3.set_ylabel(r"$L2$ Distance", fontsize=11, fontweight="bold")
        ax3.set_xlabel(f"Genomic Position along {chrom} (bp)", fontsize=12, fontweight="bold")
        for ax in [ax1, ax2, ax3]:
            ax.ticklabel_format(style="plain", axis="x")
            ax.xaxis.grid(True, linestyle=":", alpha=0.65)
            ax.yaxis.grid(True, linestyle="-", alpha=0.35)
            ax.set_facecolor("#ffffff")
            ax.tick_params(axis="both", labelsize=10, colors="#475569")
            for spine in ax.spines.values():
                spine.set_color("#d9e1ec")
        output_filename = RUN_DIR / f"chrom_{chrom}_profile.png"
        plt.tight_layout()
        plt.savefig(output_filename, dpi=300)
        plt.close()
        app_log(f" -> Successfully saved: {output_filename.name}")

    app_log("All chromosome tracking plots are successfully generated and saved!")


@st.cache_data(show_spinner=False, ttl=86400)
def identify_gene_via_ucsc(chrom, pos):
    chrom_str = str(chrom)
    chrom_fixed = chrom_str if chrom_str.lower().startswith("chr") else f"chr{chrom_str}"
    url = "https://api.genome.ucsc.edu/getData/track"
    params = {
        "genome": "hg38",
        "track": "ncbiRefSeq",
        "chrom": chrom_fixed,
        "start": int(pos) - 1000,
        "end": int(pos) + 1000,
    }
    try:
        data = request_json_with_retries("GET", url, params=params, timeout=8)
        items = data.get("ncbiRefSeq", [])
        if items:
            return items[0].get("name2", "Unknown Feature")
    except Exception as exc:
        app_log(f"UCSC gene lookup failed for {chrom_fixed}:{pos}: {exc}")
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
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = pd.read_csv(file_path)
    df["Mutation Type"] = df["ref"].astype(str) + " ➔ " + df["alt"].astype(str)
    sns.set_theme(
        style="whitegrid",
        rc={
            "axes.facecolor": "#ffffff",
            "figure.facecolor": "#ffffff",
            "axes.edgecolor": "#d9e1ec",
            "grid.color": "#e5eaf2",
            "text.color": "#111827",
            "axes.labelcolor": "#111827",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
        },
    )
    cohort_palette = ["#2563eb", "#0f766e", "#7c3aed", "#b45309", "#be123c", "#0369a1"]

    app_log("Generating Graph 1: Variant Score Distribution...")
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#ffffff")
    sns.histplot(data=df, x="Variant Score", kde=True, color="#2563eb", ax=ax, bins=30, alpha=0.55, edgecolor="#dbeafe")
    ax.set_title("Distribution of Carbon Variant Scores", fontsize=14, pad=15, fontweight="bold")
    ax.set_xlabel("Variant Score", fontsize=12)
    ax.set_ylabel("Count / Frequency", fontsize=12)
    ax.set_facecolor("#ffffff")
    ax.tick_params(axis="both", labelsize=10, colors="#475569")
    ax.grid(True, color="#e5eaf2", linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color("#d9e1ec")
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
        fig.patch.set_facecolor("#ffffff")
        sns.barplot(data=top_genes, x="Count", y="Mapped Gene", palette=sns.color_palette(cohort_palette, n_colors=len(top_genes)), ax=ax, hue="Mapped Gene", legend=False)
        ax.set_title("Top 15 Most Frequently Mutated Genes", fontsize=14, pad=15, fontweight="bold")
        ax.set_xlabel("Number of Variants Found", fontsize=12)
        ax.set_ylabel("Gene Symbol", fontsize=12)
        ax.set_facecolor("#ffffff")
        ax.tick_params(axis="both", labelsize=10, colors="#475569")
        ax.grid(True, axis="x", color="#e5eaf2", linewidth=0.8)
        ax.grid(False, axis="y")
        for spine in ax.spines.values():
            spine.set_color("#d9e1ec")
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
    fig.patch.set_facecolor("#ffffff")
    sns.barplot(data=mut_counts, x="Mutation Type", y="Count", palette=sns.color_palette(cohort_palette, n_colors=len(mut_counts)), ax=ax, hue="Mutation Type", legend=False)
    ax.set_title("Genomic Mutation Spectrum (Substitution Frequency)", fontsize=14, pad=15, fontweight="bold")
    ax.set_xlabel("Nucleotide Substitution Type", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_facecolor("#ffffff")
    ax.tick_params(axis="both", labelsize=10, colors="#475569")
    ax.grid(True, axis="y", color="#e5eaf2", linewidth=0.8)
    ax.grid(False, axis="x")
    for spine in ax.spines.values():
        spine.set_color("#d9e1ec")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(RUN_DIR / "3_mutation_spectrum.png", dpi=300)
    plt.close()
    app_log("All graphics successfully generated and saved as high-resolution PNGs in your directory!")


@st.cache_data(show_spinner=False, ttl=86400)
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
        if VEP_REQUEST_DELAY_SECONDS > 0:
            time.sleep(VEP_REQUEST_DELAY_SECONDS)
        results = request_json_with_retries("GET", url, headers=headers, timeout=12)
        if results:
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
    app_log(f"Processing {len(df)} variants via Ensembl GRCh37 REST nodes ({VEP_REQUEST_DELAY_SECONDS:.2f}s delay on uncached requests)...")
    mapped_rows = []
    for idx, row in df.iterrows():
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


@st.cache_data(show_spinner=False, ttl=86400)
def search_uniprot_accession_by_gene(gene_name, organism_id="9606"):
    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": f"(gene_exact:{gene_name}) AND (organism_id:{organism_id}) AND (reviewed:true)",
        "fields": "accession,gene_names,reviewed",
        "format": "json",
        "size": 1,
    }
    try:
        results = request_json_with_retries("GET", url, params=params, timeout=12).get("results", [])
        if results:
            accession = results[0].get("primaryAccession")
            if accession:
                app_log(f"UniProt search fallback resolved {gene_name} -> {accession}")
                return accession
    except Exception as exc:
        app_log(f"UniProt search fallback failed for {gene_name}: {exc}")
    return None


@st.cache_data(show_spinner=False, ttl=86400)
def get_uniprot_id(gene_name, organism_id="9606"):
    API_URL = "https://rest.uniprot.org/idmapping"
    payload = {"from": "Gene_Name", "to": "UniProtKB", "ids": gene_name, "taxId": organism_id}
    try:
        submit_res = request_with_retries("POST", f"{API_URL}/run", data=payload, timeout=12)
    except Exception as exc:
        app_log(f"UniProt ID mapping submission failed for {gene_name}: {exc}")
        return search_uniprot_accession_by_gene(gene_name, organism_id)
    job_id = submit_res.json().get("jobId")
    if not job_id:
        return search_uniprot_accession_by_gene(gene_name, organism_id)
    for _ in range(10):
        try:
            status_res = request_with_retries("GET", f"{API_URL}/status/{job_id}", timeout=12, max_attempts=2)
            status_data = status_res.json()
            if "results" in status_data or status_data.get("jobStatus") == "FINISHED":
                break
        except Exception as exc:
            app_log(f"UniProt ID mapping status check failed for {gene_name}: {exc}")
            return search_uniprot_accession_by_gene(gene_name, organism_id)
        time.sleep(2)
    else:
        app_log(f"UniProt lookup timed out for {gene_name}.")
        return search_uniprot_accession_by_gene(gene_name, organism_id)
    try:
        results_res = request_with_retries("GET", f"{API_URL}/results/{job_id}", timeout=12)
    except Exception as exc:
        app_log(f"UniProt ID mapping results retrieval failed for {gene_name}: {exc}")
        return search_uniprot_accession_by_gene(gene_name, organism_id)
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
    return search_uniprot_accession_by_gene(gene_name, organism_id)


@st.cache_data(show_spinner=False, ttl=86400)
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
    try:
        results = request_json_with_retries("POST", url, json=query, timeout=15)
        return [item["identifier"] for item in results.get("result_set", [])]
    except Exception as exc:
        app_log(f"RCSB lookup failed for UniProt {uniprot_id}: {exc}")
    return []


def retrieve_pdb_mmcif(pdb_id):
    from Bio.PDB import PDBList

    pdbl = PDBList()
    expected = PDB_DIR / f"{pdb_id.lower()}.cif"
    if expected.exists():
        app_log(f"Using local mmCIF: {expected}")
        return str(expected)
    filename = pdbl.retrieve_pdb_file(pdb_id, pdir=str(PDB_DIR), file_format="mmCif")
    app_log(f"Downloaded to: {filename}")
    return filename


def retrieve_alphafold_mmcif(uniprot_id):
    safe_uniprot = str(uniprot_id).strip()
    if not safe_uniprot:
        return None
    expected = PDB_DIR / f"af_{safe_uniprot.lower()}_model.cif"
    if expected.exists() and get_file_size(expected):
        app_log(f"Using local AlphaFold mmCIF: {expected}")
        return str(expected)

    for version in ALPHAFOLD_MODEL_VERSIONS:
        url = f"https://alphafold.ebi.ac.uk/files/AF-{safe_uniprot}-F1-model_{version}.cif"
        try:
            app_log(f"Trying AlphaFold fallback for UniProt={safe_uniprot}: {url}")
            response = request_with_retries("GET", url, timeout=25, max_attempts=2)
            if response.status_code == 404:
                continue
            if not response.content.strip():
                app_log(f"AlphaFold fallback returned an empty file for {safe_uniprot} ({version}).")
                continue
            expected.write_bytes(response.content)
            app_log(f"Downloaded AlphaFold mmCIF fallback to: {expected}")
            return str(expected)
        except Exception as exc:
            app_log(f"AlphaFold fallback failed for {safe_uniprot} ({version}): {exc}")
            continue
    return None


def parse_int_residue(value):
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def inspect_structure_residue(cif_path, target_residue):
    from Bio.PDB import MMCIFParser

    cif_size = get_file_size(cif_path)
    if cif_size is None:
        raise ValueError(f"structure file is not readable: {cif_path}")
    if cif_size == 0:
        raise ValueError(f"structure file is empty: {cif_path}")
    if cif_size > MAX_CIF_BYTES_FOR_MAPPING:
        raise ValueError(
            f"structure file is too large for safe Streamlit Cloud parsing "
            f"({format_bytes(cif_size)} > {format_bytes(MAX_CIF_BYTES_FOR_MAPPING)})"
        )
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("protein_target", str(cif_path))
    found_residues = set()
    found_chains = set()
    matching_chains = []

    for model in structure:
        for chain in model:
            found_chains.add(chain.id)
            for residue in chain:
                res_num = residue.id[1]
                found_residues.add(res_num)
                if res_num == target_residue and chain.id not in matching_chains:
                    matching_chains.append(chain.id)

    return structure, found_residues, found_chains, matching_chains


def inject_carbon_score_safely(cif_folder, cif_filename, target_residue, carbon_score):
    cif_path = os.path.join(cif_folder, cif_filename)
    output_filename = cif_filename.lower().replace(".cif", "_mapped.cif")
    output_path = os.path.join(cif_folder, output_filename)
    original_target_residue = target_residue
    target_residue = parse_int_residue(target_residue)
    if target_residue is None:
        app_log(f"Invalid target residue '{original_target_residue}' for {cif_filename}; skipping this structure.")
        return None
    if not os.path.exists(cif_path):
        app_log(f"Structural file not found at: {cif_path}")
        return None
    cif_size = get_file_size(cif_path)
    if cif_size == 0:
        app_log(f"Structural file is empty: {cif_path}")
        return None
    if cif_size and cif_size > MAX_CIF_BYTES_FOR_MAPPING:
        app_log(
            f"Structural file is too large for safe parsing in Streamlit Cloud: "
            f"{cif_path} ({format_bytes(cif_size)})."
        )
        return None
    try:
        structure, found_residues_in_file, found_chains, matching_chains = inspect_structure_residue(cif_path, target_residue)
        modified_atoms = 0
        if not matching_chains:
            app_log(
                f"Residue {target_residue} was not found in '{cif_filename}'. "
                "Skipping this PDB candidate instead of assuming VEP and PDB numbering match."
            )
            sorted_res = sorted(list(found_residues_in_file))
            app_log(f"Available chains in this file: {list(found_chains)}")
            if sorted_res:
                app_log(f"Structural residue number range present in file: {sorted_res[0]} to {sorted_res[-1]}")
            else:
                app_log("No valid residues parsed from this structural layout.")
            return None
        for model in structure:
            for chain in model:
                for residue in chain:
                    res_num = residue.id[1]
                    if res_num == target_residue:
                        for atom in residue:
                            atom.set_bfactor(float(carbon_score))
                            modified_atoms += 1
        if modified_atoms > 0:
            from Bio.PDB import MMCIFIO

            io = MMCIFIO()
            io.set_structure(structure)
            io.save(output_path)
            app_log(f"Success! Mapped Carbon Score ({carbon_score}) to residue {target_residue} across {modified_atoms} atoms.")
            app_log(f"Saved -> {output_path}")
            return output_path
        app_log(f"Residue {target_residue} was detected but no atoms were updated in '{cif_filename}'.")
    except Exception as e:
        app_log(f"Error parsing or writing structural file '{cif_filename}': {e}")
    return None


def get_structure_candidates(vep_df):
    candidates = vep_df[(vep_df["Real Mapped Target"].astype(str) != "Intergenic / Non-coding") & (vep_df["Protein_Position"].astype(str) != "N/A")].copy()
    if candidates.empty:
        return candidates
    candidates["Carbon Score"] = pd.to_numeric(candidates["Carbon Score"], errors="coerce")
    candidates = candidates.dropna(subset=["Carbon Score"])
    if candidates.empty:
        return candidates
    candidates["abs_score"] = candidates["Carbon Score"].abs()
    return candidates.sort_values("abs_score", ascending=False)


def create_variant_slice(input_cif, output_slice_cif, target_residue):
    from Bio.PDB import MMCIFIO, Select

    class VariantEnvironmentSelect(Select):
        def __init__(self, valid_chains):
            self.valid_chains = valid_chains

        def accept_chain(self, chain):
            return 1 if chain.id in self.valid_chains else 0

    target_residue = parse_int_residue(target_residue)
    if target_residue is None:
        return {"success": False, "reason": "invalid target residue", "chains_to_keep": []}
    if not Path(input_cif).exists():
        return {"success": False, "reason": f"mapped mmCIF file does not exist: {input_cif}", "chains_to_keep": []}
    if Path(input_cif).stat().st_size == 0:
        return {"success": False, "reason": f"mapped mmCIF file is empty: {input_cif}", "chains_to_keep": []}

    try:
        structure, found_residues, found_chains, chains_to_keep = inspect_structure_residue(input_cif, target_residue)
        if not chains_to_keep:
            sorted_res = sorted(list(found_residues))
            residue_summary = f"{sorted_res[0]} to {sorted_res[-1]}" if sorted_res else "none"
            reason = (
                f"target residue {target_residue} not present in mapped structure; "
                f"chains={list(found_chains)}, residue_range={residue_summary}"
            )
            app_log(f"Cannot create variant slice: {reason}")
            return {"success": False, "reason": reason, "chains_to_keep": []}

        app_log(f"Filtering complex down to variant-bearing chains: {chains_to_keep}")
        io = MMCIFIO()
        io.set_structure(structure)
        io.save(output_slice_cif, VariantEnvironmentSelect(chains_to_keep))
        output_path = Path(output_slice_cif)
        slice_size = get_file_size(output_path)
        if not output_path.exists() or slice_size == 0:
            return {"success": False, "reason": "generated slice file is empty or missing", "chains_to_keep": chains_to_keep}
        if slice_size and slice_size > MAX_SLICE_BYTES_FOR_RENDERING:
            return {
                "success": False,
                "reason": (
                    "generated slice is too large for safe browser rendering "
                    f"({format_bytes(slice_size)} > {format_bytes(MAX_SLICE_BYTES_FOR_RENDERING)})"
                ),
                "chains_to_keep": chains_to_keep,
            }
        app_log(f"Created lightweight variant slice: {output_slice_cif}")
        return {"success": True, "reason": "", "chains_to_keep": chains_to_keep}
    except Exception as exc:
        reason = f"slice creation failed: {exc}"
        app_log(reason)
        return {"success": False, "reason": reason, "chains_to_keep": []}


def render_variant_slice(output_slice_cif, target_residue, amino_acid_mutation, carbon_score):
    import py3Dmol

    target_residue = parse_int_residue(target_residue)
    if target_residue is None:
        return {"success": False, "reason": "invalid target residue", "html": None}
    slice_path = Path(output_slice_cif)
    if not slice_path.exists():
        return {"success": False, "reason": f"slice file does not exist: {output_slice_cif}", "html": None}
    slice_size = get_file_size(slice_path)
    if slice_size == 0:
        return {"success": False, "reason": f"slice file is empty: {output_slice_cif}", "html": None}
    if slice_size and slice_size > MAX_SLICE_BYTES_FOR_RENDERING:
        return {
            "success": False,
            "reason": (
                "slice file is too large for safe py3Dmol rendering "
                f"({format_bytes(slice_size)} > {format_bytes(MAX_SLICE_BYTES_FOR_RENDERING)})"
            ),
            "html": None,
        }

    try:
        _, _, _, matching_chains = inspect_structure_residue(slice_path, target_residue)
        if not matching_chains:
            return {"success": False, "reason": f"target residue {target_residue} is absent from generated slice", "html": None}
        with open(slice_path, "r") as f:
            slice_data = f.read()
        if not slice_data.strip() or "_atom_site." not in slice_data:
            return {"success": False, "reason": "slice file does not look like a valid atom-containing mmCIF", "html": None}

        view = py3Dmol.view(width=PROTEIN_VIEWER_WIDTH, height=PROTEIN_VIEWER_HEIGHT)
        view.addModel(slice_data, "cif")
        view.setStyle({}, {"cartoon": {"color": "#CECECE", "opacity": 0.8}})
        mutation_selection = {"resi": [target_residue]}
        view.addStyle(mutation_selection, {"sphere": {"color": "#FF007F", "radius": 3.0}})
        view.addLabel(
            f"Mutation Site: {amino_acid_mutation}\nCarbon Score: {carbon_score}",
            {"fontColor": "white", "backgroundColor": "#111111", "backgroundOpacity": 0.9, "fontSize": 14},
            mutation_selection,
        )
        view.zoomTo(mutation_selection)
        html = view._make_html()
        if not is_valid_structure_html(html):
            return {"success": False, "reason": "py3Dmol generated empty or oversized viewer HTML", "html": None}
        return {"success": True, "reason": "", "html": html}
    except Exception as exc:
        reason = f"py3Dmol rendering failed: {exc}"
        app_log(reason)
        return {"success": False, "reason": reason, "html": None}


def safe_structure_label(value):
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(value).lower()).strip("_")
    return cleaned or "structure"


def try_structure_candidate(
    gene_name,
    uniprot_id,
    structure_id,
    source_label,
    cif_path,
    target_residue,
    carbon_score,
    amino_acid_mutation,
    skipped,
):
    app_log(
        f"Trying structure candidate: source={source_label}, gene={gene_name}, "
        f"UniProt={uniprot_id}, structure={structure_id}, VEP protein residue={target_residue}"
    )

    cif_path = Path(cif_path)
    mapped_file_path = inject_carbon_score_safely(str(cif_path.parent), cif_path.name, target_residue, carbon_score)
    if not mapped_file_path:
        reason = (
            f"{gene_name} ({uniprot_id}, {source_label} {structure_id}): residue {target_residue} was not found "
            "using exact structural residue numbering"
        )
        skipped.append(reason)
        app_log(reason)
        return None

    output_slice_cif = PDB_DIR / f"{safe_structure_label(structure_id)}_variant_slice.cif"
    slice_result = create_variant_slice(mapped_file_path, str(output_slice_cif), target_residue)
    if not slice_result.get("success"):
        reason = f"{gene_name} ({uniprot_id}, {source_label} {structure_id}): {slice_result.get('reason', 'slice creation failed')}"
        skipped.append(reason)
        app_log(reason)
        return None

    render_result = render_variant_slice(output_slice_cif, target_residue, amino_acid_mutation, carbon_score)
    if not isinstance(render_result, dict):
        reason = f"{gene_name} ({uniprot_id}, {source_label} {structure_id}): rendering returned no result"
        skipped.append(reason)
        app_log(reason)
        return None
    if not render_result.get("success") or not is_valid_structure_html(render_result.get("html")):
        reason = f"{gene_name} ({uniprot_id}, {source_label} {structure_id}): {render_result.get('reason', 'rendering produced no valid HTML')}"
        skipped.append(reason)
        app_log(reason)
        return None

    app_log(
        f"Structure mapping succeeded: source={source_label}, gene={gene_name}, "
        f"UniProt={uniprot_id}, structure={structure_id}, residue={target_residue}"
    )
    return {
        "skipped": False,
        "gene_name": gene_name,
        "uniprot_id": uniprot_id,
        "pdb_id": structure_id,
        "structure_source": source_label,
        "target_residue": target_residue,
        "carbon_score": carbon_score,
        "amino_acid_mutation": amino_acid_mutation,
        "mapped_file": mapped_file_path,
        "slice_file": str(output_slice_cif),
        "html": render_result["html"],
        "chains": slice_result.get("chains_to_keep", []),
        "skipped_details": skipped,
    }


def run_structure_mapping(vep_csv):
    try:
        vep_df = pd.read_csv(vep_csv)
    except Exception as exc:
        message = f"Protein structure mapping skipped: could not read VEP output ({exc})."
        app_log(message)
        return make_skipped_structure_result(message)

    candidates = get_structure_candidates(vep_df)
    skipped = []
    if candidates.empty:
        message = "Protein structure mapping skipped: no coding VEP row with a usable Protein_Position was available."
        app_log(message)
        return make_skipped_structure_result(message, skipped)

    if len(candidates) > MAX_STRUCTURE_CANDIDATES:
        app_log(
            f"Limiting structure mapping to the top {MAX_STRUCTURE_CANDIDATES} coding VEP candidates "
            f"by absolute Carbon score out of {len(candidates)} candidates."
        )
        candidates = candidates.head(MAX_STRUCTURE_CANDIDATES)

    for candidate_index, row in candidates.iterrows():
        gene_name = str(row["Real Mapped Target"])
        target_residue = parse_int_residue(row["Protein_Position"])
        if target_residue is None:
            reason = f"{gene_name}: invalid Protein_Position '{row['Protein_Position']}'"
            skipped.append(reason)
            app_log(reason)
            continue
        try:
            carbon_score = float(row["Carbon Score"])
        except (TypeError, ValueError):
            reason = f"{gene_name}: invalid Carbon Score '{row['Carbon Score']}'"
            skipped.append(reason)
            app_log(reason)
            continue
        amino_acid_mutation = str(row["Amino_Acid_Mutation"])
        app_log(
            f"Structure mapping candidate VEP row {candidate_index}: "
            f"gene={gene_name}, residue={target_residue}, mutation={amino_acid_mutation}, carbon_score={carbon_score}"
        )

        try:
            uniprot_id = get_uniprot_id(gene_name)
        except Exception as exc:
            reason = f"{gene_name}: UniProt lookup failed: {exc}"
            skipped.append(reason)
            app_log(reason)
            continue
        if not uniprot_id:
            skipped.append(f"{gene_name}: no UniProt ID")
            app_log(f"Skipping {gene_name}: could not resolve UniProt ID.")
            continue

        app_log(f"Selected gene={gene_name}; resolved UniProt ID={uniprot_id}; target residue={target_residue}")
        try:
            pdb_ids = get_pdb_ids(uniprot_id)
        except Exception as exc:
            reason = f"{gene_name} ({uniprot_id}): PDB lookup failed: {exc}"
            skipped.append(reason)
            app_log(reason)
            continue
        app_log(f"PDB IDs for {uniprot_id}: {pdb_ids}")
        if not pdb_ids:
            skipped.append(f"{gene_name} ({uniprot_id}): no PDB IDs")
            app_log(f"No PDB IDs found for {gene_name} ({uniprot_id}); trying AlphaFold fallback.")

        limited_pdb_ids = pdb_ids[:MAX_PDB_IDS_PER_UNIPROT]
        if len(pdb_ids) > len(limited_pdb_ids):
            app_log(
                f"Limiting PDB attempts for {gene_name} ({uniprot_id}) to the first "
                f"{len(limited_pdb_ids)} of {len(pdb_ids)} candidates."
            )

        for pdb_id in limited_pdb_ids:
            try:
                cif_path = Path(retrieve_pdb_mmcif(pdb_id))
                result = try_structure_candidate(
                    gene_name,
                    uniprot_id,
                    pdb_id,
                    "PDB",
                    cif_path,
                    target_residue,
                    carbon_score,
                    amino_acid_mutation,
                    skipped,
                )
                if result:
                    return result
            except Exception as exc:
                reason = f"{gene_name} ({uniprot_id}, PDB {pdb_id}): unexpected structure candidate failure: {exc}"
                skipped.append(reason)
                app_log(reason)
                continue

        try:
            alphafold_cif = retrieve_alphafold_mmcif(uniprot_id)
            if not alphafold_cif:
                reason = f"{gene_name} ({uniprot_id}): no usable AlphaFold mmCIF fallback was available"
                skipped.append(reason)
                app_log(reason)
                continue
            result = try_structure_candidate(
                gene_name,
                uniprot_id,
                f"AF-{uniprot_id}",
                "AlphaFold",
                alphafold_cif,
                target_residue,
                carbon_score,
                amino_acid_mutation,
                skipped,
            )
            if result:
                return result
        except Exception as exc:
            reason = f"{gene_name} ({uniprot_id}, AlphaFold): unexpected fallback failure: {exc}"
            skipped.append(reason)
            app_log(reason)
            continue

    message = "Protein structure mapping skipped: no available PDB or AlphaFold candidate contained and rendered the target residue."
    app_log(message)
    return make_skipped_structure_result(message, skipped)


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
    safe_files = []
    for path in files:
        if not path.exists() or not path.is_file():
            continue
        size = get_file_size(path)
        if path.suffix.lower() == ".cif" and size and size > MAX_CIF_BYTES_FOR_MAPPING:
            app_log(f"Skipping oversized structure file in ZIP: {path.name} ({format_bytes(size)}).")
            continue
        safe_files.append(path)
    return safe_files


def create_outputs_zip():
    files = collect_output_files()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, arcname=str(path.relative_to(RUN_DIR)))
    return ZIP_PATH


def run_structure_mapping_nonfatal(vep_csv):
    try:
        return run_structure_mapping(vep_csv)
    except BaseException as exc:
        message = f"Protein structure mapping skipped after a protected late-stage failure: {exc}"
        app_log(message)
        return make_skipped_structure_result(message, [message])


def create_outputs_zip_nonfatal():
    try:
        return create_outputs_zip()
    except Exception as exc:
        app_log(f"Output ZIP creation skipped: {exc}")
        return None


def run_full_pipeline(uploaded_file, progress_bar, status_box, live_box=None):
    pipeline_start = time.perf_counter()
    st.session_state["log_messages"] = []
    with timed_stage("Upload handling"):
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
            app_log(f"Google Sheets analysis logging failed: {exc}")
            st.warning(
                "Could not log this analysis submission to Google Sheets. "
                "The pipeline will continue because your initial check-in is already complete."
            )

    stages = [
        "Upload complete",
        "MAF conversion",
        "hg38 reference access",
        "Context extraction",
        "Carbon inference",
        "Chromosome plots",
        "UCSC annotation",
        "Cohort plots",
        "VEP mapping",
        "Protein structure mapping",
        "Output package",
        "Analysis finished",
    ]

    def update(stage_index, message):
        st.session_state["current_pipeline_message"] = message
        progress_bar.progress(stage_index / (len(stages) - 1), text=message)
        status_box.write(message)
        app_log(message)

    update(0, "Upload complete")
    render_live_outputs(live_box, "Upload complete")
    update(1, "Converting MAF to VCF")
    with timed_stage("MAF to VCF conversion"):
        convert_maf_to_vcf(MAF_PATH, VCF_PATH)
    render_live_outputs(live_box, "glioma_mutations.vcf ready")
    update(2, "Preparing hg38 reference access")
    with timed_stage("hg38 reference access"):
        download_and_prepare_hg38()
    render_live_outputs(live_box, "hg38 reference access ready")
    update(3, "Extracting mutation contexts")
    with timed_stage("Context extraction"):
        dataset = extract_mutation_context(VCF_PATH, FASTA_PATH)
    render_live_outputs(live_box, "Mutation contexts extracted")
    update(4, "Running Carbon-500M inference")

    def inference_progress(done, total, chrom, pos, throughput):
        stage_fraction = (4 + min(done / max(total, 1), 1.0) * 0.85) / (len(stages) - 1)
        message = f"Running Carbon-500M inference: {done}/{total} variants ({chrom}:{pos}, {throughput:.2f} variants/s)"
        progress_bar.progress(stage_fraction, text=message)
        status_box.write(message)

    with timed_stage("Carbon inference"):
        run_carbon_inference(dataset, CARBON_CSV, progress_callback=inference_progress)
    dataset = None
    release_carbon_model_memory()
    render_live_outputs(live_box, "carbon_variant_scores.csv ready")
    update(5, "Generating chromosome profile plots")
    with timed_stage("Chromosome plot generation"):
        generate_chromosome_plots(CARBON_CSV)
    render_live_outputs(live_box, "Chromosome plots ready")
    update(6, "Mapping variants with UCSC")
    with timed_stage("UCSC annotation"):
        map_carbon_variants_with_ucsc(CARBON_CSV, MAPPED_CSV)
    render_live_outputs(live_box, "mapped_carbon_variants.csv ready")
    update(7, "Generating cohort summary plots")
    with timed_stage("Cohort plot generation"):
        generate_cohort_property_plots(MAPPED_CSV)
    render_live_outputs(live_box, "Cohort plots ready")
    update(8, "Running Ensembl VEP mapping")
    with timed_stage("VEP mapping"):
        run_vep_mapping(MAPPED_CSV, VEP_CSV)
    render_live_outputs(live_box, "vep_mapped_output.csv ready")
    update(9, "Mapping Carbon score into protein structure")
    with timed_stage("Structure mapping"):
        structure_result = run_structure_mapping_nonfatal(VEP_CSV)
    render_live_outputs(live_box, "Protein structure step complete", structure_result=structure_result)
    update(10, "Creating output ZIP")
    with timed_stage("ZIP creation"):
        create_outputs_zip_nonfatal()
    render_live_outputs(live_box, "Output package ready", structure_result=structure_result)
    update(11, "Analysis finished")
    app_log(f"[TIMER] Full pipeline completed in {time.perf_counter() - pipeline_start:.2f}s.")
    return structure_result


ensure_login()

st.title("CarbonVEP")
st.caption("One-click Streamlit interface for the original CarbonVEP_v4 notebook pipeline.")

uploaded_maf = st.file_uploader("Upload glioma MAF file", type=["maf", "txt", "tsv"])
run_clicked = st.button("Start CarbonVEP Analysis", type="primary", disabled=uploaded_maf is None)

if run_clicked and uploaded_maf is not None:
    progress_bar = st.progress(0, text="Starting")
    status_box = st.empty()
    live_results_box = st.empty()
    try:
        with st.spinner("Running full CarbonVEP pipeline..."):
            structure_result = run_full_pipeline(uploaded_maf, progress_bar, status_box, live_results_box)
        st.session_state["structure_result"] = structure_result
        st.success("CarbonVEP analysis complete.")
    except Exception as exc:
        st.error(f"Pipeline failed: {exc}")
        app_log(f"Pipeline failed: {exc}")

show_log()

if CARBON_CSV.exists() or MAPPED_CSV.exists() or VEP_CSV.exists():
    st.header("Results Dashboard")
    st.caption("Pipeline outputs are written to disk using the original CarbonVEP filenames and displayed here for review.")

    carbon_df = read_csv_safely(CARBON_CSV, "Carbon scores") if CARBON_CSV.exists() else None
    mapped_df = read_csv_safely(MAPPED_CSV, "mapped variants") if MAPPED_CSV.exists() else None
    vep_df = read_csv_safely(VEP_CSV, "VEP output") if VEP_CSV.exists() else None

    variant_count = 0
    if VCF_PATH.exists():
        try:
            with open(VCF_PATH) as handle:
                variant_count = sum(1 for line in handle if not line.startswith("#"))
        except Exception as exc:
            app_log(f"Could not count VCF variants: {exc}")

    summary_cols = st.columns(4)
    summary_cols[0].metric("VCF variants", variant_count)
    summary_cols[1].metric("Scored variants", len(carbon_df) if carbon_df is not None else 0)
    if carbon_df is not None and not carbon_df.empty:
        summary_cols[2].metric("Max Carbon score", f"{carbon_df['variant_score'].max():.4f}")
        summary_cols[3].metric("Min Carbon score", f"{carbon_df['variant_score'].min():.4f}")
    else:
        summary_cols[2].metric("Max Carbon score", "N/A")
        summary_cols[3].metric("Min Carbon score", "N/A")

    render_pipeline_status_badges(st.session_state.get("structure_result"))

    chromosome_options = get_chromosome_options(carbon_df)
    selected_filter_chrom = "All chromosomes"
    if chromosome_options:
        with st.container():
            st.markdown('<div class="carbon-section-title">Output Filters</div>', unsafe_allow_html=True)
            selected_filter_chrom = st.selectbox(
                "Filter mapped variant tables by chromosome",
                ["All chromosomes"] + chromosome_options,
                index=0,
            )

    tabs = st.tabs(["Overview", "Chromosomes", "Mapped Variants", "VEP", "Protein View", "Downloads", "Clinical Interpretation", "AI Assistant"])

    cohort_plots = [
        RUN_DIR / "1_variant_score_distribution.png",
        RUN_DIR / "2_top_mutated_genes.png",
        RUN_DIR / "3_mutation_spectrum.png",
    ]
    visible_cohort_plots = [plot for plot in cohort_plots if plot.exists()]
    chromosome_plots = sorted(RUN_DIR.glob("chrom_*_profile.png"), key=lambda p: chromosome_sort_key(p.name.replace("chrom_", "").replace("_profile.png", "")))

    with tabs[0]:
        st.markdown('<div class="carbon-section-title">Cohort Summary</div>', unsafe_allow_html=True)
        if visible_cohort_plots:
            plot_cols = st.columns(2)
            for idx, plot_path in enumerate(visible_cohort_plots):
                with plot_cols[idx % 2]:
                    st.image(str(plot_path), caption=plot_path.name, width=IMAGE_PREVIEW_WIDTH)
        else:
            st.info("Cohort summary plots will appear here after the pipeline generates them.")

        if carbon_df is not None:
            st.markdown('<div class="carbon-section-title">Carbon Score Preview</div>', unsafe_allow_html=True)
            st.dataframe(carbon_df.head(PREVIEW_ROWS), width="stretch")

    with tabs[1]:
        st.markdown('<div class="carbon-section-title">Chromosome Explorer</div>', unsafe_allow_html=True)
        if chromosome_plots:
            st.caption("Open each chromosome panel to review its positional Carbon score profile.")
            for plot_path in chromosome_plots:
                chrom_label = plot_path.name.replace("chrom_", "").replace("_profile.png", "")
                with st.expander(f"Chromosome {chrom_label}", expanded=False):
                    st.image(str(plot_path), caption=plot_path.name, width=IMAGE_PREVIEW_WIDTH)
                    if carbon_df is not None and "chrom" in carbon_df.columns:
                        chrom_rows = carbon_df[carbon_df["chrom"].astype(str) == chrom_label]
                        st.dataframe(chrom_rows.head(PREVIEW_ROWS), width="stretch")
        else:
            st.info("Chromosome-specific plots will appear here after generation.")

    with tabs[2]:
        st.markdown('<div class="carbon-section-title">Mapped Carbon Variants</div>', unsafe_allow_html=True)
        if mapped_df is not None:
            filtered_mapped_df = filter_dataframe_by_chromosome(mapped_df, selected_filter_chrom)
            st.caption(f"Showing {len(filtered_mapped_df)} of {len(mapped_df)} mapped variants.")
            st.dataframe(filtered_mapped_df, width="stretch")
        else:
            st.info("Mapped variant output is not available yet.")

    with tabs[3]:
        st.markdown('<div class="carbon-section-title">VEP Mapped Output</div>', unsafe_allow_html=True)
        if vep_df is not None:
            st.dataframe(vep_df, width="stretch")
        else:
            st.info("VEP output is not available yet.")

    with tabs[4]:
        st.markdown('<div class="carbon-section-title">Interactive Protein Viewer</div>', unsafe_allow_html=True)
        structure_result = st.session_state.get("structure_result")
        if structure_result:
            structure_html = structure_result.get("html")
            skipped_details = structure_result.get("skipped_details", [])
            if structure_result.get("skipped") or not is_valid_structure_html(structure_html):
                st.warning(
                    structure_result.get(
                        "message",
                        "Protein structure mapping did not produce a renderable viewer for this run.",
                    )
                )
                if not structure_result.get("skipped") and not is_valid_structure_html(structure_html):
                    st.write("The structure candidate finished without valid viewer HTML, so the 3D viewer was not rendered.")
                if skipped_details:
                    with st.expander("Structure mapping candidate details", expanded=True):
                        for detail in skipped_details:
                            st.write(f"- {detail}")
                elif structure_result.get("message"):
                    with st.expander("Structure mapping details", expanded=False):
                        st.write(structure_result["message"])
            else:
                gene_name = structure_result.get("gene_name", "selected gene")
                target_residue = structure_result.get("target_residue", "unknown residue")
                structure_id = structure_result.get("pdb_id", "selected structure")
                structure_source = structure_result.get("structure_source", "Structure")
                st.success(f"Mapped {gene_name} residue {target_residue} on {structure_source} {structure_id}.")
                components.html(structure_html, height=PROTEIN_VIEWER_HEIGHT + 40, scrolling=False)
                if structure_result.get("uniprot_id"):
                    st.write(f"UniProt ID: `{structure_result['uniprot_id']}`")
                if structure_result.get("structure_source") or structure_result.get("pdb_id"):
                    st.write(f"Structure source: `{structure_source} {structure_id}`")
                if structure_result.get("mapped_file"):
                    st.write(f"Mapped structure: `{structure_result['mapped_file']}`")
                if structure_result.get("slice_file"):
                    st.write(f"Variant slice: `{structure_result['slice_file']}`")
                if structure_result.get("chains"):
                    st.write(f"Variant-bearing chains: `{', '.join(structure_result['chains'])}`")
                skipped_details = structure_result.get("skipped_details", [])
                if skipped_details:
                    with st.expander("Earlier skipped structure candidates"):
                        for detail in skipped_details:
                            st.write(f"- {detail}")
        else:
            st.info("The protein viewer will appear after a structure candidate is successfully mapped.")

    with tabs[5]:
        st.markdown('<div class="carbon-section-title">Pipeline Output Package</div>', unsafe_allow_html=True)
        st.caption("This is a direct ZIP of the files written by the pipeline, with no schema rewriting.")
        if ZIP_PATH.exists():
            zip_size = get_file_size(ZIP_PATH)
            st.write(f"Output package: `{ZIP_PATH.name}` ({format_bytes(zip_size)})")
            if zip_size and zip_size > MAX_INLINE_DOWNLOAD_BYTES:
                st.warning(
                    "The output ZIP is too large to safely load into Streamlit's in-memory download widget. "
                    f"The file was still written to disk at `{ZIP_PATH}`."
                )
            else:
                with open(ZIP_PATH, "rb") as zip_file:
                    st.download_button(
                        "Download all CarbonVEP outputs as ZIP",
                        zip_file,
                        file_name=ZIP_PATH.name,
                        mime="application/zip",
                    )
        else:
            st.info("The output ZIP will appear after a completed run.")

    with tabs[6]:
        render_clinical_interpretation(carbon_df, mapped_df, vep_df, st.session_state.get("structure_result"))

    with tabs[7]:
        render_ai_assistant(carbon_df, mapped_df, vep_df, st.session_state.get("structure_result"))
