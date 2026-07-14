# -*- coding: utf-8 -*-
"""Core utilities for the CarbonVEP Streamlit app.

These helpers are deliberately importable without Streamlit so they can be
tested offline and reused by the app without triggering model downloads.
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
import requests


APP_VERSION = "carbonvep-streamlit-2026-07-14"
SUPPORTED_CHROMOSOMES = {str(i) for i in range(1, 23)} | {"X", "Y", "M", "MT"}
VALID_DNA = set("ACGTN")
REQUIRED_MAF_COLUMNS = ["Chromosome", "Start_Position", "Reference_Allele", "Tumor_Seq_Allele2"]


@dataclass(frozen=True)
class RunPaths:
    base_dir: Path
    session_id: str
    run_id: str
    run_dir: Path
    maf: Path
    vcf: Path
    fasta: Path
    carbon_csv: Path
    mapped_csv: Path
    vep_csv: Path
    rejected_csv: Path
    plot_dir: Path
    structure_dir: Path
    zip_path: Path
    manifest_path: Path

    def to_jsonable(self) -> dict[str, str]:
        data = asdict(self)
        return {key: str(value) for key, value in data.items()}


@dataclass
class StageResult:
    name: str
    status: str = "pending"
    started_at: str | None = None
    completed_at: str | None = None
    duration_seconds: float | None = None
    output_paths: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    def start(self) -> "StageResult":
        self.started_at = utc_now()
        self._start_perf = time.perf_counter()
        self.status = "running"
        return self

    def finish(self, status: str = "success", output_paths: list[Path | str] | None = None) -> "StageResult":
        self.completed_at = utc_now()
        self.duration_seconds = round(time.perf_counter() - getattr(self, "_start_perf", time.perf_counter()), 4)
        self.status = status
        if output_paths:
            self.output_paths.extend(str(path) for path in output_paths)
        if hasattr(self, "_start_perf"):
            delattr(self, "_start_perf")
        return self

    def fail(self, error: Exception | str) -> "StageResult":
        self.error = str(error)
        return self.finish("failed")

    def to_jsonable(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("_start_perf", None)
        return data


class RetryableRequestError(RuntimeError):
    """Raised when a retried HTTP request ultimately fails."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def generate_session_id() -> str:
    return uuid.uuid4().hex


def generate_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:10]


def create_run_paths(base_run_dir: Path, session_id: str, run_id: str | None = None) -> RunPaths:
    run_id = run_id or generate_run_id()
    run_dir = Path(base_run_dir) / session_id / run_id
    plot_dir = run_dir / "plots"
    structure_dir = run_dir / "pdb_files"
    plot_dir.mkdir(parents=True, exist_ok=True)
    structure_dir.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        base_dir=Path(base_run_dir),
        session_id=session_id,
        run_id=run_id,
        run_dir=run_dir,
        maf=run_dir / "input.maf",
        vcf=run_dir / "glioma_mutations.vcf",
        fasta=run_dir / "hg38.fa",
        carbon_csv=run_dir / "carbon_variant_scores.csv",
        mapped_csv=run_dir / "mapped_carbon_variants.csv",
        vep_csv=run_dir / "vep_mapped_output.csv",
        rejected_csv=run_dir / "rejected_variants.csv",
        plot_dir=plot_dir,
        structure_dir=structure_dir,
        zip_path=run_dir / "carbonvep_outputs.zip",
        manifest_path=run_dir / "run_manifest.json",
    )


def safe_cleanup_abandoned_runs(base_run_dir: Path, max_age_hours: int = 24) -> list[str]:
    removed: list[str] = []
    cutoff = time.time() - max_age_hours * 3600
    base = Path(base_run_dir)
    if not base.exists():
        return removed
    for run_dir in base.glob("*/*"):
        try:
            if run_dir.is_dir() and run_dir.stat().st_mtime < cutoff:
                shutil.rmtree(run_dir)
                removed.append(str(run_dir))
        except OSError:
            continue
    return removed


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_chromosome(value: Any) -> str:
    chrom = str(value).strip()
    if chrom.lower().startswith("chr"):
        chrom = chrom[3:]
    chrom = "MT" if chrom.upper() == "M" else chrom.upper()
    return chrom


def normalize_allele(value: Any) -> str:
    return str(value).strip().upper()


def validate_maf_dataframe(
    df: pd.DataFrame,
    *,
    max_variants: int = 500,
    assembly: str = "hg38",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = [column for column in REQUIRED_MAF_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"MAF is missing required columns: {', '.join(missing)}")

    accepted_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str, str]] = set()

    for row_number, (_, row) in enumerate(df.iterrows(), start=1):
        record = row.to_dict()
        reasons: list[str] = []
        chrom = normalize_chromosome(row.get("Chromosome", ""))
        ref = normalize_allele(row.get("Reference_Allele", ""))
        alt = normalize_allele(row.get("Tumor_Seq_Allele2", ""))

        try:
            pos = int(row.get("Start_Position"))
        except (TypeError, ValueError):
            pos = -1
            reasons.append("invalid_position")

        if pos <= 0:
            reasons.append("non_positive_position")
        if chrom not in SUPPORTED_CHROMOSOMES:
            reasons.append("unsupported_chromosome")
        if not ref or not set(ref).issubset(VALID_DNA):
            reasons.append("invalid_ref_allele")
        if not alt or not set(alt).issubset(VALID_DNA):
            reasons.append("invalid_alt_allele")
        if ref == alt:
            reasons.append("alt_matches_ref")
        if len(ref) != 1 or len(alt) != 1:
            reasons.append("unsupported_non_snv")

        key = (chrom, pos, ref, alt)
        if key in seen:
            reasons.append("duplicate_variant")
        seen.add(key)

        record.update(
            {
                "input_row": row_number,
                "normalized_chromosome": chrom,
                "normalized_position": pos,
                "normalized_ref": ref,
                "normalized_alt": alt,
                "reference_assembly": assembly,
            }
        )
        if reasons:
            record["rejection_reason"] = "|".join(dict.fromkeys(reasons))
            rejected_rows.append(record)
        else:
            accepted_rows.append(record)

    if len(accepted_rows) > max_variants:
        kept = accepted_rows[:max_variants]
        for record in accepted_rows[max_variants:]:
            record["rejection_reason"] = "max_variant_limit_exceeded"
            rejected_rows.append(record)
        accepted_rows = kept

    return pd.DataFrame(accepted_rows), pd.DataFrame(rejected_rows)


def read_and_validate_maf(
    maf_path: Path,
    *,
    rejected_path: Path,
    max_upload_mb: int = 50,
    max_variants: int = 500,
    assembly: str = "hg38",
) -> pd.DataFrame:
    if not Path(maf_path).exists() or Path(maf_path).stat().st_size == 0:
        raise ValueError("Uploaded MAF is empty.")
    if Path(maf_path).stat().st_size > max_upload_mb * 1024 * 1024:
        raise ValueError(f"Uploaded MAF exceeds the configured {max_upload_mb} MB size limit.")
    try:
        df = pd.read_csv(maf_path, sep="\t", comment="#", low_memory=False)
    except Exception as exc:
        raise ValueError(f"Uploaded MAF could not be parsed as tab-separated text: {exc}") from exc
    if df.empty:
        raise ValueError("Uploaded MAF contains no data rows.")
    accepted, rejected = validate_maf_dataframe(df, max_variants=max_variants, assembly=assembly)
    if not rejected.empty:
        rejected.to_csv(rejected_path, index=False)
    else:
        pd.DataFrame(columns=list(df.columns) + ["rejection_reason"]).to_csv(rejected_path, index=False)
    if accepted.empty:
        raise ValueError("No supported SNV rows remain after validation. See rejected_variants.csv.")
    return accepted


def build_snv_context(reference_sequence: str, pos_1_based: int, window_bp: int, ref: str, alt: str) -> tuple[str, str, int]:
    if window_bp % 2 != 0:
        raise ValueError("window_bp must be even so the mutation has a defined central position.")
    ref = normalize_allele(ref)
    alt = normalize_allele(alt)
    if len(ref) != 1 or len(alt) != 1:
        raise ValueError("Only SNVs are supported by this scoring path.")
    mutation_index = window_bp // 2
    if len(reference_sequence) != window_bp + 1:
        raise ValueError(f"Expected {window_bp + 1} reference bases, got {len(reference_sequence)}.")
    observed = reference_sequence[mutation_index].upper()
    if observed != ref:
        raise ValueError(f"reference_mismatch: expected {ref}, observed {observed} at one-based coordinate {pos_1_based}.")
    mutant = reference_sequence[:mutation_index] + alt + reference_sequence[mutation_index + 1 :]
    return reference_sequence.upper(), mutant.upper(), mutation_index


def sanitize_url(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def request_with_retries(
    method: str,
    url: str,
    *,
    max_attempts: int = 3,
    backoff_seconds: float = 0.75,
    retry_statuses: set[int] | None = None,
    session: requests.Session | None = None,
    **kwargs: Any,
) -> requests.Response:
    retry_statuses = retry_statuses or {429, 500, 502, 503, 504}
    requester = session or requests
    sanitized = sanitize_url(url)
    last_error: Exception | None = None
    response: requests.Response | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = requester.request(method, url, **kwargs)
            if response.status_code in retry_statuses:
                if attempt == max_attempts:
                    break
                retry_after = retry_after_seconds(response.headers.get("Retry-After"))
                sleep_for = retry_after if retry_after is not None else min(backoff_seconds * (2 ** (attempt - 1)), 8.0)
                sleep_for += random.uniform(0, min(0.25, sleep_for / 4))
                time.sleep(sleep_for)
                continue
            if 400 <= response.status_code < 500:
                response.raise_for_status()
            response.raise_for_status()
            return response
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_error = exc
            if attempt == max_attempts:
                break
            sleep_for = min(backoff_seconds * (2 ** (attempt - 1)), 8.0) + random.uniform(0, 0.25)
            time.sleep(sleep_for)

    if response is not None:
        context = response.text[:250].replace("\n", " ")
        raise RetryableRequestError(
            f"{method.upper()} {sanitized} failed after {max_attempts} attempt(s); "
            f"status={response.status_code}; response={context}"
        )
    raise RetryableRequestError(
        f"{method.upper()} {sanitized} failed after {max_attempts} attempt(s); error={last_error}"
    )


def request_json_with_retries(method: str, url: str, **kwargs: Any) -> Any:
    response = request_with_retries(method, url, **kwargs)
    try:
        return response.json()
    except ValueError as exc:
        raise RetryableRequestError(
            f"{method.upper()} {sanitize_url(url)} returned non-JSON response; status={response.status_code}; "
            f"response={response.text[:250]}"
        ) from exc


def chromosome_sort_key(chromosome: Any) -> tuple[int, Any]:
    chrom = normalize_chromosome(chromosome)
    if chrom.isdigit():
        return (0, int(chrom))
    order = {"X": 23, "Y": 24, "M": 25, "MT": 25}
    return (1, order.get(chrom, chrom))


def write_run_manifest(
    paths: RunPaths,
    *,
    input_hash: str | None,
    model_name: str,
    model_revision: str,
    dtype: str,
    assembly: str,
    sequence_backend: str,
    stage_results: list[StageResult],
    generated_files: list[Path],
    warnings: list[str] | None = None,
) -> None:
    payload = {
        "application_version": APP_VERSION,
        "created_at": utc_now(),
        "session_id": paths.session_id,
        "run_id": paths.run_id,
        "input_sha256": input_hash,
        "model_name": model_name,
        "model_revision": model_revision,
        "dtype": dtype,
        "reference_assembly": assembly,
        "sequence_backend": sequence_backend,
        "paths": paths.to_jsonable(),
        "stages": [stage.to_jsonable() for stage in stage_results],
        "generated_files": [str(path) for path in generated_files if Path(path).exists()],
        "warnings": warnings or [],
    }
    paths.manifest_path.write_text(json.dumps(payload, indent=2))
