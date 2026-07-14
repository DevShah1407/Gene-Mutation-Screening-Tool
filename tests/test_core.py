from pathlib import Path

import pandas as pd
import pytest
import requests

import carbonvep_core as core


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text="ok", headers=None):
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self.headers = headers or {}
        self.content = text.encode()

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error", response=self)


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def valid_maf_df():
    return pd.DataFrame(
        {
            "Chromosome": ["1", "chr2", "X"],
            "Start_Position": [100, 200, 300],
            "Reference_Allele": ["A", "C", "G"],
            "Tumor_Seq_Allele2": ["T", "G", "A"],
        }
    )


def test_maf_schema_validation_accepts_supported_snvs():
    accepted, rejected = core.validate_maf_dataframe(valid_maf_df(), max_variants=10)
    assert len(accepted) == 3
    assert rejected.empty
    assert list(accepted["normalized_chromosome"]) == ["1", "2", "X"]


def test_maf_schema_validation_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        core.validate_maf_dataframe(pd.DataFrame({"Chromosome": ["1"]}))


def test_indels_invalid_alleles_duplicates_and_alt_equals_ref_are_rejected():
    df = pd.DataFrame(
        {
            "Chromosome": ["1", "1", "GL0001", "2", "2"],
            "Start_Position": [10, 10, 12, -1, 20],
            "Reference_Allele": ["A", "A", "C", "G", "AT"],
            "Tumor_Seq_Allele2": ["A", "A", "Z", "T", "A"],
        }
    )
    accepted, rejected = core.validate_maf_dataframe(df)
    assert accepted.empty
    reasons = "|".join(rejected["rejection_reason"].astype(str))
    assert "alt_matches_ref" in reasons
    assert "duplicate_variant" in reasons
    assert "unsupported_chromosome" in reasons
    assert "invalid_alt_allele" in reasons
    assert "non_positive_position" in reasons
    assert "unsupported_non_snv" in reasons


def test_max_variant_limit_is_deterministic():
    df = valid_maf_df()
    accepted, rejected = core.validate_maf_dataframe(df, max_variants=2)
    assert len(accepted) == 2
    assert len(rejected) == 1
    assert rejected.iloc[0]["rejection_reason"] == "max_variant_limit_exceeded"


def test_one_based_coordinate_to_zero_based_mutation_index_and_snv_construction():
    wt, mut, idx = core.build_snv_context("A" * 100 + "C" + "G" * 100, 101, 200, "C", "T")
    assert idx == 100
    assert wt[100] == "C"
    assert mut[100] == "T"
    assert len(wt) == len(mut) == 201


def test_reference_mismatch_is_explicit():
    with pytest.raises(ValueError, match="reference_mismatch"):
        core.build_snv_context("A" * 100 + "C" + "G" * 100, 101, 200, "G", "T")


def test_chromosome_sorting_is_natural():
    values = ["chr10", "chr2", "chr1", "chrX", "chrY", "chrM"]
    assert sorted(values, key=core.chromosome_sort_key) == ["chr1", "chr2", "chr10", "chrX", "chrY", "chrM"]


def test_retry_does_not_retry_404(monkeypatch):
    monkeypatch.setattr(core.time, "sleep", lambda _: None)
    session = FakeSession([FakeResponse(404, text="not found")])
    with pytest.raises(requests.HTTPError):
        core.request_with_retries("GET", "https://example.org/missing?token=secret", session=session)
    assert len(session.calls) == 1


def test_retry_retries_429_then_succeeds(monkeypatch):
    monkeypatch.setattr(core.time, "sleep", lambda _: None)
    session = FakeSession([FakeResponse(429, text="slow down", headers={"Retry-After": "0"}), FakeResponse(200, payload={"ok": True})])
    response = core.request_with_retries("GET", "https://example.org/rate", session=session)
    assert response.status_code == 200
    assert len(session.calls) == 2


def test_retry_retries_timeout_and_500(monkeypatch):
    monkeypatch.setattr(core.time, "sleep", lambda _: None)
    session = FakeSession([requests.Timeout("boom"), FakeResponse(500, text="server error"), FakeResponse(200, payload={})])
    response = core.request_with_retries("GET", "https://example.org/flaky", session=session, max_attempts=3)
    assert response.status_code == 200
    assert len(session.calls) == 3


def test_json_decode_failure_is_clear(monkeypatch):
    monkeypatch.setattr(core.time, "sleep", lambda _: None)
    session = FakeSession([FakeResponse(200, payload=ValueError("bad json"), text="<html>")])
    with pytest.raises(core.RetryableRequestError, match="non-JSON"):
        core.request_json_with_retries("GET", "https://example.org/html", session=session)


def test_two_sessions_have_isolated_run_paths(tmp_path):
    first = core.create_run_paths(tmp_path, "session-a", "run-1")
    second = core.create_run_paths(tmp_path, "session-b", "run-1")
    first.carbon_csv.write_text("a\n1\n")
    second.carbon_csv.write_text("a\n2\n")
    assert first.carbon_csv.read_text() != second.carbon_csv.read_text()
    assert first.run_dir != second.run_dir


def test_stale_run_cleanup_only_removes_old_run_dirs(tmp_path):
    old = core.create_run_paths(tmp_path, "s1", "old")
    fresh = core.create_run_paths(tmp_path, "s2", "fresh")
    old.carbon_csv.write_text("old")
    fresh.carbon_csv.write_text("fresh")
    old_time = core.time.time() - 48 * 3600
    for path in [old.run_dir, old.carbon_csv]:
        path.touch()
        Path(path).chmod(0o755)
    import os

    os.utime(old.run_dir, (old_time, old_time))
    removed = core.safe_cleanup_abandoned_runs(tmp_path, max_age_hours=24)
    assert str(old.run_dir) in removed
    assert fresh.run_dir.exists()
