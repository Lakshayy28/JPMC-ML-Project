from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fri.config import load_settings
from fri.data.canonical import build_canonical_tables
from fri.data.loaders import load_raw_tables
from fri.data.validation import validate_canonical_tables


def test_validation_passes_for_sample_canonical_tables() -> None:
    settings = load_settings(REPO_ROOT / "configs" / "default.yaml")
    raw_data = load_raw_tables(settings.dataset.raw_root, sample_preferred=True)
    tables = build_canonical_tables(raw_data, settings.enrichment)
    report = validate_canonical_tables(tables)

    assert report["all_checks_passed"] is True
    assert report["row_counts"]["transactions"] > 0
