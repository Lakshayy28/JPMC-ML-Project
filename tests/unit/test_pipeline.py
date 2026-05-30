from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fri.config import load_settings
from fri.data.canonical import build_canonical_tables
from fri.data.loaders import load_raw_tables
from fri.features.baseline import build_feature_sets


def test_canonical_tables_include_derived_entities() -> None:
    settings = load_settings(REPO_ROOT / "configs" / "default.yaml")
    raw_data = load_raw_tables(settings.dataset.raw_root, sample_preferred=True)
    tables = build_canonical_tables(raw_data, settings.enrichment)

    required_tables = {
        "parties",
        "accounts",
        "transactions",
        "alerts",
        "banks",
        "devices",
        "ip_addresses",
        "merchants",
        "account_device_links",
        "account_ip_links",
        "transaction_merchant_links",
    }
    assert required_tables.issubset(tables)
    assert not tables["devices"].empty
    assert not tables["ip_addresses"].empty
    assert not tables["merchants"].empty


def test_feature_sets_include_labels() -> None:
    settings = load_settings(REPO_ROOT / "configs" / "default.yaml")
    raw_data = load_raw_tables(settings.dataset.raw_root, sample_preferred=True)
    tables = build_canonical_tables(raw_data, settings.enrichment)
    feature_sets = build_feature_sets(tables)

    assert "label" in feature_sets["transaction"].columns
    assert "label" in feature_sets["party"].columns
    assert feature_sets["transaction"]["label"].isin([0, 1]).all()
    assert feature_sets["party"]["label"].isin([0, 1]).all()
