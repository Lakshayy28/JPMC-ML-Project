from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fri.data.loaders import discover_layout, load_raw_tables


def test_discover_sample_layout() -> None:
    layout = discover_layout(REPO_ROOT / "data" / "external" / "AMLSim", sample_preferred=True)
    assert layout.name == "sample_outputs"
    assert layout.transactions_file.name == "tx.csv"


def test_load_raw_tables_has_expected_keys() -> None:
    raw_data = load_raw_tables(REPO_ROOT / "data" / "external" / "AMLSim", sample_preferred=True)
    assert set(raw_data.tables) == {"accounts", "transactions", "alerts", "cash_transactions", "alert_transactions"}
    assert not raw_data.tables["accounts"].empty
    assert not raw_data.tables["transactions"].empty
