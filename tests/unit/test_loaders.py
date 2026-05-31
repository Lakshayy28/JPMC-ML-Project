from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fri.config import load_settings
from fri.data.loaders import (
    discover_archive_graph_layout,
    discover_layout,
    load_raw_tables,
    stream_archive_graph_nodes,
    stream_archive_graph_transactions,
)


def test_discover_sample_layout() -> None:
    layout = discover_layout(REPO_ROOT / "data" / "external" / "AMLSim", sample_preferred=True)
    assert layout.name == "sample_outputs"
    assert layout.transactions_file.name == "tx.csv"


def test_load_raw_tables_has_expected_keys() -> None:
    raw_data = load_raw_tables(REPO_ROOT / "data" / "external" / "AMLSim", sample_preferred=True)
    assert set(raw_data.tables) == {"accounts", "transactions", "alerts", "cash_transactions", "alert_transactions"}
    assert not raw_data.tables["accounts"].empty
    assert not raw_data.tables["transactions"].empty


def test_archive_graph_streaming_has_expected_schema() -> None:
    settings = load_settings(REPO_ROOT / "configs" / "default.yaml")
    archive_path = settings.dataset.graph_archive or settings.graph.archive_sample
    layout = discover_archive_graph_layout(archive_path)

    node_chunk = next(stream_archive_graph_nodes(layout.archive_path, chunksize=2_000))
    transaction_chunk = next(stream_archive_graph_transactions(layout.archive_path, chunksize=2_000))

    assert layout.sample_name == "20K_fanin200cycle200"
    assert {"nodeid", "isFraud", "init_balance", "fraudStep"}.issubset(node_chunk.columns)
    assert {"sourceNodeId", "targetNodeId", "value", "time"}.issubset(transaction_chunk.columns)
