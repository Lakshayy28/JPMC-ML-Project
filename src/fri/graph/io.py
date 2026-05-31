from __future__ import annotations

import io
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ArchiveGraphData:
    archive_path: Path
    sample_name: str
    metadata: dict[str, int | str]
    nodes: pd.DataFrame
    transactions: pd.DataFrame


def _parse_metadata(metadata_text: str) -> dict[str, int | str]:
    parsed: dict[str, int | str] = {"raw_text": metadata_text.strip()}
    node_match = re.search(r"nodes:\s*([\d,]+)\s*\((\d+) fraud nodes\)", metadata_text)
    if node_match:
        parsed["reported_node_count"] = int(node_match.group(1).replace(",", ""))
        parsed["reported_fraud_node_count"] = int(node_match.group(2))
    transaction_match = re.search(r"transactions:\s*([\d,]+)", metadata_text)
    if transaction_match:
        parsed["reported_transaction_count"] = int(transaction_match.group(1).replace(",", ""))
    patterns_match = re.search(r"patterns:\s*(.+)", metadata_text)
    if patterns_match:
        parsed["patterns"] = patterns_match.group(1).strip()
    return parsed


def load_archive_graph_data(archive_path: str | Path) -> ArchiveGraphData:
    path = Path(archive_path)
    sample_name = path.stem
    with tarfile.open(path, "r:gz") as archive:
        metadata_bytes = archive.extractfile(f"{sample_name}/metadata.txt")
        nodes_bytes = archive.extractfile(f"{sample_name}/nodes.csv")
        transactions_bytes = archive.extractfile(f"{sample_name}/transactions.csv")
        if metadata_bytes is None or nodes_bytes is None or transactions_bytes is None:
            raise FileNotFoundError(f"Archive {path} is missing one of metadata.txt, nodes.csv, or transactions.csv")

        metadata_text = metadata_bytes.read().decode("utf-8")
        nodes = pd.read_csv(io.BytesIO(nodes_bytes.read()))
        transactions = pd.read_csv(io.BytesIO(transactions_bytes.read()))

    return ArchiveGraphData(
        archive_path=path,
        sample_name=sample_name,
        metadata=_parse_metadata(metadata_text),
        nodes=nodes,
        transactions=transactions,
    )
