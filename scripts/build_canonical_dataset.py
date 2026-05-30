from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fri.config import load_settings
from fri.data.canonical import build_canonical_tables, write_canonical_tables
from fri.data.loaders import load_raw_tables


def main() -> None:
    settings = load_settings()
    raw_data = load_raw_tables(settings.dataset.raw_root, sample_preferred=settings.dataset.sample_preferred)
    canonical_tables = build_canonical_tables(raw_data, settings.enrichment)
    write_canonical_tables(canonical_tables, settings.dataset.processed_root)

    print(f"Loaded AMLSim layout: {raw_data.layout.name}")
    print(f"Wrote canonical tables to: {settings.dataset.processed_root}")
    for table_name, table in canonical_tables.items():
        print(f"- {table_name}: {len(table)} rows")


if __name__ == "__main__":
    main()
