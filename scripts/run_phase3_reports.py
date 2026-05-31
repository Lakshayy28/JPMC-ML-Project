from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fri.config import load_settings
from fri.data.loaders import load_processed_tables
from fri.data.profiling import profile_canonical_tables
from fri.data.validation import validate_canonical_tables


def main() -> None:
    settings = load_settings()
    tables = load_processed_tables(settings.dataset.processed_root)

    artifacts_dir = REPO_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    validation_report = validate_canonical_tables(tables)
    profiling_report = profile_canonical_tables(tables)

    validation_path = artifacts_dir / "data_validation_report.json"
    profiling_path = artifacts_dir / "eda_summary.json"

    validation_path.write_text(json.dumps(validation_report, indent=2), encoding="utf-8")
    profiling_path.write_text(json.dumps(profiling_report, indent=2), encoding="utf-8")

    print(f"Wrote validation report to: {validation_path}")
    print(f"Wrote EDA summary to: {profiling_path}")


if __name__ == "__main__":
    main()
