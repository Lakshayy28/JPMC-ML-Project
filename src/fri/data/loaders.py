from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class AmlsimLayout:
    name: str
    base_dir: Path
    accounts_file: Path
    transactions_file: Path
    alerts_file: Path | None
    cash_transactions_file: Path | None
    alert_transactions_file: Path | None


@dataclass(frozen=True)
class RawAmlsimData:
    layout: AmlsimLayout
    tables: dict[str, pd.DataFrame]


def discover_layout(root: str | Path, sample_preferred: bool = True) -> AmlsimLayout:
    root_path = Path(root)

    sample_candidates = [
        root_path / "sample" / "outputs",
        root_path,
    ]
    full_candidates = [
        root_path / "outputs",
        root_path,
    ]

    if sample_preferred:
        for candidate in sample_candidates:
            if (candidate / "tx.csv").exists() and (candidate / "accounts.csv").exists():
                return AmlsimLayout(
                    name="sample_outputs",
                    base_dir=candidate,
                    accounts_file=candidate / "accounts.csv",
                    transactions_file=candidate / "tx.csv",
                    alerts_file=candidate / "alerts.csv" if (candidate / "alerts.csv").exists() else None,
                    cash_transactions_file=candidate / "cash_tx.csv" if (candidate / "cash_tx.csv").exists() else None,
                    alert_transactions_file=None,
                )

    for candidate in full_candidates:
        if (candidate / "transactions.csv").exists() and (candidate / "accounts.csv").exists():
            return AmlsimLayout(
                name="converted_outputs",
                base_dir=candidate,
                accounts_file=candidate / "accounts.csv",
                transactions_file=candidate / "transactions.csv",
                alerts_file=(candidate / "alert_accounts.csv") if (candidate / "alert_accounts.csv").exists() else None,
                cash_transactions_file=(candidate / "cash_tx.csv") if (candidate / "cash_tx.csv").exists() else None,
                alert_transactions_file=(candidate / "alert_transactions.csv") if (candidate / "alert_transactions.csv").exists() else None,
            )

    if not sample_preferred:
        for candidate in sample_candidates:
            if (candidate / "tx.csv").exists() and (candidate / "accounts.csv").exists():
                return AmlsimLayout(
                    name="sample_outputs",
                    base_dir=candidate,
                    accounts_file=candidate / "accounts.csv",
                    transactions_file=candidate / "tx.csv",
                    alerts_file=candidate / "alerts.csv" if (candidate / "alerts.csv").exists() else None,
                    cash_transactions_file=candidate / "cash_tx.csv" if (candidate / "cash_tx.csv").exists() else None,
                    alert_transactions_file=None,
                )

    raise FileNotFoundError(
        f"Could not discover an AMLSim layout under {root_path}. Expected either sample/outputs or outputs files."
    )


def _read_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_raw_tables(root: str | Path, sample_preferred: bool = True) -> RawAmlsimData:
    layout = discover_layout(root, sample_preferred=sample_preferred)
    tables = {
        "accounts": _read_csv(layout.accounts_file),
        "transactions": _read_csv(layout.transactions_file),
        "alerts": _read_csv(layout.alerts_file),
        "cash_transactions": _read_csv(layout.cash_transactions_file),
        "alert_transactions": _read_csv(layout.alert_transactions_file),
    }
    return RawAmlsimData(layout=layout, tables=tables)


def load_processed_tables(processed_root: str | Path) -> dict[str, pd.DataFrame]:
    root_path = Path(processed_root)
    if not root_path.exists():
        raise FileNotFoundError(f"Processed data directory does not exist: {root_path}")
    return {path.stem: pd.read_csv(path) for path in sorted(root_path.glob("*.csv"))}
