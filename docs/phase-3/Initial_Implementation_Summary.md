# Initial Implementation Summary

## Purpose

Capture what has already been executed after the phase-1 AMLSim decision so the repository documents both design intent and concrete implementation state.

## What Was Executed

### Dataset Acquisition

- cloned IBM AMLSim into `data/external/AMLSim`
- pinned repository commit: `7338a4bcb1af9bcfea2201ad7daccfe2a4d569ca`
- confirmed that no GitHub releases are published for AMLSim
- selected the bundled `sample/outputs` layout for the first executable slice while targeting the richer converted schema defined by `paramFiles/1K/schema.json` and `paramFiles/1K/conf.json`

### Documentation Added

- phase-1 fraud taxonomy
- phase-1 dataset assessment
- phase-1 entity relationship mapping

### Scaffold Added

- Python package scaffold under `src/fri`
- configuration file at `configs/default.yaml`
- AMLSim layout discovery and raw table loading
- canonical normalization into source-native and derived entities
- deterministic Device, IP Address and Merchant enrichment
- baseline feature builders for transaction and party scoring
- baseline model trainers using logistic regression and random forest
- executable scripts for canonical dataset building and baseline training
- unit-test scaffold under `tests/unit`

## Generated Artifacts

### Processed Canonical Tables

Written to `data/processed/amlsim` from the bundled AMLSim sample layout.

| Table | Rows |
| --- | --- |
| `parties.csv` | 30 |
| `accounts.csv` | 30 |
| `transactions.csv` | 133 |
| `alerts.csv` | 2 |
| `banks.csv` | 1 |
| `devices.csv` | 14 |
| `ip_addresses.csv` | 24 |
| `merchants.csv` | 19 |
| `account_device_links.csv` | 30 |
| `account_ip_links.csv` | 30 |
| `transaction_merchant_links.csv` | 133 |

### Feature Set Shapes

| Feature Set | Shape |
| --- | --- |
| Transaction | `133 x 23` |
| Party | `30 x 14` |

### Baseline Metrics Artifact

Written to `artifacts/baseline_metrics.json`.

Snapshot from the current sample-backed run:

| Target | Model | PR-AUC | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| Transaction | Logistic Regression | 0.9762 | 0.8571 | 1.0000 | 0.9231 | 0.9940 |
| Transaction | Random Forest | 1.0000 | 1.0000 | 0.8333 | 0.9091 | 1.0000 |
| Party | Logistic Regression | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Party | Random Forest | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |

## Important Caveats

1. These metrics come from the tiny bundled AMLSim sample outputs, not from a production-scale AMLSim simulation.
2. Party labels in sample mode are resolved through alert-to-account ownership mapping because the sample alert file does not align cleanly with the sample account owner identifiers.
3. Transaction labels are direct only when alert transaction files exist; in the sample layout the first slice uses alert-adjacent account membership as a documented proxy.
4. Device, IP Address and Merchant entities are deterministic derived entities and should never be treated as source-native AMLSim evidence.

## Immediate Next Steps

1. Generate a larger AMLSim converted output using the `1K` configuration or another controlled parameter set.
2. Expand the processed layer to ingest full party, account mapping and alert transaction exports when they are available.
3. Add the phase-3 data dictionary and validation specification against the canonical processed tables now being produced.
4. Replace proxy transaction labels with direct alert transaction labels once full converted outputs are generated.