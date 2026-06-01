# Drift Detection Engine Spec

## Purpose

Measure whether recent transactional behavior has shifted relative to an immediately preceding baseline window using the new temporal feature layer.

## Cohort Construction

Implemented in `src/fri/temporal/drift.py`.

Current default logic:

- recent cohort: last 7 event steps
- baseline cohort: preceding 30 event steps
- feature domain: temporal transaction features only
- ranking signal: Kolmogorov-Smirnov statistic, supported by mean delta and relative mean change

## Current Cohorts On The AMLSim Sample

Artifact: `artifacts/temporal/temporal_drift_report.json`

| Cohort | Step Range | Rows |
| --- | --- | --- |
| Baseline | 13 to 42 | 80 |
| Recent | 43 to 49 | 15 |

Analyzed temporal features: 27

## Highest Drift Signals

| Feature | KS Statistic | Relative Change | Interpretation |
| --- | --- | --- | --- |
| `source_account_amount_prev_30` | 0.5250 | +24.77% | recent source-side rolling value concentration increased |
| `source_party_amount_prev_30` | 0.5250 | +24.77% | party-level 30-step historical load also increased |
| `source_account_amount_prev_7` | 0.5250 | +36.91% | short-horizon outgoing amount pressure increased |
| `source_account_count_prev_30` | 0.4000 | +15.01% | recent source accounts entered with higher prior activity counts |
| `source_account_amount_ratio_1_to_30` | 0.2458 | -51.55% | very-short-horizon share of 30-step flow dropped even while total recent amount rose |

## Interpretation

1. the recent cohort is entering with larger historical amount footprints than the immediately prior baseline window
2. both account-level and party-level source-side features move together, which makes the shift structurally plausible rather than a single-column artifact
3. the velocity-ratio decline suggests the shift is not only more activity, but a redistribution across the 30-step window rather than a pure last-step spike

## Current Implementation Surface

- drift engine: `src/fri/temporal/drift.py`
- report entrypoint: `scripts/run_temporal_intelligence.py`
