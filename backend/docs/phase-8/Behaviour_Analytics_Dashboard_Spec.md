# Behaviour Analytics Dashboard Spec

## Purpose

Define the operator-facing temporal behavior view backed by the current phase-8 artifact outputs.

## Current Delivery Mode

The repo does not yet include a UI dashboard. The implemented output is artifact-backed and suitable for later FastAPI or Streamlit presentation.

Artifacts:

- `artifacts/temporal/temporal_activity_summary.json`
- `artifacts/temporal/temporal_drift_report.json`

## Recommended Panels

1. step-level transaction volume and total amount
2. alert-related rate and cash rate by event step
3. recent window scorecards for 1, 7, and 30 steps
4. top parties by recent outgoing activity
5. top transactions by source-side velocity ratio
6. top drift features between the recent and baseline cohorts

## Current AMLSim Sample Snapshot

Current recent-window scorecards from the activity artifact:

| Window | Transactions | Amount | Alert-Related Transactions |
| --- | --- | --- | --- |
| 1 step | 2 | 61.85 | 0 |
| 7 steps | 15 | 1359.16 | 4 |
| 30 steps | 81 | 7242.30 | 19 |

Observed step range: 1 to 49

## Implementation Surface

- activity summary builder: `src/fri/temporal/reporting.py`
- report entrypoint: `scripts/run_temporal_intelligence.py`

## Next Dashboard Step

Render these artifact payloads into Streamlit panels once the explainability and API phases begin.
