# Temporal Feature Service Spec

## Purpose

Add rolling-window behavioral features to the canonical transaction and party feature layers so the platform can reason about short-horizon velocity, recent concentration, and behavioral acceleration rather than only static aggregates.

## Configuration

Current config lives in `configs/default.yaml`.

- windows: `[1, 7, 30]`
- recent window for drift: `7`
- baseline window for drift: `30`

## Transaction-Level Temporal Features

Implemented in `src/fri/features/temporal.py` and merged into `src/fri/features/baseline.py`.

Current feature families:

- source account counts and amounts in prior 1, 7, and 30 step windows
- destination account counts and amounts in prior 1, 7, and 30 step windows
- source party counts and amounts in prior 1, 7, and 30 step windows
- average and maximum inter-event gaps for the source account and source party
- 1-to-30 velocity and amount ratios for the source account and source party

## Party-Level Temporal Features

Current feature families:

- activity span in event steps
- outgoing counts and amounts over recent 1, 7, and 30 step windows
- incoming counts and amounts over recent 1, 7, and 30 step windows
- average and maximum recent gaps
- outgoing velocity ratio from recent 1-step to 30-step activity

## Validated Output Shape

Current validated output on the processed AMLSim sample:

| Feature Set | Rows | Columns |
| --- | --- | --- |
| Transaction | 133 | 50 |
| Party | 30 | 34 |

## Current Model Impact

The tabular baseline artifact was refreshed after temporal integration.

Artifact: `artifacts/baseline_metrics.json`

Observed transaction-level results on the current sample:

| Model | PR-AUC | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.9762 | 0.8571 | 1.0000 | 0.9231 | 0.9940 |
| Random Forest | 1.0000 | 1.0000 | 0.8333 | 0.9091 | 1.0000 |

These numbers confirm the pipeline is wired correctly, but they should still be treated as sample-scale validation rather than production evidence.

## Current Implementation Surface

- feature service: `src/fri/features/temporal.py`
- feature integration: `src/fri/features/baseline.py`
- baseline refresh entrypoint: `scripts/train_baseline.py`
