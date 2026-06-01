# Evaluation Report

## Purpose

Record the current baseline model outcomes from the implemented tabular and graph-driven baselines.

## Evaluation Scope

This report covers two evaluation tracks that are both implemented and executable today:

1. tabular transaction and party baselines on the canonical processed AMLSim sample outputs
2. graph-node fraud baselines on the bundled `20K_fanin200cycle200` AMLSim archive

These are not yet a final apples-to-apples comparison because they run on different artifact surfaces and different labels.

## Tabular Baselines

Artifact: `artifacts/baseline_metrics.json`

### Transaction Target

| Model | PR-AUC | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.9762 | 0.8571 | 1.0000 | 0.9231 | 0.9940 |
| Random Forest | 1.0000 | 1.0000 | 0.8333 | 0.9091 | 1.0000 |

### Party Target

| Model | PR-AUC | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Random Forest | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |

### Interpretation

The tabular metrics are inflated by the tiny sample size and proxy labels. They validate the pipeline, not final model quality.

## Graph Baselines

Artifact: `artifacts/graph/graph_baseline_metrics.json`

Target: node-level fraud prediction on the `20K_fanin200cycle200` graph archive using topology-derived node features.

| Model | PR-AUC | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.5065 | 0.6269 | 0.2683 | 0.3758 | 0.8887 |
| Random Forest | 0.6239 | 0.6811 | 0.4878 | 0.5685 | 0.9191 |

## Current Best Model By Track

| Track | Best Current Model | Reason |
| --- | --- | --- |
| Tabular transaction | Random Forest | strongest PR-AUC and high precision on the sample slice |
| Tabular party | Logistic Regression | avoids the degenerate no-positive-prediction behavior seen in the random forest |
| Graph node fraud | Random Forest | better recall, F1 and PR-AUC than the logistic baseline |

## What The Results Mean

1. the graph feature track is already materially informative on the larger AMLSim archive
2. random forest handles the non-linear graph-feature space better than the linear baseline in the current run
3. the project now has a credible classical baseline plus a graph-enhanced baseline trajectory

## Current Gaps Before Final Evaluation

1. no GNN results yet
2. no shared evaluation split across the tabular and graph tracks
3. no full converted AMLSim dataset with direct alert transaction labels in the processed layer
4. no temporal rolling-window evaluation yet