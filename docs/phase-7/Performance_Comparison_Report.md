# Performance Comparison Report

## Purpose

Compare the current graph learning tracks implemented on the `20K_fanin200cycle200` AMLSim archive.

## Evaluation Tracks

1. handcrafted graph topology features only
2. spectral graph embeddings only
3. combined topology features plus spectral embeddings

## Graph Feature Only Baseline

Artifact: `artifacts/graph/graph_baseline_metrics.json`

| Model | PR-AUC | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.5065 | 0.6269 | 0.2683 | 0.3758 | 0.8887 |
| Random Forest | 0.6239 | 0.6811 | 0.4878 | 0.5685 | 0.9191 |

## Embedding Only Baseline

Artifact: `artifacts/graph/graph_embedding_metrics.json`

| Model | PR-AUC | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.2151 | 0.7727 | 0.0377 | 0.0719 | 0.6536 |
| Random Forest | 0.3457 | 0.7738 | 0.1441 | 0.2430 | 0.7144 |

## Combined Graph Features And Embeddings

Artifact: `artifacts/graph/graph_embedding_metrics.json`

| Model | PR-AUC | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.5081 | 0.6231 | 0.2749 | 0.3815 | 0.8886 |
| Random Forest | 0.6490 | 0.7386 | 0.4324 | 0.5455 | 0.9224 |

## Current Ranking

| Rank | Track | Best Model | Notes |
| --- | --- | --- | --- |
| 1 | Combined features plus embeddings | Random Forest | best PR-AUC and ROC-AUC overall |
| 2 | Graph features only | Random Forest | strongest pure handcrafted graph baseline |
| 3 | Embeddings only | Random Forest | useful but materially weaker on recall and PR-AUC |

## Interpretation

1. embeddings alone are not yet enough to outperform the handcrafted graph topology features
2. embeddings become useful when combined with graph-native statistics
3. the best current phase-7 result is the hybrid model using both graph features and spectral embeddings
4. the project now has a credible path from classical graph analytics to hybrid graph ML before moving to GNNs

## Next Comparison Steps

1. add Node2Vec-like embeddings for a stronger non-neural embedding baseline
2. add a first GCN or GraphSAGE model when the environment supports the required stack
3. compare graph tracks against a canonical account-graph task, not only archive-node fraud prediction