# Performance Comparison Report

## Purpose

Compare the current graph learning tracks implemented on the `20K_fanin200cycle200` AMLSim archive.

## Evaluation Tracks

1. handcrafted graph topology features only
2. spectral graph embeddings only
3. combined topology features plus spectral embeddings
4. pure-PyTorch full-batch GCN using the combined feature bundle as node input

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

## PyTorch GCN Baseline

Artifact: `artifacts/graph/pytorch_gcn_metrics.json`

| Model | PR-AUC | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| 2-layer GCN | 0.5660 | 0.2671 | 0.7982 | 0.4002 | 0.8643 |

## Current Ranking

| Rank | Track | Best Model | Notes |
| --- | --- | --- | --- |
| 1 | Combined features plus embeddings | Random Forest | best PR-AUC and ROC-AUC overall |
| 2 | Graph features only | Random Forest | strongest pure handcrafted graph baseline |
| 3 | PyTorch GCN | 2-layer GCN | first neural graph baseline, materially higher recall than the classical graph baselines |
| 4 | Embeddings only | Random Forest | useful but materially weaker on recall and PR-AUC |

## Interpretation

1. embeddings alone are not yet enough to outperform the handcrafted graph topology features
2. embeddings become useful when combined with graph-native statistics
3. the first PyTorch GCN produces the strongest recall so far on the archive graph task, but it trades away too much precision to beat the best classical hybrid model on PR-AUC
4. the best current overall graph result is still the hybrid classical model using both graph features and spectral embeddings
5. the project now has a validated bridge from classical graph analytics into a true neural graph baseline without requiring PyG or GPU infrastructure

## Next Comparison Steps

1. add Node2Vec-like embeddings for a stronger non-neural embedding baseline
2. tune the GCN decision threshold and loss weighting to recover precision while preserving recall
3. add a GraphSAGE or PyG-backed mini-batch model when the dataset scale or experimentation rate justifies the extra stack
4. compare graph tracks against a canonical account-graph task, not only archive-node fraud prediction