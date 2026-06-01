# Performance Comparison Report

## Purpose

Compare the current graph learning tracks implemented on the `20K_fanin200cycle200` AMLSim archive.

## Evaluation Tracks

1. handcrafted graph topology features only
2. spectral graph embeddings only
3. combined topology features plus spectral embeddings
4. PyTorch Geometric GraphSAGE using mini-batch neighborhood sampling over streamed archive tensors

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

## PyTorch GraphSAGE Baseline

Artifact: `artifacts/graph/pytorch_gcn_metrics.json`

| Model | PR-AUC | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| 2-layer GraphSAGE | 0.7880 | 0.4674 | 0.7783 | 0.5840 | 0.9113 |

## Current Ranking

| Rank | Track | Best Model | Notes |
| --- | --- | --- | --- |
| 1 | PyTorch GraphSAGE | 2-layer GraphSAGE | best PR-AUC and strongest recall among the implemented graph tracks |
| 2 | Combined features plus embeddings | Random Forest | still the strongest classical graph baseline and the best current ROC-AUC |
| 3 | Graph features only | Random Forest | strongest pure handcrafted topology baseline |
| 4 | Embeddings only | Random Forest | useful but materially weaker on recall and PR-AUC |

## Interpretation

1. embeddings alone are not yet enough to outperform the handcrafted graph topology features
2. embeddings become useful when combined with graph-native statistics
3. the GraphSAGE refactor now surpasses the classical hybrid model on PR-AUC while remaining close on ROC-AUC
4. the local validation runtime used the PyG k-hop fallback because `pyg-lib` or `torch-sparse` is not installed here, but the training loop is already wired to switch to native `NeighborLoader` when that backend exists on the remote GPU host
5. the project now has a validated bridge from classical graph analytics into a scalable PyG mini-batch training path

## Next Comparison Steps

1. add Node2Vec-like embeddings for a stronger non-neural embedding baseline
2. install `pyg-lib` or `torch-sparse` on the target Colab runtime so the same code path uses native `NeighborLoader`
3. tune the GraphSAGE decision threshold and loss weighting to recover more precision while preserving recall
4. compare graph tracks against a canonical account-graph task, not only archive-node fraud prediction