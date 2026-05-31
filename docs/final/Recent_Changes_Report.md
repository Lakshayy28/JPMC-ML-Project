# Recent Changes Report

## Purpose

Document the most recent implementation changes made to the Financial Risk Intelligence repository, the rationale behind them, what was validated, and the latest model performance snapshot after the refactor.

## Reporting Date

- 2026-06-01

## Summary

Two major change sets were completed recently:

1. Training visibility and orchestration improvements across all model tracks.
2. System-wide archive unification onto the 20K AMLSim graph sample, plus a refactor of the PyTorch graph model from a homogeneous GraphSAGE path to a heterogeneous, edge-aware GAT pipeline.

These changes removed the previous mismatch where tabular baselines were trained on a tiny processed sample while graph models were trained on the 20K archive.

## Change Set 1: Training Visibility and Evaluation Orchestration

### What Changed

- Added structured terminal progress logging to:
  - `scripts/train_baseline.py`
  - `scripts/train_graph_baseline.py`
  - `scripts/train_graph_embedding_baseline.py`
  - `src/fri/models/pytorch_gnn.py`
- Added a new orchestration utility:
  - `scripts/run_complete_evaluation.py`
- The orchestration script now:
  - runs all four training pipelines in sequence
  - stops immediately on subprocess failure via `check=True`
  - reloads fresh JSON metrics from `artifacts/`
  - prints a consolidated markdown comparison table across model tracks

### Why It Was Needed

- Previous training runs were largely silent.
- There was no single command that retrained every track and produced one comparable performance view.
- This made debugging, monitoring training progress, and reviewing experiment results unnecessarily difficult.

### Operational Result

- Baseline training now prints dataset loading, feature extraction, and per-estimator progress.
- The neural training path now prints epoch progress on epoch 1 and every 10 epochs.
- A single command now refreshes the artifact set:

```bash
python scripts/run_complete_evaluation.py
```

## Change Set 2: Archive Unification and Advanced GNN Refactor

### What Changed

The following files were updated to move the entire evaluation stack onto the same 20K archive dataset:

- `configs/default.yaml`
- `src/fri/graph/service.py`
- `src/fri/models/pytorch_gnn.py`
- `scripts/train_baseline.py`
- `scripts/train_graph_baseline.py`
- `scripts/train_graph_embedding_baseline.py`
- `scripts/train_pytorch_gcn.py`
- `scripts/run_complete_evaluation.py`
- selected unit tests under `tests/unit/`

### Data Layer Unification

- `train_baseline.py` no longer uses the small processed canonical sample as its training source.
- A shared archive feature bundle is now built directly from `data/external/AMLSim/sample/20K_fanin200cycle200.tgz`.
- The bundle is constructed in `src/fri/graph/service.py` and provides:
  - normalized archive nodes
  - normalized archive transactions
  - account-level tabular features derived from the full 20K archive
  - graph node features
  - spectral embeddings
  - derived merchant entities and merchant interaction links
  - 1-day, 7-day, and 30-day temporal velocity features

### Tabular Track Changes

- Tabular baselines now train on archive-derived account features instead of the former 133-row sample-backed feature matrices.
- This makes the tabular results directly comparable with graph and neural results.

### Graph Classical and Embedding Track Changes

- Graph classical baselines now consume the same shared archive-derived node feature bundle.
- The graph embedding track now merges archive-derived temporal and structural features with spectral embeddings built on the same 20K graph.

### PyTorch GNN Changes

`src/fri/models/pytorch_gnn.py` was substantially refactored.

The old path:

- homogeneous `Data`
- GraphSAGE-style message passing
- fixed decision threshold

The new path:

- heterogeneous `HeteroData`
- two node types:
  - `account`
  - `merchant`
- three edge relations:
  - `account -> transfers -> account`
  - `account -> buys_from -> merchant`
  - `merchant -> rev_buys_from -> account`
- edge-aware `GATConv` layers wrapped in `HeteroConv`
- edge attributes injected into message passing:
  - amount
  - event time
  - derived transaction type code
- temporal velocity features concatenated into account and merchant inputs before training
- dynamic validation threshold sweep across probabilities from 0.05 to 0.95

### Important Caveat

The 20K archive does not expose native merchant entities or native transaction type columns. Because of that:

- merchant nodes are currently deterministic derived buckets based on archive target-node patterns
- transaction type encoding is also derived rather than sourced from raw merchant semantics

This keeps the hetero architecture functional and reproducible, but it is still an approximation of a richer heterogeneous financial graph.

## Validation Performed

### Targeted Validation

The following focused validation was run successfully:

```bash
python -m pytest tests/unit/test_pytorch_gnn.py tests/unit/test_graph_embeddings.py tests/unit/test_graph.py
python -m py_compile src/fri/models/pytorch_gnn.py src/fri/graph/service.py scripts/train_baseline.py scripts/train_graph_baseline.py scripts/train_graph_embedding_baseline.py scripts/train_pytorch_gcn.py scripts/run_complete_evaluation.py
```

### End-to-End Validation

The full training and evaluation orchestration completed successfully:

```bash
python scripts/run_complete_evaluation.py
```

This regenerated:

- `artifacts/baseline_metrics.json`
- `artifacts/graph/graph_baseline_metrics.json`
- `artifacts/graph/graph_embedding_metrics.json`
- `artifacts/graph/pytorch_gcn_metrics.json`

## Latest Performance Snapshot

The latest consolidated metrics after archive unification are below.

| Track | Model | Precision | Recall | F1-Score | PR-AUC | ROC-AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Tabular | Logistic Regression | 0.9835 | 0.6608 | 0.7905 | 0.7779 | 0.8887 |
| Tabular | Random Forest | 0.9517 | 0.6120 | 0.7449 | 0.7854 | 0.9098 |
| Graph Classical | Logistic Regression | 0.9625 | 0.6829 | 0.7990 | 0.8410 | 0.9432 |
| Graph Classical | Random Forest | 0.9627 | 0.6297 | 0.7614 | 0.8633 | 0.9613 |
| Graph Embedding | Embedding Only Logistic Regression | 0.7727 | 0.0377 | 0.0719 | 0.2151 | 0.6536 |
| Graph Embedding | Embedding Only Random Forest | 0.7738 | 0.1441 | 0.2430 | 0.3457 | 0.7144 |
| Graph Embedding | Combined Features + Embeddings Logistic Regression | 0.9536 | 0.6829 | 0.7959 | 0.8389 | 0.9423 |
| Graph Embedding | Combined Features + Embeddings Random Forest | 0.9690 | 0.6231 | 0.7584 | 0.8590 | 0.9630 |
| PyTorch Geometric GNN | Hetero GAT | 0.7319 | 0.6962 | 0.7136 | 0.7417 | 0.9186 |

## Interpretation

- The tabular trap has been removed from the evaluation path. All principal model families now operate on features derived from the same 20K archive.
- Classical graph-aware models remain the strongest overall performers on F1 and PR-AUC in the current implementation.
- The new heterogeneous GAT path is valid, reproducible, and richer than the previous homogeneous neural baseline, but it does not yet outperform the strongest classical archive baselines.
- The current Hetero GAT run selected an optimal threshold of `0.50` during validation.

## Current Repository Impact

Recent work materially improved the repository in four ways:

1. Comparisons are now apples-to-apples across model families.
2. Training runs are easier to observe and debug in real time.
3. The graph neural stack now supports heterogeneous structure, edge attributes, and temporal inputs.
4. The evaluation pipeline can be rerun from one command and produces an updated consolidated metrics view.

## Recommended Next Steps

1. Tune the Hetero GAT architecture further, especially hidden size, number of heads, dropout, and class weighting.
2. Replace derived merchant proxies with real merchant semantics if a richer AMLSim export or additional data source becomes available.
3. Add experiment tracking for threshold, architecture, and feature bundle variants so neural iterations can be compared systematically.