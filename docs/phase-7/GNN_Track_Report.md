# GNN Track Report

## Purpose

Document the first neural graph baseline added on top of the AMLSim archive graph pipeline.

## Implemented Baseline

- model family: PyTorch Geometric 2-layer GraphSAGE
- execution mode: mini-batch neighborhood sampling with GPU auto-detection
- graph source: `data/external/AMLSim/sample/20K_fanin200cycle200.tgz`
- feature source: streamed archive-native node and edge aggregates
- node input width: 12 features
- hidden size: 64
- dropout: 0.3
- optimizer: Adam
- learning rate: 0.01
- weight decay: 0.0005
- early stopping patience: 20 epochs
- default loader profile: batch size 1024, fan-out [25, 10]

## Training Result

Artifact: `artifacts/graph/pytorch_gcn_metrics.json`

| Metric | Value |
| --- | --- |
| PR-AUC | 0.7880 |
| Precision | 0.4674 |
| Recall | 0.7783 |
| F1 | 0.5840 |
| ROC-AUC | 0.9113 |
| Best epoch | 75 |
| Best validation PR-AUC | 0.7658 |

Checkpoint artifact: `artifacts/graph/pytorch_graphsage_model.pt`

## Interpretation

1. the GraphSAGE refactor is now a real scalable neural baseline rather than a full-batch experiment
2. PR-AUC now exceeds every classical graph track on the same archive task while keeping recall high
3. the current local runtime used the PyG k-hop fallback because `pyg-lib` or `torch-sparse` is not installed here; on the target Colab GPU environment the same code will switch to `NeighborLoader` automatically when that backend is available
4. the neural path no longer depends on NetworkX graph construction or in-memory sparse adjacency normalization

## Current Implementation Surface

- trainer: `src/fri/models/pytorch_gnn.py`
- entrypoint: `scripts/train_pytorch_gcn.py`
- config: `configs/default.yaml` under `gnn`

## Next GNN Steps

1. install `pyg-lib` or `torch-sparse` on the remote Colab runtime to activate native `NeighborLoader`
2. tune the prediction threshold and class weighting for better precision-recall balance
3. compare the current archive-native feature set against richer transaction-time features and learned edge attributes
