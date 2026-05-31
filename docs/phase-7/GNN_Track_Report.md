# GNN Track Report

## Purpose

Document the first neural graph baseline added on top of the AMLSim archive graph pipeline.

## Implemented Baseline

- model family: pure-PyTorch 2-layer GCN
- execution mode: CPU, full-batch sparse propagation
- graph source: `data/external/AMLSim/sample/20K_fanin200cycle200.tgz`
- feature source: `combined` graph bundle from `src/fri/graph/service.py`
- node input width: 27 features
- hidden size: 64
- dropout: 0.3
- optimizer: Adam
- learning rate: 0.01
- weight decay: 0.0005
- early stopping patience: 20 epochs

## Training Result

Artifact: `artifacts/graph/pytorch_gcn_metrics.json`

| Metric | Value |
| --- | --- |
| PR-AUC | 0.5660 |
| Precision | 0.2671 |
| Recall | 0.7982 |
| F1 | 0.4002 |
| ROC-AUC | 0.8643 |
| Best epoch | 86 |
| Best validation PR-AUC | 0.5791 |

## Interpretation

1. the GCN is already a meaningful neural baseline, not just a placeholder integration
2. recall is materially higher than the classical graph baselines, which suggests message passing is surfacing suspicious neighborhoods effectively
3. precision is still weak, so the immediate next step is calibration, threshold tuning, and possibly a cleaner feature/input contract
4. staying in pure PyTorch keeps the environment simple while the repo proves out the graph-neural task definition

## Current Implementation Surface

- trainer: `src/fri/models/pytorch_gnn.py`
- entrypoint: `scripts/train_pytorch_gcn.py`
- config: `configs/default.yaml` under `gnn`

## Next GNN Steps

1. compare `combined` input features against a leaner graph-topology-only input set
2. tune the prediction threshold and class weighting for better precision-recall balance
3. add GraphSAGE or PyG only after this CPU-first baseline stops being the bottleneck
