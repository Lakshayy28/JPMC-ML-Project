# Project Status Report

## Purpose

Summarize what is completed in the Financial Risk Intelligence Platform repository today, what has been validated, what remains incomplete, and what the next execution steps should be.

## Executive Summary

The project is currently implemented through phase 7 in meaningful form.

Completed and validated:

- phase 0 foundational product and architecture documentation
- phase 1 fraud taxonomy, dataset assessment and entity mapping
- phase 3 canonical data layer, data validation and EDA reporting
- phase 4 baseline feature layer for transaction, party and graph-node features
- phase 5 first classical evaluation track
- phase 6 graph schema and graph analytics on a larger AMLSim archive
- phase 7 first graph embedding pipeline and hybrid graph ML comparison

Not yet implemented:

- phase 8 temporal risk intelligence
- phase 9 explainability service and dashboard outputs
- phase 10 MLflow-based MLOps lifecycle
- phase 11 FastAPI production scoring service
- phase 12 monitoring and alerting stack
- phase 13 Streamlit dashboard experience
- final portfolio packaging assets

## Current Repository State

### Dataset Choice

- canonical MVP dataset: IBM AMLSim
- local acquisition model: clone AMLSim into `data/external/AMLSim`
- helper script: `sh scripts/fetch_amlsim_repo.sh`

### Current Data Layer

Canonical processed tables are written under `data/processed/amlsim`.

Current sample-backed row counts:

| Table | Rows |
| --- | --- |
| Parties | 30 |
| Accounts | 30 |
| Transactions | 133 |
| Alerts | 2 |
| Banks | 1 |
| Devices | 14 |
| IP Addresses | 24 |
| Merchants | 19 |

Current validation status:

- required-column checks pass
- uniqueness checks pass
- referential integrity checks pass
- expected sparsity only in sample-mode transaction destination and timestamp fields

Key artifacts:

- `artifacts/data_validation_report.json`
- `artifacts/eda_summary.json`

## Current Modelling State

### Tabular Baselines

Artifact: `artifacts/baseline_metrics.json`

These baselines are pipeline-validating, but the current sample is too small for serious performance claims.

### Graph Analytics

Archive sample used for graph phases:

- `data/external/AMLSim/sample/20K_fanin200cycle200.tgz`

Computed graph summary:

| Metric | Value |
| --- | --- |
| Nodes | 20,000 |
| Edges | 117,341 |
| Density | 0.000293 |
| Weak components | 21 |
| Largest weak component | 19,980 |
| Average clustering | 0.004445 |

Key artifacts:

- `artifacts/graph/archive_graph_summary.json`
- `artifacts/graph/graph_baseline_metrics.json`

### Graph Embedding Track

Implemented approach:

- sparse adjacency projection
- truncated SVD spectral embeddings
- hybrid comparison against handcrafted graph topology features

Key artifacts:

- `artifacts/graph/graph_embedding_metrics.json`

Current best graph-side result:

| Track | Model | PR-AUC | Recall | ROC-AUC |
| --- | --- | --- | --- | --- |
| Graph features only | Random Forest | 0.6239 | 0.4878 | 0.9191 |
| Embeddings only | Random Forest | 0.3457 | 0.1441 | 0.7144 |
| Features + embeddings | Random Forest | 0.6490 | 0.4324 | 0.9224 |

Interpretation:

- embeddings alone are weaker than handcrafted graph features
- hybrid graph features plus embeddings are currently the strongest phase-7 result

## Documentation Coverage

Completed docs now exist for:

- phase 0
- phase 1
- phase 3
- phase 4
- phase 5
- phase 6
- phase 7

Most useful current entry points:

- `docs/Documentation_Roadmap.md`
- `docs/phase-3/Initial_Implementation_Summary.md`
- `docs/phase-5/Evaluation_Report.md`
- `docs/phase-7/Performance_Comparison_Report.md`

## Main Gaps

1. the processed tabular layer still uses the bundled AMLSim sample outputs rather than a richer converted AMLSim run
2. transaction labels in sample mode remain partly proxy-based rather than fully direct alert-transaction labels
3. there is no temporal drift or rolling-window engine yet
4. there is no explainability service or API layer yet
5. there is no GNN stack yet

## Recommended Next Steps

### Next Build Block

1. phase 8: add rolling-window temporal features and temporal drift reporting
2. add direct time-windowed transaction features into the canonical feature layer
3. compare temporal-enhanced tabular models with current non-temporal baselines

### Next Infra Block

1. phase 10: introduce experiment persistence and MLflow once temporal features are stable
2. phase 11: expose scoring through a minimal FastAPI service after label contracts settle

### Next Graph ML Block

1. add a stronger non-neural embedding baseline such as Node2Vec-style random walks
2. add the first GNN track using GCN or GraphSAGE
3. use GPU help only when the PyTorch-based GNN phase begins or if training scale grows materially

## When User Help May Be Needed

No GPU help is needed yet.

The first likely point to ask for help is when moving from the current CPU-friendly spectral embedding pipeline to a PyTorch/PyG-based GNN training stack, especially if you want:

- GraphSAGE or GCN training on larger AMLSim graphs
- faster repeated experiments
- more sophisticated graph mini-batching

## Commit Intent

The repository is ready to be committed in two logical chunks:

1. completed data, graph analytics and graph ML implementation through phase 7
2. project status and repository hygiene updates