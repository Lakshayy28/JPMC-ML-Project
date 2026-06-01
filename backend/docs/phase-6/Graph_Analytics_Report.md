# Graph Analytics Report

## Purpose

Summarize the current graph analytics results from the bundled AMLSim archive and the first graph-driven baseline.

## Analyzed Graph

- archive sample: `20K_fanin200cycle200`
- source artifact: `data/external/AMLSim/sample/20K_fanin200cycle200.tgz`
- reported metadata: 20,000 nodes, 120,558 transactions, 1,803 fraud nodes, 200 cycles and 200 fan-in patterns

## Computed Graph Summary

Artifact: `artifacts/graph/archive_graph_summary.json`

| Metric | Value |
| --- | --- |
| Node count | 20,000 |
| Edge count | 117,341 |
| Density | 0.000293 |
| Average in-degree | 5.8671 |
| Average out-degree | 5.8671 |
| Weak component count | 21 |
| Largest weak component size | 19,980 |
| Average clustering | 0.004445 |

## Observations

1. the graph is sparse but highly connected, with almost all nodes concentrated in one giant weakly connected component
2. the archive contains a meaningful fraud population, making it suitable for graph-node classification experiments
3. the highest PageRank and out-degree nodes include both fraud and non-fraud nodes, which is expected in mixed fan-in and cycle patterns

## High-Risk Structural Signals Observed

- several of the highest out-degree nodes are labeled as fraud in the archive
- fraud nodes appear among the top-ranked PageRank nodes, indicating that ring and funnel structures create visible structural prominence
- community detection identifies medium and large communities inside the giant connected component, which is useful for future ring discovery workflows

## Graph Baseline Outcome

Artifact: `artifacts/graph/graph_baseline_metrics.json`

| Model | PR-AUC | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.5065 | 0.6269 | 0.2683 | 0.3758 | 0.8887 |
| Random Forest | 0.6239 | 0.6811 | 0.4878 | 0.5685 | 0.9191 |

## Interpretation

The graph-only feature set is already learning meaningful fraud structure from the AMLSim archive. The random forest baseline materially improves recall and F1 over the linear model, which supports the project direction of combining graph-derived signals with richer models in later phases.

## Next Graph Steps

1. compute graph features on the canonical account graph built from processed tables
2. add Node2Vec or random-walk embeddings for graph-enhanced classical baselines
3. add a first GNN track once the environment is ready for PyTorch-based training
4. connect graph outputs to the API and dashboard layers after label provenance is strengthened