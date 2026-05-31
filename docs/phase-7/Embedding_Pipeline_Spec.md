# Embedding Pipeline Spec

## Purpose

Describe the first implemented graph embedding pipeline for the project.

## Current Embedding Method

Current implementation: `src/fri/graph/embeddings.py`

### Method

- graph input: directed AMLSim transaction graph
- matrix representation: sparse adjacency matrix using `edge_count` weights
- decomposition method: `TruncatedSVD`
- default dimensionality: 16
- output: one embedding vector per node with columns `embedding_0 ... embedding_n`

## Why This Method Exists Now

This pipeline gives the project a real embedding track without waiting on:

- GPU access
- PyTorch installation and model training
- larger GNN-specific infrastructure

It is not a replacement for Node2Vec or GNNs, but it is a valid graph embedding baseline that can be trained and compared immediately.

## Pipeline Steps

1. build a directed transaction graph from AMLSim archive transactions
2. order nodes deterministically
3. convert the graph to a sparse adjacency matrix
4. apply truncated SVD to obtain dense node vectors
5. persist embeddings for downstream training and analysis

## Inputs And Outputs

| Item | Value |
| --- | --- |
| Current sample | `20K_fanin200cycle200` |
| Current output file | `artifacts/graph/archive_node_embeddings.csv` |
| Current metric artifact | `artifacts/graph/graph_embedding_metrics.json` |

## Current Limitations

1. embeddings are structural but not explicitly temporal
2. the method does not model message passing like a GCN or GraphSAGE model
3. directionality is represented only through the adjacency structure, not through a learned neural architecture
4. embeddings are currently archive-graph specific rather than canonical-account-graph specific

## Next Embedding Steps

1. add random-walk-based embeddings
2. compute embeddings on the canonical account graph built from processed tables
3. compare embedding-only, graph-feature-only and hybrid models under a shared split
4. add a GNN training track once the runtime stack is ready