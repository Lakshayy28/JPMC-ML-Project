# Graph Feature Service Spec

## Purpose

Define the current graph feature service used to produce graph-native features and embeddings for downstream model training.

## Implemented Service

Current implementation: `src/fri/graph/service.py`

### Function

`build_graph_feature_bundle(graph, include_communities, community_seed, embedding_dimensions, embedding_random_state)`

### Outputs

| Output | Meaning |
| --- | --- |
| `node_features` | handcrafted graph-topology features from `compute_node_features` |
| `embeddings` | spectral graph embeddings from `compute_spectral_node_embeddings` |
| `combined` | merged feature set containing both topology features and embeddings |

## Current Inputs

The current service operates on a `networkx.DiGraph` built from:

- AMLSim archive transaction graphs
- canonical account graphs from processed tables

## Topology Features Produced

- in-degree
- out-degree
- weighted in-degree
- weighted out-degree
- PageRank
- clustering coefficient
- weak-component membership and size
- community membership and size when enabled

## Embedding Features Produced

- spectral embeddings from sparse graph adjacency using truncated SVD
- default dimensionality: 16
- current persisted archive embedding artifact: `artifacts/graph/archive_node_embeddings.csv`

## Design Rules

1. feature bundles are deterministic under a fixed graph and random seed
2. embeddings are graph-derived and should be recomputed whenever graph structure changes
3. labels are not produced by the service; they are joined downstream from node attributes or canonical tables
4. the service is offline-first for training and benchmarking, not yet an online serving API

## Next Service Extensions

1. add canonical account graph feature bundles as persisted artifacts
2. add Node2Vec-style random-walk embeddings
3. add GNN latent representation extraction once a PyTorch training stack is in place