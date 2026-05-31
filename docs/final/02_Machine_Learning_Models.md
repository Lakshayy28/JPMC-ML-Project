# Machine Learning Models

## Purpose

Document the model stack implemented in the Financial Risk Intelligence Engine, with emphasis on the transition from inconsistent experimental comparisons to a unified archive-based evaluation framework and the final Heterogeneous Graph Attention Network used in production serving.

## Executive Summary

The repository no longer compares unrelated modeling tracks on mismatched data slices. The current design unifies tabular baselines, graph classical baselines, graph-embedding baselines, and the deployed deep learning model around the same 20K AMLSim archive sample.

This change is critical because it converts model comparison from a misleading research artifact into a meaningful engineering benchmark.

At the current repository state, the model family consists of four major layers:

- archive-derived account-level tabular baselines
- archive-derived graph classical baselines
- archive-derived embedding and combined graph-feature baselines
- a deployed PyTorch Geometric Hetero GAT checkpoint for live inference

## The "Tabular Trap" Resolution

### The Original Comparison Risk

In mixed experimental repositories, one of the most common failure modes is evaluating different model families on different data abstractions.

That problem has effectively been removed here.

The current training entrypoints show that all major tracks now begin from the same 20K AMLSim archive:

- `scripts/train_baseline.py`
- `scripts/train_graph_baseline.py`
- `scripts/train_graph_embedding_baseline.py`
- `scripts/train_pytorch_gcn.py`

Each of those scripts loads the same archive source from configuration and routes through the same archive feature-construction logic in `src/fri/graph/service.py`.

### What Was Unified

The important architectural change is that the system no longer treats tabular modeling as a separate world with a separate tiny data representation.

Instead:

- the tabular baseline uses `tabular_account_features` from the archive feature bundle
- the graph classical baseline uses `node_features` from the same archive feature bundle
- the embedding baseline uses `embeddings` and `combined` frames from the same archive feature bundle
- the Hetero GAT path converts the same bundle into PyTorch Geometric `HeteroData`

This means the model comparisons are now apples-to-apples at the data-source level. The difference between tracks is model family and representation depth, not a different upstream dataset.

### Why This Matters

This unification resolves the tabular trap in three ways:

- feature provenance is consistent across model families
- labels are sourced from the same archive node population
- measured differences are attributable to modeling choices rather than dataset mismatch

For an enterprise engineering team or model-risk auditor, this is a much more defensible evaluation posture than comparing unrelated feature spaces drawn from unrelated samples.

## Feature Engineering Stack

### Unified Archive Feature Bundle

`src/fri/graph/service.py` is the main feature-engineering surface for the deployed modeling stack.

Its `build_archive_feature_bundle(...)` function constructs a shared package containing:

- normalized archive nodes
- normalized archive transactions
- account tabular features
- graph node features
- merchant features
- merchant links
- optional graph embeddings
- combined graph-feature-plus-embedding frames

This is the canonical modeling substrate used by the current baseline and GNN tracks.

### Account Behavioral Features

The account feature pipeline begins with `_account_flow_features(...)`, which derives account-level aggregates such as:

- outgoing transaction count
- outgoing amount total and mean
- outgoing counterparty count
- outgoing first and last event time
- incoming transaction count
- incoming amount total and mean
- incoming counterparty count
- incoming first and last event time
- outgoing and incoming time spans
- total transaction count
- total amount
- net outgoing amount

These features provide the basic financial-flow summary for each account node.

### Temporal Rolling Windows

The temporal enrichment layer is implemented through `_window_account_features(...)` and uses the configured windows:

- 1 day
- 7 days
- 30 days

For each window, the system computes rolling behavioral rates such as:

- `outgoing_tx_velocity_{window}d`
- `incoming_tx_velocity_{window}d`
- `outgoing_amount_velocity_{window}d`
- `incoming_amount_velocity_{window}d`
- `outgoing_counterparty_velocity_{window}d`
- `incoming_counterparty_velocity_{window}d`

These are the temporal features that let the system encode acceleration, burstiness, and short-vs-long horizon transaction behavior rather than relying only on static totals.

### Graph Structural Features

The same archive is also transformed into a transaction graph, after which graph-level structural features are computed and merged back into the account representation.

The resulting account feature set includes structural signals such as:

- `in_degree`
- `out_degree`
- `weighted_in_degree`
- `weighted_out_degree`
- `pagerank`
- `clustering_coefficient`
- weak-component identifiers and sizes
- community identifiers and sizes

This is where the system moves beyond conventional tabular fraud scoring into explicitly relational reasoning.

### Derived Merchant Nodes

The 20K AMLSim archive does not expose native merchant entities directly in the way a payments platform might.

To support a heterogenous graph, `src/fri/graph/service.py` derives merchant proxies through `_derive_archive_merchants(...)`.

That process creates:

- a `merchant_id` derived from a stable bucketization strategy
- merchant transaction aggregates
- merchant temporal activity features
- account-to-merchant interaction summaries such as:
  - `merchant_transaction_count`
  - `merchant_total_amount`
  - `unique_merchant_count`

This merchant layer is the key reason the deployed graph is heterogenous rather than homogeneous.

### Embeddings And Combined Views

When embedding generation is enabled, the archive graph is passed through the spectral embedding path and merged back into the graph feature frame.

The repository therefore supports three graph-centric baseline views:

- graph features only
- embeddings only
- graph features plus embeddings

That makes the experimental ladder more informative because it separates structural summary features from learned low-dimensional graph coordinates.

## Classical Baselines

### Shared Baseline Estimators

The core tabular estimator logic lives in `src/fri/models/baseline.py`.

All baseline tracks use the same estimator pair:

- Logistic Regression
- Random Forest Classifier

The preprocessing stack is also shared:

- median imputation plus standardization for numeric features
- most-frequent imputation plus one-hot encoding for categorical features

This shared baseline framework ensures the classical model comparisons are consistent across different feature representations.

### Archive Account Tabular Baselines

`scripts/train_baseline.py` trains the archive account tabular baseline.

The important detail is that it no longer trains on a disconnected processed-table sandbox. It uses the archive-derived `tabular_account_features` frame, converts `is_fraud` into the training label, and evaluates the baseline estimators on that unified account-level feature space.

This is the direct resolution of the earlier tabular mismatch issue.

### Graph Classical Baselines

`scripts/train_graph_baseline.py` uses `src/fri/models/graph_baseline.py` to train classical estimators on archive-derived graph node features.

The model family is still Logistic Regression plus Random Forest, but the input space now contains graph-structural variables merged with temporal account features.

This creates a fair comparison between:

- non-relational tabular account summaries
- relationally informed graph node summaries

### Embedding-Only And Combined Baselines

`src/fri/models/graph_embedding.py` defines two additional baseline views:

- `embedding_only`
- `combined_graph_features_and_embeddings`

Both are still trained through the shared classical baseline framework, but they operate on different graph-derived representations:

- the embedding-only track tests whether low-dimensional graph coordinates alone are predictive
- the combined track tests whether embeddings add signal beyond engineered graph and temporal features

This is a useful experimental bridge between hand-engineered graph summaries and full deep graph learning.

## Deep Learning Engine: Hetero GAT

### Model Identity

The deployed deep learning model is implemented as `SpatialTemporalHeteroGAT` in `src/fri/models/pytorch_gnn.py`.

The active checkpoint is recorded in the metrics artifact as:

- `model_name: pytorch_hetero_gat`
- checkpoint file: `artifacts/graph/pytorch_hetero_gat_model.pt`

One important repository nuance is that some filenames retain legacy `pytorch_gcn` naming, including `scripts/train_pytorch_gcn.py` and `artifacts/graph/pytorch_gcn_metrics.json`. Despite that legacy naming, the actual loaded model metadata identifies the deployed architecture as `pytorch_hetero_gat`.

### Graph Representation

The Hetero GAT path converts the unified feature bundle into PyTorch Geometric `HeteroData`.

The node types are:

- `account`
- `merchant`

The edge relations are:

- `("account", "transfers", "account")`
- `("account", "buys_from", "merchant")`
- `("merchant", "rev_buys_from", "account")`

This representation is important because it lets the model distinguish between:

- peer-to-peer account transfers
- account-to-merchant interactions
- reverse merchant-to-account message passing

That is substantially richer than flattening all connectivity into one undifferentiated adjacency matrix.

### Node And Edge Inputs

According to the persisted model metrics, the deployed graph currently uses:

- `feature_dimension = 49` for account nodes
- `merchant_feature_dimension = 18` for merchant nodes
- `edge_feature_dimension = 3` for edges

The edge feature columns are explicitly:

- `amount`
- `event_time`
- `transaction_type_code`

This means the model is not only attending over graph structure. It is also conditioning message passing on transaction-level numeric attributes, including monetary magnitude and temporal positioning.

### Internal Architecture

`SpatialTemporalHeteroGAT` contains:

- an account encoder linear layer
- a merchant encoder linear layer
- two stacked heterogenous convolution layers
- a final classifier over account-node embeddings

Each convolution stage is implemented as a `HeteroConv` that wraps relation-specific `GATConv` operators for the three edge types.

For every edge family, the GAT layers are configured with:

- relation-aware source and target dimensions
- `edge_dim=edge_dim`
- `add_self_loops=False`
- multi-head attention with `heads=4`
- `concat=False`

After each heterogenous attention block, the model applies:

- ELU activation
- dropout

The final prediction head produces account-level logits for binary fraud classification.

### How Spatial And Temporal Signals Are Fused

The model fuses spatial and temporal information in two distinct places.

First, temporal behavior is embedded directly into the node feature space through rolling-window velocity features and activity spans.

Second, transaction-level chronology is embedded into the edge feature space through `event_time`, while monetary weight is embedded through `amount`.

As a result, message passing is informed by:

- who is connected to whom
- what type of relation connects them
- how much money moved across the relation
- when the relation occurred
- how rapidly the connected account or merchant has been behaving over 1-day, 7-day, and 30-day horizons

This is the operational meaning of the repository’s spatial-temporal Hetero GAT design.

### Inference-Oriented Graph Preparation

Before serving or evaluation, the graph is normalized through `prepare_hetero_inference_data(...)`.

That process:

- creates stratified splits where possible
- normalizes account features based on the training slice
- normalizes merchant-node features
- normalizes edge attributes per relation type
- moves the heterogenous graph to the resolved runtime device

This ensures the inference path uses the same feature scaling logic that the trained checkpoint expects.

## Current Model Positioning

The repository’s model stack should be interpreted as an experimental ladder with increasing representational power:

- Logistic Regression and Random Forest on archive account tabular features
- Logistic Regression and Random Forest on archive graph node features
- Logistic Regression and Random Forest on embeddings and combined graph-feature-plus-embedding views
- full heterogenous graph attention over typed nodes and attributed edges

That ladder is technically useful because it lets the team answer progressively stronger questions:

- how much signal exists in simple account summaries
- how much additional signal exists in graph structure
- how much graph geometry is captured by embeddings alone
- how much performance is unlocked by end-to-end heterogenous message passing

## Summary

The Financial Risk Intelligence Engine now uses a model architecture strategy that is both methodologically cleaner and operationally stronger than the repository’s earlier state.

The key change was not only the introduction of the Hetero GAT itself. It was the decision to unify the entire evaluation ladder onto the same 20K archive-derived feature bundle. That makes the final Hetero GAT checkpoint a legitimate top-tier model in a coherent comparison framework rather than an isolated deep-learning experiment.

At deployment time, the system serves the `pytorch_hetero_gat_model.pt` checkpoint over a heterogenous graph built from archive-derived account features, merchant proxies, temporal rolling velocities, and relation-level edge attributes. That is the central machine learning engine behind the current production API.