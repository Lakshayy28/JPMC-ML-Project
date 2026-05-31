# Phase 9 Explainability Changes

## Purpose

Document the latest Phase 9 implementation work that added model explainability for the heterogenous graph attention network used in the Financial Risk Intelligence repository.

## Reporting Date

- 2026-06-01

## Summary

Phase 9 is now implemented as a first working explainability slice for the trained heterogenous GAT model.

The repository now includes:

- a reusable explainability service built on PyTorch Geometric's native `Explainer`
- a command-line execution script that explains the highest-risk account in the 20K archive graph
- a persisted explanation artifact under `artifacts/graph/`
- unit coverage for the new explainability path

## Files Added Or Updated

### New Files

- `src/fri/explainability/__init__.py`
- `src/fri/explainability/service.py`
- `scripts/run_explainability.py`
- `tests/unit/test_explainability.py`

### Supporting Updates

- `src/fri/models/pytorch_gnn.py`
- `src/fri/graph/service.py`

## What Was Implemented

### Explainability Service

The new service lives in `src/fri/explainability/service.py`.

It provides:

- `NodeExplanationReport`
  - `node_id`
  - `risk_score`
  - `top_node_features`
  - `critical_edges`
- `HeteroGraphExplainerService`

The service wraps PyG's explainability stack using:

- `Explainer`
- `GNNExplainer(epochs=200)` by default
- `explanation_type='model'`
- `node_mask_type='attributes'`
- `edge_mask_type='object'`
- node-level multiclass configuration for the `account` prediction task

The service can explain a specific account node in the heterogenous graph and returns a structured report with:

- feature attributions mapped back to the human-readable account feature names stored in model metrics
- ranked structural edges linked to the explained account
- readable edge payloads using raw transaction values instead of normalized model tensors

### Explainability Execution Script

The new script lives in `scripts/run_explainability.py`.

It performs the following flow:

1. loads the unified 20K archive feature bundle
2. rebuilds the heterogenous graph bundle used by the Hetero GAT model
3. loads the trained checkpoint at `artifacts/graph/pytorch_hetero_gat_model.pt`
4. runs inference across all account nodes
5. selects the account with the highest predicted fraud probability
6. calls the explainability service on that account
7. writes the result to `artifacts/graph/account_explanation_sample.json`
8. prints a readable terminal summary of top features and critical edges

## Supporting Refactors

To make explainability work cleanly with the trained model, small public helper surfaces were added to `src/fri/models/pytorch_gnn.py`:

- graph-bundle conversion from the shared feature bundle
- deterministic normalization for inference-time graph preparation
- checkpoint-based model reconstruction for the Hetero GAT path
- a model forward path that accepts heterogenous tensor dictionaries so it can be called by PyG's `Explainer`

`src/fri/graph/service.py` was also widened so `build_graph_feature_bundle(...)` can construct the unified archive-derived bundle directly from archive nodes and transactions when needed by downstream scripts.

## Validation Performed

### Targeted Validation

The following focused validation completed successfully:

```bash
python -m pytest tests/unit/test_explainability.py tests/unit/test_pytorch_gnn.py
python -m py_compile src/fri/explainability/service.py src/fri/models/pytorch_gnn.py src/fri/graph/service.py scripts/run_explainability.py
```

### End-to-End Validation

The real explainability flow was validated against the trained 20K archive model using a short explainer pass:

```bash
python scripts/run_explainability.py --explainer-epochs 3 --top-k-features 3 --top-k-edges 3
```

This successfully generated:

- `artifacts/graph/account_explanation_sample.json`

## Current Sample Explanation Output

The latest validated sample explanation identified account `19204` as the highest-risk node in the current model run.

### Predicted Risk Score

- fraud probability: `0.9985`

### Top Feature Attributions

- `incoming_amount_velocity_1d`
- `outgoing_counterparty_velocity_1d`
- `incoming_counterparty_velocity_7d`

### Top Structural Contributors

The strongest attributed structural signals in the validated sample were direct account-to-account transfer edges involving account `19204`, including raw transfer amounts and event times such as:

- transfer `19204 -> 5966`, amount `18.69`, event time `136`
- transfer `16461 -> 19204`, amount `286.54`, event time `71`
- transfer `10976 -> 19204`, amount `449.26`, event time `32`

## Operational Notes

- The explainability script defaults to `200` explainer epochs for fuller attribution quality.
- Validation used a shorter run only to confirm the full pipeline and artifact generation quickly.
- Edge reports now use raw archive transaction values so the output is readable for analyst-style review.

## Current Limitations

- Merchant nodes are still deterministic derived proxies because the 20K archive does not expose native merchant entities.
- Transaction type encoding is also derived rather than sourced from a richer raw event schema.
- The explanation artifact currently focuses on the highest-risk account sample, not batch explanation across many accounts.

## Recommended Next Steps

1. Add batch explainability for the top-N risky accounts instead of only the single highest-risk node.
2. Add feature-group summaries so temporal, structural, and merchant-linked contributions can be reviewed at a higher level.
3. Integrate explanation artifacts into a future analyst dashboard or API layer.