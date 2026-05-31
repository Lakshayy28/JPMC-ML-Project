# Training, Inference, and XAI

## Purpose

Document how the Financial Risk Intelligence Engine trains the deployed Hetero GAT model, selects an operational decision threshold, serves inference through the API, and generates graph-native explanations for flagged accounts.

## Executive Summary

The current deep learning path is implemented in `src/fri/models/pytorch_gnn.py` and operationalized through:

- `scripts/train_pytorch_gcn.py`
- `src/fri/api/state.py`
- `src/fri/api/main.py`
- `src/fri/explainability/service.py`
- `scripts/run_explainability.py`

Despite some legacy naming inherited from earlier GCN work, the active model and metrics identify the deployed architecture as `pytorch_hetero_gat`.

The training path is not a classical mini-batch neighbor-sampling pipeline in the current repository state. It is a full-batch heterogenous graph training loop with explicit train, validation, and test index slices over a single prepared `HeteroData` object. That distinction matters for accurate documentation.

## Training Strategy

### Training Entry Point

`scripts/train_pytorch_gcn.py` is the top-level execution script for the graph deep learning track.

It performs the following steps:

1. load runtime settings from `configs/default.yaml`
2. resolve the configured 20K archive path
3. build a PyTorch Geometric heterogenous graph bundle from the archive
4. resolve the training device
5. call `train_pytorch_gcn(...)`
6. persist the metrics to `artifacts/graph/pytorch_gcn_metrics.json`
7. persist the trained checkpoint to `artifacts/graph/pytorch_hetero_gat_model.pt`

### Full-Batch Heterogenous Training

The actual optimization logic is implemented in `train_pyg_minibatch(...)` inside `src/fri/models/pytorch_gnn.py`.

The function name is a historical artifact. In the current code path, the training loop explicitly discards loader-oriented arguments:

- `batch_size`
- `fan_out`
- `num_workers`

and then runs full-batch optimization over the already materialized heterogenous graph.

The sequence is:

1. stratify and split account labels into train, validation, and test indices where possible
2. normalize account features using the training slice statistics
3. normalize merchant features and edge attributes
4. move the graph to the resolved device
5. construct the Hetero GAT model
6. optimize on the train indices while evaluating on the validation indices

This is a full-graph node-classification regime rather than neighborhood-sampled stochastic training.

### Loss Function And Class Imbalance Handling

The training loop explicitly addresses class imbalance by computing class weights from the training indices.

The weight vector is built as:

```text
[1.0, (negative_count / positive_count) * pos_weight_multiplier]
```

and passed into:

- `nn.CrossEntropyLoss(weight=class_weights)`

This means the positive fraud class is upweighted relative to the negative class according to the imbalance ratio, scaled by the configured `pos_weight_multiplier`.

In the active persisted metrics artifact, the deployed run records:

- `pos_weight_multiplier = 0.5`

This indicates the imbalance correction was applied but deliberately moderated rather than left at the raw negative-to-positive ratio.

### Optimizer And Regularization

The training loop uses:

- `torch.optim.Adam`

with configurable:

- learning rate
- weight decay
- dropout

The active defaults wired from configuration are:

- `learning_rate = 0.01`
- `weight_decay = 0.0005`
- `dropout = 0.3`

These parameters govern optimization stability, regularization, and representation robustness during graph training.

### Early-Stopping Lookahead Mechanism

The training loop maintains a patience-driven model-selection process over the validation slice.

The relevant state variables are:

- `best_selection_score`
- `best_val_ap`
- `best_val_f1`
- `best_threshold`
- `best_state`
- `patience_counter`

On every epoch:

1. the model trains on the train indices
2. the validation slice is evaluated
3. the optimal threshold for that validation slice is estimated
4. a selection score is computed
5. if the score improves, the best model state is checkpointed and patience resets
6. if the score does not improve, patience increments

Training stops once:

- `patience_counter >= patience`

This is effectively an early-stopping lookahead mechanism keyed to validation behavior rather than just raw training loss.

### Validation Selection Criterion

The repository uses a practical selection heuristic:

- prefer validation F1 when available
- otherwise fall back to negative validation loss

This is important because it aligns model selection with classification usefulness rather than optimizing only for loss reduction.

## Threshold Optimization

### Why Threshold Optimization Exists

Fraud scoring systems should not assume that `0.50` is always the best operational decision boundary.

The repository explicitly addresses this by sweeping thresholds on the validation set.

### Implementation Details

`_optimal_threshold(...)` evaluates thresholds from:

- `0.05` to `0.95`

using 19 evenly spaced points.

For each threshold, it computes:

- binary predictions from validation probabilities
- F1 score against validation labels

It then selects the threshold that yields the best validation F1.

### What Gets Persisted

The training payload written to `artifacts/graph/pytorch_gcn_metrics.json` stores:

- `optimal_threshold`
- `decision_threshold`
- `configured_decision_threshold`

This is important because the system preserves both:

- the configured starting threshold from settings
- the optimized threshold selected from validation behavior

In the currently persisted metrics artifact, the deployed run records:

- `optimal_threshold = 0.49999999999999994`
- `decision_threshold = 0.49999999999999994`
- `configured_decision_threshold = 0.5`

In practice, this run converged back to a 0.50 operating point, but that was the result of an explicit sweep rather than an assumption.

## Inference Path

### Inference In The Serving Engine

The inference logic used by the API lives in `EngineState.predict_account(...)` inside `src/fri/api/state.py`.

The supporting probability path is implemented in `EngineState.risk_probabilities()`.

That function performs:

```python
with torch.no_grad():
    logits = self.model(self.data)
    probabilities = torch.softmax(logits, dim=1)[:, 1]
```

This matters for serving because `torch.no_grad()` disables gradient tracking and therefore reduces runtime overhead and memory consumption during online scoring.

### `/predict` Route Behavior

The FastAPI route `GET /predict/{account_id}` does the following:

1. retrieve the singleton `EngineState`
2. map the external account ID to the internal graph node index
3. run inference against the already loaded heterogenous graph
4. read the fraud probability for that node
5. compare the score against the stored serving threshold
6. return:
   - `account_id`
   - `fraud_probability`
   - `is_high_risk`
   - `threshold_used`

The speed of this path does not come from batch decomposition. It comes from the fact that the model, graph, normalization state, and threshold are already resident in `EngineState`.

### Inference Data Preparation

The runtime graph is prepared through `prepare_hetero_inference_data(...)`, which:

- derives a train split for normalization statistics
- normalizes account features using train-only moments
- normalizes merchant features separately
- normalizes edge attributes per relation type
- moves the data to the resolved device

This ensures the live API uses feature scaling consistent with the training assumptions encoded into the checkpoint.

## Explainability (XAI)

### Phase 9 Overview

Phase 9 introduced a dedicated explainability layer implemented in `src/fri/explainability/service.py`.

The repository does not treat explainability as a post-hoc dashboard-only concern. It exposes model introspection as a first-class service that can be called both offline and through the live API.

### Core Explainability Service

The main abstraction is:

- `HeteroGraphExplainerService`

It returns:

- `NodeExplanationReport`
  - `node_id`
  - `risk_score`
  - `top_node_features`
  - `critical_edges`

### PyG Explainer Integration

The service constructs a PyTorch Geometric `Explainer` with:

- `algorithm = GNNExplainer(...)`
- `explanation_type = "model"`
- `node_mask_type = "attributes"`
- `edge_mask_type = "object"`

and model configuration:

- `mode = "multiclass_classification"`
- `task_level = "node"`
- `return_type = "raw"`

This is not a generic SHAP-style feature explainer retrofitted onto graph outputs. It is a graph-native explainer that operates directly on the heterogenous graph model.

### How Account Explanations Are Generated

`HeteroGraphExplainerService.explain_account(...)` performs two separate operations.

First, it computes the predicted fraud probability for the requested account with the live model.

Second, it invokes the PyG explainer on:

- `data.x_dict`
- `data.edge_index_dict`
- `edge_attr_dict`
- `index = account_index`

That means the explanation is tied to a specific account node within the heterogenous graph rather than a detached feature vector.

### Human-Readable Feature Attribution

The explainer’s account node mask is converted back into readable ranked feature names by `_top_node_features(...)`.

Those names come from the persisted training metrics artifact, which stores the deployed account feature column ordering.

This is operationally important because the API returns explanations in analyst-readable terms such as:

- `incoming_amount_velocity_1d`
- `weighted_in_degree`
- `unique_merchant_count`

rather than opaque tensor positions.

### Human-Readable Structural Edge Reporting

The explainability service also reconstructs structurally important incident edges.

It scans the attributed edge masks across all graph relation types and produces ranked edge reports containing:

- relation type
- edge importance
- source and target node IDs
- transaction attributes

Where possible, those reports are populated from stored raw transaction records rather than only normalized tensor values. This is a critical design choice because it makes the explanations interpretable for investigators and auditors.

### API Explainability Budgeting And Caching

The live API explanation path uses:

- `epochs = 50`

for the explainer budget, while the offline script defaults to heavier runs such as 200 epochs.

Additionally, `EngineState` wraps explanation generation in an LRU cache keyed by account ID and epoch budget.

That combination gives the deployed system two distinct explanation modes:

- a lighter synchronous API mode for operational latency control
- a deeper offline mode for artifact generation and investigation

### Offline Explainability Script

`scripts/run_explainability.py` performs an end-to-end offline explanation workflow:

1. rebuild the unified archive feature bundle
2. reconstruct the heterogenous graph
3. load the trained checkpoint
4. run inference across all account nodes
5. select the highest-risk account
6. run the explainability service on that node
7. write the explanation artifact to `artifacts/graph/account_explanation_sample.json`

This script is useful because it proves that the explainability path is not coupled only to the API layer. It can also be used for validation, artifact generation, and analyst review.

## What The Current Training Artifact Records

The persisted graph metrics artifact currently reports:

- `model_name = pytorch_hetero_gat`
- `loader_backend = full_batch_hetero`
- `device = cpu`
- `best_epoch = 120`
- `best_validation_average_precision = 0.7199`
- `best_validation_f1 = 0.7136`
- `average_precision = 0.7417`
- `precision = 0.7319`
- `recall = 0.6962`
- `f1 = 0.7136`
- `roc_auc = 0.9186`

These values indicate the repository is not merely configured for the Hetero GAT path. It has an actual trained and evaluated checkpoint that the live service is using.

## Summary

The current training and inference architecture is a disciplined graph-learning pipeline rather than a loose experiment.

Training uses a full-batch heterogenous graph optimization loop with class-weighted loss, validation-driven early stopping, and explicit threshold tuning. Serving uses a preloaded graph and model under `torch.no_grad()` for efficient scoring. Explainability is implemented as a graph-native PyTorch Geometric explainer that converts node and edge attributions back into readable operational evidence.

Together, those pieces form a coherent lifecycle:

- train a heterogenous graph fraud model on the unified archive
- optimize its decision threshold on validation behavior
- serve fast account-level inference through FastAPI
- explain suspicious predictions in terms that investigators can review