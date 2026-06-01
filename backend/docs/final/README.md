# Financial Risk Intelligence Engine — Comprehensive System Documentation

## Purpose

Welcome to the definitive suite of system documentation for the Financial Risk Intelligence platform.

The codebase is split into **`backend/`** (this tree: API, ML, data, Docker) and **`frontend/`** (React dashboard at the repository root). Paths in the documents below are relative to `backend/` unless noted otherwise.

This project has evolved from a basic data science sandbox into a production-grade, containerized microservice ecosystem utilizing a Heterogeneous Graph Attention Network (Hetero GAT). This documentation is designed to serve as an authoritative, audit-ready guide for JPMC ML project engineers, reviewers, model validation teams, and operational MLOps auditors.

Each section is designed to stand on its own while contributing to a coherent, end-to-end understanding of the platform's lifecycle from raw AMLSim source data to live, explainable, and monitored containerized inference.

---

## Documentation Registry

The comprehensive system documentation is organized into the following five modules:

### 1. [01_System_Architecture.md](01_System_Architecture.md)
**High-Level Architecture and Flow Analysis**
*   **High-Level Overview:** Analysis of the system's role as a real-time anti-money-laundering (AML) risk engine.
*   **Deployment Topology:** Structure of the containerized environment, single-container setup, and published port maps.
*   **Component Layout:** End-to-end system chain highlighting the unified archive-to-graph-to-inference pipeline.
*   **Microservices vs. State:** Discussion of `EngineState` as the stateful in-memory serving singleton.
*   **Performance Optimization:** Architectural benefits of startup-paid ingestion overhead and online LRU caching.
*   **Visual Mappings:** Features both a fully rendered Mermaid.js flowchart and an explicitly readable text-based markdown flowchart.

### 2. [02_Machine_Learning_Models.md](02_Machine_Learning_Models.md)
**Feature Engineering, Tabular Trap Resolution, and Model Implementations**
*   **The Tabular Trap Resolution:** Technical detailing of how the repository unified all tabular and graph-based models on the same canonical 20K AMLSim archive population to ensure rigor.
*   **Feature Engineering Stack:** Extraction of structural graph features, 1-day, 7-day, and 30-day temporal rolling velocities, and stable bucketing for derived merchant nodes.
*   **Classical Baselines:** The Logistic Regression and Random Forest evaluation ladder across tabular, graph-analytic, and spectral embedding representations.
*   **Deep Learning Engine:** Deep dive into `SpatialTemporalHeteroGAT`, details on `HeteroData` nodes and relations, edge attribute conditioning (monetary amount and transaction time), and spatial-temporal fusion mechanisms.

### 3. [03_Training_Inference_and_XAI](03_Training_Inference_and_XAI.md)
**Optimization, serving behavior, and explainability implementation**
*   **Training Optimization:** Detailing the full-batch heterogeneous graph optimization loop, class-weighted cross-entropy loss to combat severe imbalance, and the patience-based validation early-stopping lookahead.
*   **Decision Threshold Sweep:** Explanation of the validation probability sweep to maximize classification F1 rather than raw loss.
*   **Inference Pipeline:** Explains why online serving uses `torch.no_grad()` on preloaded memory-resident state for millisecond-level results.
*   **Explainability (XAI):** Deep dive into Phase 9's graph-native explainability service wrapping PyTorch Geometric's `GNNExplainer` to reverse-engineer classifications into human-readable feature importances and incident transaction-level structural reports.

### 4. [04_MLOps_and_Deployment](04_MLOps_and_Deployment.md)
**Containerization, Drift Monitoring, and Telemetry Runbook**
*   **Containerization Strategy:** Technical brief on the CPU-optimized PyTorch base build, minimal Docker layers, and host-binded volume mapping for persistent state.
*   **Concept Drift Detection:** In-depth explanation of Phase 12's Kolmogorov-Smirnov (KS) two-sample comparison engine evaluating incoming feature distributions against preloaded archive baselines.
*   **Operational Logging:** Persistent JSONL event telemetry logging.
*   **Metrics Service:** Live Prometheus telemetry instrumentation.
*   **Interactive Runbook:** Step-by-step developer guide for starting, validating, calling, and stopping the containerized service.

### 5. [05_Real_Time_Fraud_Scenario.md](05_Real_Time_Fraud_Scenario.md)
**Visual Mule-Account Case Study and Model Decision Walkthrough**
*   **Real Case Narrative:** A repository-backed case study built around confirmed fraud account `19204`.
*   **Visual Network Graphs:** Mermaid diagrams showing the suspicious multi-account funding pattern, repeated counterparty reuse, and derived merchant-linked entities.
*   **Model Evidence:** Explanation of how the live Hetero GAT identifies the fraud through feature attributions, graph structure, and high-importance transfer edges.
*   **Analyst Interpretation:** Translation of model signals into a practical mule-account investigation story suitable for engineering, audit, or fraud-ops review.

---

## Unified System Flow Diagram

For a quick conceptual map, here is the complete end-to-end flow of data and request states in the system:

```text
       Raw Input Ingestion
               │
               ▼
┌───────────────────────────────┐
│     20K AMLSim Archive        │   Loaded globally once during engine startup
│ (20K_fanin200cycle200.tgz)    │
└──────────────┬────────────────┘
               │
               ▼  build_graph_feature_bundle()
┌───────────────────────────────┐
│   Unified Archive Feature     ├─► [Tabular Account Features] ──► (Drift Baseline)
│           Bundle              ├─► [Derived Merchant Features]
└──────────────┬────────────────┘─► [Graph Node / Spectral Embeddings]
               │
               ▼  build_pyg_graph_data_from_feature_bundle()
┌───────────────────────────────┐
│  PyG Heterogeneous Graph Data │   Nodes: account, merchant
│         (HeteroData)          │   Edges: transfers, buys_from, rev_buys_from
└──────────────┬────────────────┘
               │
               ▼  load_trained_hetero_gat_model()
┌───────────────────────────────┐
│          EngineState          ├── Predict: GET /predict/{account_id}
│   In-Memory Serving Singleton │── Explain: GET /explain/{account_id} (LRU-Cached)
│       (Running on CPU)        │── Monitor: POST /analyze-drift & GET /metrics
└───────────────────────────────┘
```

---

## Key Terminology and Legacy Mappings

To ensure maximum audit-readiness and clarity across the codebase, be aware of the following terminological matches within the repository:

| Operational Term | Codebase / Filename Anchor | Notes / Justification |
| :--- | :--- | :--- |
| **GNN Serving Model** | `pytorch_hetero_gat` | The actual serialized model type inside `pytorch_hetero_gat_model.pt`. |
| **GNN Legacy Scripting** | `pytorch_gcn` | Retained in paths such as `train_pytorch_gcn.py` and `pytorch_gcn_metrics.json` to preserve backward script compatibility. |
| **Model Checkpoint** | `pytorch_hetero_gat_model.pt` | The primary deep learning weights payload loaded at startup. |
| **Decision Threshold** | `optimal_threshold` | Dynamically swept during validation tuning to maximize metrics on highly imbalanced data. Defaults back to the configured decision threshold (e.g., `0.50`) if no sweep occurs. |
| **Feature Set** | `unified_archive_spatiotemporal_bundle` | The consolidated feature collection ensuring apples-to-apples baseline comparisons. |

---

## Architectural Principles

The documentation set describes a system built around four key engineering principles:

1.  **Representational Rigor (Apples-to-Apples):** Every baseline model—from simple Logistic Regression to the modern GNN—is trained and evaluated against feature representations extracted from the exact same 20K AMLSim node population. There are no mismatched data traps in this system.
2.  **Startup-Paid Latency Complexity:** The heavy industrial lifting of reading compressed tar archives, deriving rolling-velocity tables, resolving network topology, constructing heterogeneous PyG objects, and loading PyTorch weights is paid once during startup in `lifespan` initialization. Steady-state online serving is extremely streamlined.
3.  **Graph-Native Explainability (Model Introspection):** The explainability layer is not an external post-hoc tabular proxy. It uses PyG's model-native `Explainer` to perform relational introspection, providing auditors with both node feature attributions and actual incident edge importances.
4.  **Co-Located serving and monitoring:** Concept drift and Prometheus instrumentation are not isolated secondary components. They run in-process, reading from the same `EngineState` that serves inference, providing high-fidelity monitoring at negligible runtime overhead.
