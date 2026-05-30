# Product Requirements Document

## Project Name

Graph-Based Financial Transaction Risk Intelligence System

## Document Purpose

Define the product scope, operational goals, user workflows and measurable success criteria for a financial risk intelligence platform focused on fraud detection, network risk discovery and analyst decision support.

## Problem Statement

Financial institutions need to detect suspicious transactions and suspicious entities, not just isolated fraudulent payments. Traditional transaction-level rules miss coordinated fraud rings, mule account networks, synthetic identities and evolving behavioural anomalies. The platform must detect both point anomalies and connected risk patterns while producing outputs that investigators can trust and act on.

## Product Vision

Build a production-grade risk intelligence platform that combines transaction analytics, graph analytics, machine learning, temporal monitoring and explainability to score financial behaviour and accelerate fraud and AML investigations.

## Business Objectives

1. Increase early detection of fraudulent and suspicious financial behaviour.
2. Improve analyst efficiency by prioritizing high-risk entities and transactions.
3. Detect coordinated network behaviour such as fraud rings and mule structures.
4. Provide explainable risk scores suitable for banking and fintech review workflows.
5. Demonstrate production-ready ML engineering practices including monitoring and reproducibility.

## Primary Users

### Fraud Analyst

Needs prioritized alerts, linked-entity investigation views, and plain-language explanations for model outputs.

### AML Investigator

Needs customer and account-level behaviour summaries, graph context and anomalous flow signals.

### ML Engineer

Needs reproducible pipelines, feature definitions, model comparisons and drift tracking.

### Platform Engineer

Needs clear service boundaries, deployment standards and observability requirements.

### Hiring Reviewer Or Interviewer

Needs a coherent portfolio story showing business relevance and production engineering depth.

## In Scope

- ingest and normalize fraud-oriented financial datasets
- construct entity and transaction views across customers, accounts, devices, merchants and IPs
- generate classical ML, graph-derived and temporal features
- train baseline supervised and unsupervised models
- construct graph structures for cluster discovery and graph ML experiments
- expose transaction and customer risk scoring APIs
- provide model explanations and analyst-facing dashboards
- monitor drift, latency and operational health

## Out Of Scope

- direct integration with live bank core systems
- case management workflow automation
- sanctions screening
- production-grade identity resolution with external KYC vendors
- full regulatory reporting workflow submission

## Assumptions

- IBM AMLSim is the canonical MVP dataset for ingestion, schema design and first baseline training.
- The source-native AMLSim backbone is account and transaction centric; Device, IP and Merchant entities will be introduced through deterministic synthetic enrichment in the MVP processed layer.
- The initial deployment target is a portfolio-grade local or cloud-hosted environment, not a regulated bank production environment.
- Risk scores are decision-support signals and do not replace investigator judgment.
- Entity identifiers across datasets may require synthetic harmonization during data engineering.

## Success Metrics

### Model Metrics

- transaction-level precision at top-k alerts
- recall on known fraud labels where labels exist
- PR-AUC for imbalanced classification tasks
- lift over random ranking for investigation prioritization

### Graph And Temporal Metrics

- number of suspicious clusters discovered and validated
- percentage of fraud cases connected to shared device, IP or merchant relationships
- detection coverage of bursty or drifting behaviours over rolling windows

### Product Metrics

- median scoring latency for online inference
- time to generate investigator-ready explanation output
- analyst workflow completeness for transaction, entity and graph review

### Engineering Metrics

- reproducible training runs tracked in MLflow
- data validation pass rate on ingestion
- model and feature drift detection coverage in monitoring

## User Workflows

### Investigation Workflow

1. Analyst receives a prioritized alert for a transaction or entity.
2. Analyst opens the risk profile and explanation summary.
3. Analyst reviews linked entities, graph neighbors and temporal behaviour anomalies.
4. Analyst determines whether the alert indicates isolated fraud, network abuse or normal behaviour.
5. Analyst records decision rationale using explanation outputs and graph evidence.

### Alerting Workflow

1. Inference service scores incoming transactions or customers.
2. Thresholding and ranking logic creates alerts.
3. Monitoring services track alert volume, latency and drift signals.
4. Dashboard surfaces high-risk alerts and clusters to analysts.

### Risk Scoring Workflow

1. Raw data is validated and transformed.
2. Feature pipelines compute classical, graph and temporal features.
3. Models produce raw probabilities, anomaly scores or embeddings.
4. Risk orchestration combines signals into normalized risk outputs.
5. Explainability service generates reason codes and supporting evidence.

## Functional Requirements

| ID | Requirement |
| --- | --- |
| FR-01 | The system must ingest transaction and entity data from CSV-based source datasets. |
| FR-02 | The system must validate required fields, types and nullability before feature generation. |
| FR-03 | The system must support customer, account, device, merchant and transaction entities. |
| FR-04 | The system must compute classical aggregate and velocity-based features over rolling windows. |
| FR-05 | The system must build graph relationships from shared devices, IPs, merchants and transactions. |
| FR-06 | The system must train at least one supervised classifier and one anomaly detection model. |
| FR-07 | The system must support graph-derived features or embeddings in downstream scoring. |
| FR-08 | The system must expose transaction scoring and customer scoring APIs. |
| FR-09 | The system must produce entity-level risk profiles containing score, explanation and relationship context. |
| FR-10 | The system must provide analyst-facing visualizations for alerts, graphs and risk exploration. |
| FR-11 | The system must track experiments, metrics and model versions. |
| FR-12 | The system must monitor data drift, feature drift, prediction drift and latency. |
| FR-13 | The system must support reason-code style explainability for model outputs. |
| FR-14 | The system must compare classical-only, graph-augmented and graph-ML approaches. |

## Non-Functional Requirements

| ID | Requirement |
| --- | --- |
| NFR-01 | The system should be modular so ingestion, feature engineering, training, inference and monitoring can evolve independently. |
| NFR-02 | The system should be reproducible across runs, with versioned data assumptions, experiments and model artifacts. |
| NFR-03 | The online scoring path should be designed for low-latency responses appropriate for demo-scale real-time APIs. |
| NFR-04 | The platform should produce explanations understandable by non-ML investigators. |
| NFR-05 | The architecture should be deployable via Docker and use PostgreSQL-compatible persistence. |
| NFR-06 | The project should remain portfolio-friendly, meaning setup and demo flows are understandable by reviewers. |

## Risks And Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Simulated datasets may not reflect real fraud operations perfectly | Medium | Document dataset limitations and frame results as proof of engineering capability |
| Label quality may be weak or incomplete | High | Use both supervised and unsupervised approaches and highlight evaluation limits |
| Graph construction may create noisy connections | High | Define strict edge semantics and validate clusters qualitatively |
| Feature leakage may inflate metrics | High | Enforce time-aware splits and feature documentation |
| Scope creep across ML, graph, MLOps and UI work | High | Use phased delivery with explicit acceptance criteria |

## Milestone Acceptance Criteria

### Phase 0

- PRD, RTM and architecture vision approved for implementation use

### Phase 1 To 4

- datasets assessed, entities mapped, and feature definitions documented

### Phase 5 To 9

- baseline and advanced models evaluated with explainability artifacts available

### Phase 10 To 13

- training, inference, monitoring and dashboard layers integrated into a demonstrable platform

## Open Questions

1. Will graph storage remain in-memory and offline for MVP, or will Neo4j be included in the serving architecture?
2. What level of authentication is required for the public-facing demo APIs?