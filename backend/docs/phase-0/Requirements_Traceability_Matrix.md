# Requirements Traceability Matrix

## Purpose

This matrix ensures that each business need from the PRD has an implementation target and a validation path.

## Traceability Matrix

| Requirement ID | Requirement Summary | Business Objective | Planned Implementation Area | Validation Method | Target Phase | Status |
| --- | --- | --- | --- | --- | --- | --- |
| FR-01 | Ingest IBM AMLSim transaction and entity datasets from CSV sources | BO-1, BO-5 | AMLSim ingestion pipelines and raw data loaders | ingestion test and schema validation against pinned AMLSim artifact | 3 | Planned |
| FR-02 | Validate required fields, types and nullability | BO-1, BO-5 | validation pipeline | data quality test suite | 3 | Planned |
| FR-03 | Support source-native party, account, transaction, alert and bank entities plus derived device, merchant and IP entities | BO-1, BO-3 | canonical entity model and deterministic enrichment jobs | schema review, lineage checks and entity relationship checks | 1, 3, 6 | Planned |
| FR-04 | Compute aggregate and velocity-based features | BO-1, BO-2 | feature engineering pipelines | feature unit tests and sample calculations | 4 | Planned |
| FR-05 | Build graph relationships from transactions and shared derived infrastructure | BO-3 | graph construction service | graph schema review, edge integrity checks and enrichment reproducibility checks | 6 | Planned |
| FR-06 | Train supervised and anomaly detection baselines | BO-1, BO-5 | model training pipeline | evaluation report and experiment tracking | 5 | Planned |
| FR-07 | Support graph-derived features or embeddings | BO-3, BO-5 | graph feature pipeline and embedding jobs | offline experiment comparison | 7 | Planned |
| FR-08 | Expose transaction and customer scoring APIs | BO-2, BO-5 | FastAPI inference service | API integration tests | 11 | Planned |
| FR-09 | Produce entity-level risk profiles with explanation and relationship context | BO-2, BO-4 | inference orchestration and risk profile service | API contract tests and dashboard walkthrough | 9, 11, 13 | Planned |
| FR-10 | Provide analyst visualizations for alerts, graph and risk exploration | BO-2, BO-4 | Streamlit dashboard | scenario-based UX validation | 13 | Planned |
| FR-11 | Track experiments, metrics and model versions | BO-5 | MLflow integration | training pipeline validation | 10 | Planned |
| FR-12 | Monitor drift and latency | BO-5 | monitoring and alerting stack | monitoring dashboard and synthetic drift test | 12 | Planned |
| FR-13 | Support reason-code style explainability | BO-4 | explainability service using SHAP and feature summaries | explanation review against sample predictions | 9 | Planned |
| FR-14 | Compare classical, graph-augmented and graph-ML systems | BO-1, BO-3, BO-5 | experiment design and model comparison report | benchmark review | 5, 7 | Planned |
| NFR-01 | Preserve modular system boundaries | BO-5 | layered architecture and service contracts | architecture review | 2, 11 | Planned |
| NFR-02 | Ensure reproducibility | BO-5 | versioned experiments, configs and pipelines | rerun reproducibility check | 10 | Planned |
| NFR-03 | Maintain demo-grade low-latency scoring | BO-2, BO-5 | optimized inference service | latency benchmark | 11, 12 | Planned |
| NFR-04 | Keep outputs understandable for investigators | BO-2, BO-4 | explanation UX and dashboard wording | analyst scenario review | 9, 13 | Planned |
| NFR-05 | Deploy with Docker and PostgreSQL-compatible persistence | BO-5 | containerized deployment architecture | container smoke tests | 11 | Planned |
| NFR-06 | Keep setup and demo flows portfolio-friendly | BO-5 | repository structure and documentation | fresh-start setup review | 10, 11, Final | Planned |

## Business Objective Legend

| ID | Objective |
| --- | --- |
| BO-1 | Increase early detection of fraudulent and suspicious behaviour |
| BO-2 | Improve analyst efficiency and prioritization |
| BO-3 | Detect coordinated network behaviour |
| BO-4 | Provide explainable and defensible risk scores |
| BO-5 | Demonstrate production-ready ML engineering practices |

## Usage Notes

- Each new implementation artifact should reference the relevant requirement IDs.
- Each report or test asset should show which requirements it validates.
- If scope changes, update this matrix before implementation diverges from the PRD.