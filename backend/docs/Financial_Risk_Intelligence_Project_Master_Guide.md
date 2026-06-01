# Financial Risk Intelligence Project Master Guide

# Graph-Based Financial Transaction Risk Intelligence System

## Project Goal

Build a production-grade Financial Risk Intelligence Platform that detects fraud rings, mule accounts, synthetic identities and anomalous financial behaviour using:

- Classical ML
- Graph ML
- Temporal Modelling
- Explainability
- MLOps

## Why This Project?

This project demonstrates:

- Classical Machine Learning
- Graph Machine Learning
- Probabilistic Thinking
- Temporal Modelling
- Explainability
- Production Engineering
- MLOps
- Financial Risk Intelligence

---

# Phase 0 – Project Charter & PRD

## Deliverables

- Problem Statement
- Business Objectives
- Stakeholders
- Success Metrics
- Functional Requirements
- Non-Functional Requirements
- Investigation Workflow
- Alerting Workflow
- Risk Scoring Workflow

## Artifacts

- PRD
- Requirements Traceability Matrix
- Architecture Vision

---

# Phase 1 – Domain Understanding

## Study Topics

- Fraud Rings
- Mule Accounts
- Synthetic Identity Fraud
- Account Takeover
- Anti-Money Laundering

## Datasets

### IBM AMLSim
Primary dataset for MVP and canonical source schema for the first implementation slice.

### PaySim Fraud Dataset
Secondary benchmark dataset for later transaction-fraud comparison work.

### Elliptic Bitcoin Dataset
Graph-based illicit transaction dataset.

## Deliverables

- Fraud Taxonomy
- Dataset Assessment
- Entity Relationship Mapping

---

# Phase 2 – System Architecture

## Create

- High-Level Architecture Diagram
- Deployment Diagram
- Data Flow Diagram
- Sequence Diagram
- ER Diagram

## Layers

1. Data Ingestion
2. Feature Engineering
3. Graph Analytics
4. ML Training
5. Inference
6. Monitoring
7. Dashboard

---

# Phase 3 – Data Engineering

## Build Core Tables

- customers.csv
- accounts.csv
- devices.csv
- merchants.csv
- transactions.csv

## Tasks

- Data Cleaning
- Missing Value Handling
- Data Validation
- Class Imbalance Analysis
- EDA

## Deliverables

- Data Dictionary
- EDA Report
- Validation Pipeline

---

# Phase 4 – Feature Engineering

## Customer Features

- Transaction Count
- Average Amount
- Velocity
- Geographical Spread

## Merchant Features

- Risk Ratio
- Unique Customer Count

## Device Features

- Shared Account Count

## Temporal Features

- 1 Day Activity
- 7 Day Activity
- 30 Day Activity

## Deliverables

- Feature Store Design
- Feature Documentation

---

# Phase 5 – Classical Machine Learning

## Models

1. Logistic Regression
2. Random Forest
3. XGBoost
4. Isolation Forest

## Metrics

- Precision
- Recall
- F1
- ROC-AUC
- PR-AUC

## Deliverables

- Baseline Models
- Evaluation Report
- Feature Importance Report

---

# Phase 6 – Graph Construction

## Nodes

- Customers
- Accounts
- Devices
- Merchants
- IP Addresses

## Edges

- Transactions
- Shared Device
- Shared IP
- Shared Merchant

## Tools

- NetworkX
- Neo4j (Optional)

## Deliverables

- Graph Schema
- Fraud Cluster Discovery
- Graph Analytics Report

---

# Phase 7 – Graph ML

## Learn

- Node2Vec
- GraphSAGE
- GCN Fundamentals
- Graph Embeddings

## Goal

Compare:

- Classical ML Only
- Graph Features + Classical ML
- Graph Neural Networks

## Deliverables

- Embedding Pipeline
- Graph Feature Service
- Performance Comparison Report

---

# Phase 8 – Temporal Risk Intelligence

## Build

- Behaviour Drift Detection
- Velocity Monitoring
- Rolling Window Analytics

## Deliverables

- Temporal Feature Service
- Drift Detection Engine
- Behaviour Analytics Dashboard

---

# Phase 9 – Explainability

## Tools

- SHAP
- Permutation Importance

## Deliverables

- Explainability Service
- Risk Explanation Dashboard

---

# Phase 10 – MLOps

## Implement

- MLflow
- Experiment Tracking
- Model Registry
- Reproducible Pipelines

## Deliverables

- Training Pipeline
- Model Lifecycle Documentation

---

# Phase 11 – Deployment

## Backend

- FastAPI

## Infrastructure

- Docker
- PostgreSQL

## APIs

- POST /score_transaction
- POST /score_customer
- GET /risk_profile

## Deliverables

- Production APIs
- Deployment Guide

---

# Phase 12 – Monitoring

## Monitor

- Data Drift
- Feature Drift
- Prediction Drift
- Latency

## Tools

- EvidentlyAI

## Deliverables

- Monitoring Dashboard
- Alerting Framework

---

# Phase 13 – Dashboard

## Build

- Fraud Investigation Dashboard
- Graph Explorer
- Risk Explorer

## Tools

- Streamlit
- Plotly

## Deliverables

- Analyst UI
- Executive Dashboard

---

# Final Portfolio Package

1. GitHub Repository
2. PRD
3. Architecture Document
4. Dataset Documentation
5. ML Evaluation Report
6. Graph ML Report
7. Deployment Guide
8. Demo Video
9. Resume Project Entry
10. Interview Story Bank

---

# Final Resume Positioning

Designed and deployed a graph-based financial risk intelligence platform combining:

- Gradient Boosted Models
- Graph Embeddings
- Anomaly Detection
- Temporal Behaviour Modelling
- Explainability
- Real-Time APIs
- Monitoring and Drift Detection

This project is intended to showcase production-grade Applied AI/ML engineering capability for banking and fintech environments.
