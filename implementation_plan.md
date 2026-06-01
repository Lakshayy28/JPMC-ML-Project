# Advanced Model Strategies: Implementation Plan

## Current Baseline (SpatialTemporalHeteroGAT)

| Metric | Value |
|--------|-------|
| Precision | 0.7319 |
| Recall | 0.6962 |
| F1 | 0.7136 |
| PR-AUC | 0.7417 |
| ROC-AUC | 0.9186 |

---

## Applied Scientist Assessment

### ✅ Strategy 3: PNA Aggregation — **IMPLEMENT (Highest Impact)**

**Why it works for this dataset:** The AMLSim graph has highly heterogeneous degree distributions — some accounts have 1 neighbor, others have 100+. GAT's sum-based attention treats all neighborhoods identically regardless of degree. PNA's degree-scaled multi-aggregation (`mean`, `max`, `min`, `std`) naturally captures:

- **Smurfing detection** via `std` (high variance = many small deposits from different sources)
- **Hub identification** via degree-scaling (payroll accounts vs. laundering hubs)
- **Fan-in/fan-out asymmetry** via `min`/`max` (legitimate accounts have balanced flow; laundering nodes don't)

**Implementation:**
- Replace `GATConv` with `PNAConv` in each `HeteroConv` layer
- Compute per-edge-type degree histograms for PNA scalers
- Use aggregators: `['mean', 'min', 'max', 'std']`, scalers: `['identity', 'amplification', 'attenuation']`

---

### ✅ Strategy 2: GraphMAE Pretraining — **IMPLEMENT (Moderate Impact)**

**Why it works:** The 91/9 class split means the model sees ~10x more benign gradient updates. GraphMAE forces the encoder to learn what "normal" financial routing looks like *before* seeing any labels. The fine-tuned model then has a much sharper decision boundary.

**Implementation (simplified but effective):**
- **Phase 1 (Pre-training):** Mask 30% of node features randomly. Train the encoder + a lightweight decoder MLP to reconstruct the masked features using cosine similarity loss. No labels used.
- **Phase 2 (Fine-tuning):** Freeze the encoder's first conv layer. Train the classifier on fraud labels with the same CrossEntropyLoss.

---

### ⚠️ Strategy 1: TGN — **SKIP (Replaced with Focal Loss)**

> [!IMPORTANT]
> **Why TGN is NOT practical for this dataset:**
> 1. AMLSim uses **discrete integer time steps** (0–99), not continuous timestamps. TGN's temporal memory module (GRU/LSTM) needs fine-grained continuous time to be effective.
> 2. TGN requires a **fundamentally different data format** (chronological edge event stream) — it cannot use the static HeteroData graph at all.
> 3. The temporal signal is already captured by our **6 velocity features per window × 3 windows = 18 temporal features** — this is already a strong temporal representation.
> 4. TGN adds massive complexity (memory modules, temporal sampling) for marginal gain on discrete-time data.

**My Alternative: Focal Loss** — A simpler, more targeted fix for the same underlying problem (class imbalance → poor Recall). Focal Loss down-weights easy-to-classify benign nodes and focuses gradient updates on the hard fraud cases near the decision boundary. This directly addresses the Recall gap (0.70 vs perfect 1.0) without any architectural changes.

---

## Proposed Changes

### [NEW] Advanced training script

#### [NEW] [audit_advanced_models.py](file:///Users/lakshaychandra/JPMC%20ML%20Project/backend/scripts/audit_advanced_models.py)

Standalone script that trains 4 model variants and compares:
1. **Baseline GAT** (reference — already trained)
2. **PNA variant** — `SpatialTemporalHeteroPNA` with multi-aggregation
3. **GAT + GraphMAE** — Pretrain with masked feature reconstruction, then fine-tune
4. **GAT + Focal Loss** — Replace CrossEntropyLoss with Focal Loss

All variants use the same data pipeline, splits, normalization, and evaluation logic from the backend.

---

### Notebook update

#### [MODIFY] [FRI_Applied_Scientist_Audit.ipynb](file:///Users/lakshaychandra/JPMC%20ML%20Project/FRI_Applied_Scientist_Audit.ipynb)

Add new cells after the baseline evaluation with:
- PNA model definition + training
- GraphMAE pretrain + fine-tune cycle
- Focal Loss variant
- Comparative results table

---

## Verification Plan

### Automated Tests
- Run `audit_advanced_models.py` locally to verify all 3 experimental models train without errors
- Compare all metrics against the GAT baseline
- Verify that at least one variant beats the baseline F1 of 0.7136

### Expected Outcome
| Model | Expected F1 Range | Rationale |
|-------|-------------------|-----------|
| Baseline GAT | 0.714 (reference) | — |
| PNA | 0.72–0.76 | Multi-aggregation captures variance signals |
| GAT + GraphMAE | 0.72–0.75 | Better representations from self-supervised pretraining |
| GAT + Focal Loss | 0.71–0.74 | Better gradient focus on hard examples |
