# Fraud Taxonomy

## Purpose

Define the fraud and AML pattern taxonomy for the Financial Risk Intelligence Platform using IBM AMLSim as the canonical MVP dataset. This document separates patterns that AMLSim supports natively from patterns that must be approximated through derived features or synthetic enrichment.

## Dataset Grounding

This taxonomy is grounded in the pinned AMLSim repository artifact cloned into the workspace:

- repository: `https://github.com/IBM/AMLSim`
- pinned commit: `7338a4bcb1af9bcfea2201ad7daccfe2a4d569ca`
- practical branch guidance from AMLSim README: use `master`
- observed sample outputs: `sample/outputs/accounts.csv`, `sample/outputs/alerts.csv`, `sample/outputs/cash_tx.csv`, `sample/outputs/tx.csv`
- observed parameterized typologies: `paramFiles/1K/alertPatterns.csv` includes `fan_in`, `fan_out`, and `cycle`

## Taxonomy Design Principles

1. Prefer patterns that can be expressed as entity relationships and temporal behavior, not only as isolated transaction rules.
2. Keep taxonomy aligned with what AMLSim actually emits in the first implementation slice.
3. Mark Device, IP Address and Merchant signals as derived MVP entities, not source-native AMLSim evidence.
4. Distinguish native label support from inferred risk proxies so model evaluation remains honest.

## Top-Level Taxonomy

| Family | Core Subtypes | AMLSim Coverage | MVP Handling |
| --- | --- | --- | --- |
| Fraud Rings | fan-in, fan-out, cycle, scatter-gather style relays | Strong for fan-in, fan-out and cycle; indirect for more complex relay patterns | Native graph and alert-based detection |
| Mule Accounts | funnel accounts, pass-through accounts, payout mules, mule clusters | Moderate; represented through alert membership and rapid movement patterns | Native account and transaction signals plus derived infrastructure reuse |
| Synthetic Identity Fraud | slow-burn synthetics, bust-out, linked synthetic cohorts | Weak natively | Approximate through party lifecycle and derived shared infrastructure patterns |
| Account Takeover | device deviation, geo-impossible access, session-based compromise, cash-out bursts | Weak natively | Requires derived Device and IP entities plus temporal anomalies |
| AML Patterns | structuring, layering, round-tripping, rapid funds movement, false-positive SAR lookalikes | Strong for layering-style network patterns; moderate for structuring | Native transactions and alert typologies plus engineered thresholds |

## Fraud Rings

### Definition

A fraud ring is a coordinated network of accounts or parties moving value according to a repeatable topology rather than independent customer behavior.

### Subtypes

| Subtype | Description | AMLSim Evidence |
| --- | --- | --- |
| Fan-In Ring | many source accounts send to a central collector | directly modeled in `paramFiles/1K/alertPatterns.csv` |
| Fan-Out Ring | central source distributes to many beneficiary accounts | directly modeled in `paramFiles/1K/alertPatterns.csv` |
| Cycle Ring | value rotates across a loop of accounts | directly modeled in `paramFiles/1K/alertPatterns.csv` and sample alerts |
| Scatter-Gather Relay | staged distribution then regrouping | not explicit in observed sample outputs, but approximable from multi-hop transaction paths |
| Nested Ring | overlapping or cascading rings | not native in sample outputs; discovered analytically from shared membership and communities |

### Detection Signals

- high fan-in or fan-out at the account level
- repeated counterparties across alerted accounts
- strongly connected components with short cycle length
- temporal burstiness across multiple linked accounts
- shared derived Device or IP reuse across ring participants

## Mule Accounts

### Definition

A mule account primarily exists to receive, relay or cash out funds on behalf of another actor while showing limited value retention.

### Subtypes

| Subtype | Description | AMLSim Coverage |
| --- | --- | --- |
| Funnel Account | aggregates inbound payments from many accounts | supported by fan-in patterns |
| Pass-Through Mule | rapid in-and-out movement with low balance retention | approximated through transaction velocity and balance turnover |
| Payout Mule | receives from a coordinator and disperses to many endpoints | supported by fan-out patterns |
| Mule Cluster | multiple mule accounts linked to a ring | supported indirectly through alert membership and graph communities |

### Detection Signals

- outgoing amount closely tracks incoming amount over short windows
- large number of unique counterparties relative to account age
- high cash-out count or repeated withdrawal behavior
- derived infrastructure reuse among accounts that should be independent
- alert adjacency to known ring accounts

## Synthetic Identity Fraud

### Definition

Synthetic identity fraud uses fabricated or mixed identity attributes to create accounts that behave legitimately long enough to support later abuse.

### Subtypes

| Subtype | Description | AMLSim Coverage |
| --- | --- | --- |
| Slow-Burn Synthetic | low-activity buildup before later escalation | weak native support |
| Bust-Out Synthetic | sudden large-value activity after quiet period | weak native support |
| Linked Synthetic Cohort | multiple identities sharing hidden infrastructure | requires derived Device, IP and Merchant entities |

### Detection Signals

- short account age followed by sharp increase in transfer amount or count
- sparse relationship history before abrupt network growth
- repeated derived Device or IP linkage across seemingly unrelated parties
- party or account naming patterns that cluster unnaturally after enrichment

## Account Takeover

### Definition

Account takeover is compromise of an otherwise legitimate account followed by unauthorized behavior and often rapid cash-out.

### Subtypes

| Subtype | Description | AMLSim Coverage |
| --- | --- | --- |
| Device Deviation | account begins operating from unseen device clusters | not source-native |
| IP Deviation | account begins operating from unusual or shared network sources | not source-native |
| Rapid Cash-Out | immediate burst of outgoing activity after compromise | partially approximable from temporal transaction behavior |
| Coordinated Takeover Ring | multiple hijacked accounts used in one operation | requires enrichment and graph analytics |

### Detection Signals

- sudden change in derived Device or IP assignments relative to party baseline
- off-pattern timing and velocity spikes
- rapid increase in outgoing cash transactions
- shared derived infrastructure across multiple high-risk accounts

## AML Patterns

### Definition

AML patterns describe suspicious movement of funds intended to obscure provenance, distribute value across networks or reintroduce funds after layering.

### Subtypes

| Subtype | Description | AMLSim Coverage |
| --- | --- | --- |
| Structuring | repeated sub-threshold or regularized amounts | approximated through amount and periodicity features |
| Layering | multi-hop movement across accounts | strong through graph topology and alert patterns |
| Round-Tripping | money exits and returns through a loop | strong where cycle patterns exist |
| Rapid Funds Movement | limited dwell time between receipt and transfer | strong through step-level transaction timing |
| False-Positive Lookalikes | payroll, treasury or legitimate treasury-like flows | not labeled explicitly; must be handled in review and evaluation |

## Detection Signal Reference

| Signal Class | Examples | Most Useful For |
| --- | --- | --- |
| Entity Signals | account age, initial balance, cash transaction count, alert membership, business type | mules, synthetics, structuring |
| Relationship Signals | fan-in, fan-out, unique counterparties, reciprocal flows, shared Device/IP/Merchant links | fraud rings, mules, takeover clusters |
| Temporal Signals | burstiness, dwell time, rolling 1/7/30-step counts, sudden change after quiet period | takeover, mules, bust-out patterns |
| Hybrid Signals | alert-adjacent infrastructure reuse, network growth after inactivity, cash-out after receipt | rings, AML patterns, takeover |

## AMLSim Support Matrix

| Pattern Area | Native In AMLSim | Derived In MVP | Out Of Scope For First Slice |
| --- | --- | --- | --- |
| Account-to-account transaction graph | Yes | No | No |
| Alert membership and typology labels | Yes | No | No |
| Party exports and account mapping | Yes in converted schema | No | No |
| Device intelligence | No | Yes | No |
| IP intelligence | No | Yes | No |
| Merchant intelligence | No | Yes | No |
| Session telemetry | No | No | Yes |
| External sanctions or KYC enrichment | No | No | Yes |

## False Positive And False Negative Caveats

### False Positive Risks

- legitimate hub-and-spoke payment behavior can resemble fan-out fraud rings
- treasury or payroll-like activity can resemble layering or payout mule behavior
- synthetic Device or IP reuse can overstate infrastructure linkage if enrichment rules are too coarse

### False Negative Risks

- AMLSim sample outputs do not include real device/session traces, so stealthy takeover behavior may be missed
- synthetic identity fraud may look normal unless lifecycle and derived infrastructure features are added
- transaction-level labels are weaker in the bundled sample outputs than in full converted alert transaction exports

## Implementation Implications

1. The first baseline should prioritize ring, mule and AML-pattern detection because they align best with AMLSim-native evidence.
2. Synthetic identity and takeover scenarios must be documented as enrichment-dependent use cases rather than fully source-backed use cases.
3. Phase 4 feature engineering must include rolling-window velocity, counterparty diversity and infrastructure reuse features.
4. Phase 6 graph construction must preserve both transaction edges and derived infrastructure edges so the taxonomy remains represented in the graph schema.

## Acceptance Criteria

- each fraud family maps to concrete entities, relationships and temporal signals
- AMLSim-native support versus derived support is explicit for every major fraud family
- the taxonomy can be used directly to define features, graph edges and model labels for the first implementation slice