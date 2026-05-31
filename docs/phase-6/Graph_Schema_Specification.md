# Graph Schema Specification

## Purpose

Define the graph schema currently implemented and the graph schema planned for the main platform graph.

## Current Implemented Graphs

### Archive Transaction Graph

Used for graph analytics and graph-node baseline training on bundled AMLSim archive samples.

| Element | Definition |
| --- | --- |
| Node | AMLSim archive node from `nodes.csv` |
| Node ID | `nodeid` |
| Node Attributes | `is_fraud`, `initial_balance`, `fraud_step` |
| Edge | directed money transfer from `transactions.csv` |
| Edge Attributes | `edge_count`, `total_amount`, `first_time`, `last_time` |

### Canonical Account Graph

Implemented in code and available for later reporting from the processed tables.

| Element | Definition |
| --- | --- |
| Node | canonical account |
| Node ID | `account_id` |
| Node Attributes | `party_id`, `is_alerted`, `initial_balance`, `bank_id` |
| Edge | directed account-to-account transfer |
| Edge Attributes | `edge_count`, `total_amount`, `first_time`, `last_time` |

## Planned Multi-Entity Platform Graph

The full platform graph should include:

| Node Type | Source |
| --- | --- |
| Party | source-native or sample fallback canonical table |
| Account | canonical account table |
| Device | derived infrastructure table |
| IP Address | derived infrastructure table |
| Merchant | derived infrastructure table |
| Bank | canonical bank table |

| Edge Type | Meaning |
| --- | --- |
| `account_transfers_to_account` | transaction movement |
| `party_owns_account` | ownership relationship |
| `bank_hosts_account` | institution boundary |
| `account_uses_device` | derived infrastructure linkage |
| `account_uses_ip` | derived infrastructure linkage |
| `transaction_links_to_merchant` | derived merchant linkage |
| `alert_involves_account` | alert membership |

## Graph Feature Semantics

Current graph features computed in code:

- in-degree
- out-degree
- weighted in-degree
- weighted out-degree
- PageRank
- clustering coefficient
- weak component membership
- community membership when enabled

## Design Rules

1. source-native entities must remain distinguishable from derived entities
2. transaction edges are always directed
3. edge aggregation preserves count and total amount
4. graph features should be recomputable from canonical tables without hidden state