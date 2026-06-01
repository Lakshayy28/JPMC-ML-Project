# Feature Documentation

## Purpose

Document the current baseline feature sets used for transaction, party and graph-node modelling.

## Transaction Features

| Feature | Meaning |
| --- | --- |
| `amount` | raw transaction amount |
| `event_step` | AMLSim event step |
| `is_cash` | whether the row came from cash transaction ingestion |
| `outgoing_transaction_count` | number of outgoing transactions from the source account |
| `outgoing_amount_total` | total outgoing transaction amount for the source account |
| `unique_destinations` | distinct destination accounts reached by the source account |
| `incoming_transaction_count` | number of transactions into the destination account |
| `incoming_amount_total` | total incoming amount into the destination account |
| `unique_sources` | distinct source accounts sending into the destination account |
| `source_initial_balance` | initial balance of the source account |
| `source_is_alerted` | whether the source account is linked to an alert |
| `source_shared_device_count` | number of accounts sharing the source account's derived device |
| `source_shared_ip_count` | number of accounts sharing the source account's derived IP |
| `destination_initial_balance` | initial balance of the destination account |
| `destination_is_alerted` | whether the destination account is linked to an alert |
| `destination_shared_device_count` | number of accounts sharing the destination account's derived device |
| `destination_shared_ip_count` | number of accounts sharing the destination account's derived IP |

### Transaction Label

- target column: `label`
- current meaning: alert-related or alert-adjacent transaction
- lineage column: `label_source`

## Party Features

| Feature | Meaning |
| --- | --- |
| `account_count` | number of accounts owned by the party |
| `alert_count` | number of alerts linked to the party |
| `outgoing_transaction_count` | total outgoing transactions from party-owned accounts |
| `outgoing_amount_total` | total outgoing amount from party-owned accounts |
| `unique_destination_accounts` | distinct destination accounts reached by the party |
| `incoming_transaction_count` | total incoming transactions into party-owned accounts |
| `incoming_amount_total` | total incoming amount into party-owned accounts |
| `unique_source_accounts` | distinct source accounts sending to the party |
| `device_count` | number of distinct derived devices linked to the party |
| `ip_count` | number of distinct derived IPs linked to the party |

### Party Label

- target column: `label`
- current meaning: party linked to at least one alert

## Graph Node Features

| Feature | Meaning |
| --- | --- |
| `in_degree` | count of incoming graph edges |
| `out_degree` | count of outgoing graph edges |
| `weighted_in_degree` | total incoming transaction weight |
| `weighted_out_degree` | total outgoing transaction weight |
| `pagerank` | PageRank score over the directed transaction graph |
| `clustering_coefficient` | local clustering score over the undirected projection |
| `weak_component_id` | weakly connected component identifier |
| `weak_component_size` | size of the node's weakly connected component |
| `community_id` | detected community identifier when community detection is enabled |
| `community_size` | size of the detected community |
| `initial_balance` | source node initial balance |
| `fraud_step` | fraud activation step if present in AMLSim archive nodes |

## Leakage Notes

1. graph-node training drops `fraud_step` and the label column before fitting the baseline model
2. transaction labels are currently proxy labels in sample mode and should not be treated as direct ground truth
3. Device/IP/Merchant features are derived and deterministic, which is useful for structure but not equivalent to production telemetry