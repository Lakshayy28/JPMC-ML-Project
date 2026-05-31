from __future__ import annotations

import networkx as nx
import pandas as pd


def build_weighted_digraph(
    transactions: pd.DataFrame,
    *,
    source_column: str,
    destination_column: str,
    amount_column: str,
    time_column: str,
    node_frame: pd.DataFrame | None = None,
    node_id_column: str | None = None,
    node_attributes: list[str] | None = None,
) -> nx.DiGraph:
    graph = nx.DiGraph()

    if node_frame is not None and node_id_column is not None:
        attributes = node_attributes or [column for column in node_frame.columns if column != node_id_column]
        for row in node_frame[[node_id_column, *attributes]].itertuples(index=False):
            node_id = getattr(row, node_id_column)
            attr_map = {attribute: getattr(row, attribute) for attribute in attributes}
            graph.add_node(node_id, **attr_map)

    grouped = transactions.groupby([source_column, destination_column], dropna=False).agg(
        edge_count=(amount_column, "size"),
        total_amount=(amount_column, "sum"),
        first_time=(time_column, "min"),
        last_time=(time_column, "max"),
    )

    for (source, destination), row in grouped.iterrows():
        if source == "" or destination == "":
            continue
        graph.add_edge(
            source,
            destination,
            edge_count=int(row["edge_count"]),
            total_amount=float(row["total_amount"]),
            first_time=int(row["first_time"]),
            last_time=int(row["last_time"]),
        )

    return graph


def build_archive_transaction_graph(nodes: pd.DataFrame, transactions: pd.DataFrame) -> nx.DiGraph:
    normalized_nodes = nodes.rename(
        columns={
            "nodeid": "node_id",
            "isFraud": "is_fraud",
            "init_balance": "initial_balance",
            "fraudStep": "fraud_step",
        }
    ).copy()
    normalized_nodes["is_fraud"] = normalized_nodes["is_fraud"].astype(int)

    normalized_transactions = transactions.rename(
        columns={
            "sourceNodeId": "source_node_id",
            "targetNodeId": "destination_node_id",
            "value": "amount",
            "time": "event_time",
        }
    )

    return build_weighted_digraph(
        normalized_transactions,
        source_column="source_node_id",
        destination_column="destination_node_id",
        amount_column="amount",
        time_column="event_time",
        node_frame=normalized_nodes,
        node_id_column="node_id",
        node_attributes=["is_fraud", "initial_balance", "fraud_step"],
    )


def build_canonical_account_graph(tables: dict[str, pd.DataFrame]) -> nx.DiGraph:
    accounts = tables["accounts"].copy()
    transactions = tables["transactions"].copy()
    return build_weighted_digraph(
        transactions,
        source_column="source_account_id",
        destination_column="destination_account_id",
        amount_column="amount",
        time_column="event_step",
        node_frame=accounts,
        node_id_column="account_id",
        node_attributes=["party_id", "is_alerted", "initial_balance", "bank_id"],
    )
