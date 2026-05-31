from __future__ import annotations

import networkx as nx
import pandas as pd
from sklearn.decomposition import TruncatedSVD


def compute_spectral_node_embeddings(
    graph: nx.DiGraph,
    *,
    dimensions: int = 16,
    weight: str = "edge_count",
    random_state: int = 42,
) -> pd.DataFrame:
    if graph.number_of_nodes() == 0:
        return pd.DataFrame()

    ordered_nodes = list(graph.nodes())
    component_limit = max(1, min(dimensions, len(ordered_nodes) - 1))
    adjacency = nx.to_scipy_sparse_array(graph, nodelist=ordered_nodes, weight=weight, dtype=float)
    reducer = TruncatedSVD(n_components=component_limit, random_state=random_state)
    reduced = reducer.fit_transform(adjacency)

    embeddings = pd.DataFrame(reduced, columns=[f"embedding_{index}" for index in range(component_limit)])
    embeddings.insert(0, "node_id", ordered_nodes)
    return embeddings
