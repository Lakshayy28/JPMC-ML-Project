from __future__ import annotations

import networkx as nx
import pandas as pd

from fri.graph.analytics import compute_node_features
from fri.graph.embeddings import compute_spectral_node_embeddings


def build_graph_feature_bundle(
    graph: nx.DiGraph,
    *,
    include_communities: bool = True,
    community_seed: int = 42,
    embedding_dimensions: int = 16,
    embedding_random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    node_features = compute_node_features(
        graph,
        include_communities=include_communities,
        community_seed=community_seed,
    )
    embeddings = compute_spectral_node_embeddings(
        graph,
        dimensions=embedding_dimensions,
        random_state=embedding_random_state,
    )
    combined = node_features.merge(embeddings, on="node_id", how="left") if not node_features.empty else embeddings
    return {
        "node_features": node_features,
        "embeddings": embeddings,
        "combined": combined,
    }
