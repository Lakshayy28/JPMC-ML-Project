from __future__ import annotations

import networkx as nx
import pandas as pd


def compute_graph_summary(graph: nx.DiGraph) -> dict[str, float | int]:
    if graph.number_of_nodes() == 0:
        return {
            "node_count": 0,
            "edge_count": 0,
            "density": 0.0,
            "average_in_degree": 0.0,
            "average_out_degree": 0.0,
            "weak_component_count": 0,
            "largest_weak_component_size": 0,
            "average_clustering": 0.0,
        }

    undirected = graph.to_undirected()
    weak_components = list(nx.weakly_connected_components(graph))
    largest_component_size = max(len(component) for component in weak_components)
    node_count = graph.number_of_nodes()

    return {
        "node_count": int(node_count),
        "edge_count": int(graph.number_of_edges()),
        "density": float(nx.density(graph)),
        "average_in_degree": float(sum(dict(graph.in_degree()).values()) / node_count),
        "average_out_degree": float(sum(dict(graph.out_degree()).values()) / node_count),
        "weak_component_count": int(len(weak_components)),
        "largest_weak_component_size": int(largest_component_size),
        "average_clustering": float(nx.average_clustering(undirected)),
    }


def _community_assignments(graph: nx.Graph, seed: int = 42) -> tuple[dict[object, int], dict[int, int]]:
    try:
        communities = nx.community.louvain_communities(graph, seed=seed, weight="edge_count")
    except Exception:
        communities = list(nx.community.greedy_modularity_communities(graph, weight="edge_count"))

    node_to_community: dict[object, int] = {}
    community_sizes: dict[int, int] = {}
    for community_id, nodes in enumerate(communities):
        community_sizes[community_id] = len(nodes)
        for node in nodes:
            node_to_community[node] = community_id
    return node_to_community, community_sizes


def compute_node_features(graph: nx.DiGraph, *, include_communities: bool = True, community_seed: int = 42) -> pd.DataFrame:
    if graph.number_of_nodes() == 0:
        return pd.DataFrame()

    pagerank = nx.pagerank(graph, weight="total_amount")
    weak_component_map: dict[object, int] = {}
    weak_component_size_map: dict[object, int] = {}
    for component_id, nodes in enumerate(nx.weakly_connected_components(graph)):
        component_size = len(nodes)
        for node in nodes:
            weak_component_map[node] = component_id
            weak_component_size_map[node] = component_size

    undirected = graph.to_undirected()
    clustering = nx.clustering(undirected)
    community_map: dict[object, int] = {}
    community_sizes: dict[int, int] = {}
    if include_communities:
        community_map, community_sizes = _community_assignments(undirected, seed=community_seed)

    records: list[dict[str, object]] = []
    for node_id, attrs in graph.nodes(data=True):
        records.append(
            {
                "node_id": node_id,
                "in_degree": int(graph.in_degree(node_id)),
                "out_degree": int(graph.out_degree(node_id)),
                "weighted_in_degree": float(graph.in_degree(node_id, weight="total_amount")),
                "weighted_out_degree": float(graph.out_degree(node_id, weight="total_amount")),
                "pagerank": float(pagerank.get(node_id, 0.0)),
                "clustering_coefficient": float(clustering.get(node_id, 0.0)),
                "weak_component_id": int(weak_component_map.get(node_id, -1)),
                "weak_component_size": int(weak_component_size_map.get(node_id, 0)),
                "community_id": int(community_map.get(node_id, -1)) if include_communities else -1,
                "community_size": int(community_sizes.get(community_map.get(node_id, -1), 0)) if include_communities else 0,
                **attrs,
            }
        )
    return pd.DataFrame.from_records(records)


def top_nodes_by_metric(feature_frame: pd.DataFrame, metric: str, top_n: int = 10) -> list[dict[str, object]]:
    if feature_frame.empty or metric not in feature_frame.columns:
        return []
    return feature_frame.sort_values(metric, ascending=False).head(top_n).to_dict("records")
