"""
Graph export module formatting NetworkX propagation trees into JSON-serializable payloads
optimized for interactive visualization in React.
"""

from typing import Any, Dict, List, Optional
import networkx as nx

GRAPH_DISCLAIMER_TEXT = (
    "This visualization represents an observed rumor propagation tree reconstructed from recorded "
    "retweet/reply timestamps and parent-child linkages in the Twitter15/Twitter16 benchmark. "
    "It is NOT a complete follower/following social friendship network, nor does it capture all unrecorded passive impressions."
)


def export_graph_for_visualization(
    G: nx.DiGraph,
    max_nodes: int = 150,
) -> Dict[str, Any]:
    """
    Serializes a NetworkX DiGraph into nodes and links format for web rendering.
    Safely limits rendering payload to max_nodes if the cascade is huge, prioritizing
    the root node and high-degree / early-delay propagation paths.
    """
    total_nodes = G.number_of_nodes()
    total_edges = G.number_of_edges()

    if total_nodes == 0:
        return {
            "post_id": G.graph.get("post_id", ""),
            "dataset": G.graph.get("dataset", ""),
            "label": G.graph.get("label", "unknown"),
            "source_text": G.graph.get("source_text", ""),
            "total_nodes": 0,
            "total_edges": 0,
            "displayed_nodes": 0,
            "displayed_edges": 0,
            "is_truncated": False,
            "nodes": [],
            "edges": [],
            "disclaimer": GRAPH_DISCLAIMER_TEXT,
        }

    root_uid = G.graph.get("root_uid")
    if not root_uid or not G.has_node(root_uid):
        root_uid = min(G.nodes, key=lambda n: G.nodes[n].get("delay", 0.0))

    is_truncated = False
    chosen_nodes = set(G.nodes())

    if total_nodes > max_nodes:
        is_truncated = True
        # Include root
        subset = {root_uid}
        
        # Sort remaining nodes by priority: smaller depth first, then higher degree, then smaller delay
        scored_nodes = []
        for n in G.nodes():
            if n == root_uid:
                continue
            depth = G.nodes[n].get("depth", 999)
            deg = G.degree(n)
            delay = G.nodes[n].get("delay", 99999.0)
            # Tuple for sorting: depth asc, degree desc (-deg), delay asc
            scored_nodes.append((depth, -deg, delay, n))
            
        scored_nodes.sort()
        for _, _, _, n in scored_nodes:
            if len(subset) >= max_nodes:
                break
            subset.add(n)
        chosen_nodes = subset

    # Build node objects
    nodes_payload = []
    for n in chosen_nodes:
        data = G.nodes[n]
        in_d = G.in_degree(n)
        out_d = G.out_degree(n)
        nodes_payload.append({
            "id": str(n),
            "uid": str(n),
            "tweet_id": str(data.get("tweet_id", "")),
            "delay": round(float(data.get("delay", 0.0)), 2),
            "depth": int(data.get("depth", 0)),
            "is_root": bool(n == root_uid),
            "in_degree": in_d,
            "out_degree": out_d,
            "total_degree": in_d + out_d,
        })

    # Build edge objects between chosen nodes
    edges_payload = []
    for u, v, edata in G.edges(data=True):
        if u in chosen_nodes and v in chosen_nodes:
            edges_payload.append({
                "source": str(u),
                "target": str(v),
                "delay_delta": round(float(edata.get("delay_delta", 0.0)), 2),
            })

    return {
        "post_id": G.graph.get("post_id", ""),
        "dataset": G.graph.get("dataset", ""),
        "label": G.graph.get("label", "unknown"),
        "source_text": G.graph.get("source_text", ""),
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "displayed_nodes": len(nodes_payload),
        "displayed_edges": len(edges_payload),
        "is_truncated": is_truncated,
        "max_nodes_limit": max_nodes,
        "root_uid": str(root_uid),
        "nodes": nodes_payload,
        "edges": edges_payload,
        "disclaimer": GRAPH_DISCLAIMER_TEXT,
    }
