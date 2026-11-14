"""
Centrality and influence analysis engine for propagation graphs.
Calculates strictly:
1. Degree Centrality (in-degree, out-degree, total degree)
2. Betweenness Centrality (directed shortest paths)
3. PageRank (link-structure structural rank)
"""

from typing import Any, Dict, List
import networkx as nx

CENTRALITY_DISCLAIMER_TEXT = (
    "These metrics measure structural importance within the recorded propagation cascade. "
    "They reflect observed forwarding and bridging positions in this specific rumor tree, "
    "and do not measure a person's actual real-world popularity, global influence, or total Twitter follower count."
)

METRIC_EXPLANATIONS = {
    "degree_centrality": (
        "Degree Centrality measures how many direct connections a user has in the cascade. "
        "In this directed propagation tree, Out-Degree shows how many users directly forwarded or retweeted "
        "from this user (information amplification), while In-Degree reflects the user from whom information was received."
    ),
    "betweenness_centrality": (
        "Betweenness Centrality quantifies how often a user acts as a critical bridge on the shortest propagation "
        "paths between all other user pairs. High-betweenness users control information flow between different branches of the cascade."
    ),
    "pagerank": (
        "PageRank estimates structural importance based on incoming propagation links and the relative authority "
        "of connecting nodes, modeling information transfer probabilities across the cascade."
    ),
}


def calculate_centrality_metrics(G: nx.DiGraph, top_k: int = 5) -> Dict[str, Any]:
    """
    Computes Degree Centrality, Betweenness Centrality, and PageRank for the directed graph.
    Returns ranked top-k nodes for each metric plus detailed node records.
    """
    num_nodes = G.number_of_nodes()
    if num_nodes == 0:
        return {
            "top_k": top_k,
            "total_nodes": 0,
            "degree_centrality": [],
            "betweenness_centrality": [],
            "pagerank": [],
            "explanations": METRIC_EXPLANATIONS,
            "disclaimer": CENTRALITY_DISCLAIMER_TEXT,
        }

    # 1. Degree Centrality
    # In NetworkX for directed graphs:
    # in_degree_centrality = in_degree / (N - 1)
    # out_degree_centrality = out_degree / (N - 1)
    # degree_centrality = (in_degree + out_degree) / (N - 1)
    in_deg_cent = nx.in_degree_centrality(G)
    out_deg_cent = nx.out_degree_centrality(G)
    tot_deg_cent = nx.degree_centrality(G)

    # 2. Betweenness Centrality
    # For nx.DiGraph, betweenness_centrality automatically respects edge directions
    if num_nodes > 2:
        bet_cent = nx.betweenness_centrality(G, normalized=True)
    else:
        bet_cent = {n: 0.0 for n in G.nodes()}

    # 3. PageRank
    try:
        pr_cent = nx.pagerank(G, alpha=0.85, max_iter=200)
    except Exception:
        # Fallback if convergence issues occur on isolated structures
        pr_cent = {n: 1.0 / num_nodes for n in G.nodes()}

    # Helper to build node record
    def make_record(uid: str, score: float, rank: int, metric_name: str) -> Dict[str, Any]:
        node_data = G.nodes.get(uid, {})
        in_d = G.in_degree(uid)
        out_d = G.out_degree(uid)
        return {
            "rank": rank,
            "uid": uid,
            "score": round(float(score), 5),
            "metric": metric_name,
            "in_degree": in_d,
            "out_degree": out_d,
            "total_degree": in_d + out_d,
            "delay_minutes": round(float(node_data.get("delay", 0.0)), 2),
            "depth": int(node_data.get("depth", 0)),
            "is_root": bool(node_data.get("is_root", False)),
        }

    # Rank top-k for Degree Centrality (based on total degree centrality)
    sorted_deg = sorted(tot_deg_cent.items(), key=lambda item: item[1], reverse=True)
    top_deg = [
        make_record(uid, score, idx + 1, "degree_centrality")
        for idx, (uid, score) in enumerate(sorted_deg[:top_k])
    ]

    # Rank top-k for Betweenness Centrality
    sorted_bet = sorted(bet_cent.items(), key=lambda item: item[1], reverse=True)
    top_bet = [
        make_record(uid, score, idx + 1, "betweenness_centrality")
        for idx, (uid, score) in enumerate(sorted_bet[:top_k])
    ]

    # Rank top-k for PageRank
    sorted_pr = sorted(pr_cent.items(), key=lambda item: item[1], reverse=True)
    top_pr = [
        make_record(uid, score, idx + 1, "pagerank")
        for idx, (uid, score) in enumerate(sorted_pr[:top_k])
    ]

    # Combined map of all nodes with all 3 metrics for fast client highlighting
    node_metrics_map: Dict[str, Dict[str, float]] = {}
    for uid in G.nodes():
        node_metrics_map[uid] = {
            "degree": round(float(tot_deg_cent.get(uid, 0.0)), 5),
            "in_degree_cent": round(float(in_deg_cent.get(uid, 0.0)), 5),
            "out_degree_cent": round(float(out_deg_cent.get(uid, 0.0)), 5),
            "betweenness": round(float(bet_cent.get(uid, 0.0)), 5),
            "pagerank": round(float(pr_cent.get(uid, 0.0)), 5),
        }

    return {
        "top_k": top_k,
        "total_nodes": num_nodes,
        "degree_centrality": top_deg,
        "betweenness_centrality": top_bet,
        "pagerank": top_pr,
        "node_metrics_map": node_metrics_map,
        "explanations": METRIC_EXPLANATIONS,
        "disclaimer": CENTRALITY_DISCLAIMER_TEXT,
    }
