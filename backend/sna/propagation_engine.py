"""
Propagation analysis engine for rumor cascades.
Computes:
- Total recorded users and propagation edges
- Depth level distributions (stages of propagation)
- Time delay statistics (min, max, mean, median in minutes)
- Temporal cascade adoption bins and cumulative adoption trajectory
"""

import math
from typing import Any, Dict, List
import networkx as nx

PROPAGATION_DISCLAIMER_TEXT = (
    "All statistics are calculated strictly from the selected source post's observed propagation tree in the benchmark. "
    "Because the dataset captures observed retweeting/forwarding relationships rather than global impression data, "
    "this does not represent the total number of people who saw the post or overall Twitter audience reach."
)


def analyze_propagation(G: nx.DiGraph) -> Dict[str, Any]:
    """
    Analyzes cascade spread dynamics, depth hierarchy, and time-delay kinetics.
    """
    total_nodes = G.number_of_nodes()
    total_edges = G.number_of_edges()

    if total_nodes == 0:
        return {
            "total_nodes": 0,
            "total_edges": 0,
            "max_depth": 0,
            "depth_distribution": {},
            "delay_stats": {},
            "time_bins": [],
            "cumulative_timeline": [],
            "disclaimer": PROPAGATION_DISCLAIMER_TEXT,
        }

    # Extract delays and depths
    delays: List[float] = []
    depth_counts: Dict[int, int] = {}
    node_timeline: List[Dict[str, Any]] = []

    for node, data in G.nodes(data=True):
        d = float(data.get("delay", 0.0))
        depth = int(data.get("depth", 0))
        delays.append(d)
        depth_counts[depth] = depth_counts.get(depth, 0) + 1
        node_timeline.append({
            "uid": str(node),
            "delay": round(d, 2),
            "depth": depth,
            "is_root": bool(data.get("is_root", False)),
        })

    # Sort node timeline by delay
    node_timeline.sort(key=lambda item: item["delay"])

    # Compute delay statistics
    delays_sorted = sorted(delays)
    min_delay = delays_sorted[0]
    max_delay = delays_sorted[-1]
    mean_delay = sum(delays_sorted) / len(delays_sorted)
    
    mid = len(delays_sorted) // 2
    if len(delays_sorted) % 2 == 0:
        median_delay = (delays_sorted[mid - 1] + delays_sorted[mid]) / 2.0
    else:
        median_delay = delays_sorted[mid]

    max_depth = max(depth_counts.keys()) if depth_counts else 0

    # Depth level distribution formatted as sorted list of objects
    depth_levels = [
        {
            "level": lvl,
            "name": "Source Post (Root)" if lvl == 0 else f"Level {lvl} (Hop {lvl})",
            "count": depth_counts[lvl],
            "percentage": round((depth_counts[lvl] / total_nodes) * 100.0, 2),
        }
        for lvl in range(max_depth + 1)
        if lvl in depth_counts
    ]

    # Time-delay distribution bins (minutes)
    # Bins: 0-10m, 10-30m, 30-60m, 1-3h (60-180m), 3-12h (180-720m), 12-24h (720-1440m), 1-3d, >3d
    bin_defs = [
        ("0 - 10 min", 0.0, 10.0),
        ("10 - 30 min", 10.0, 30.0),
        ("30 - 60 min", 30.0, 60.0),
        ("1 - 3 hours", 60.0, 180.0),
        ("3 - 12 hours", 180.0, 720.0),
        ("12 - 24 hours", 720.0, 1440.0),
        ("1 - 3 days", 1440.0, 4320.0),
        ("> 3 days", 4320.0, float("inf")),
    ]

    time_bins = []
    for label, low, high in bin_defs:
        if high == float("inf"):
            count = sum(1 for d in delays if d > low)
        else:
            count = sum(1 for d in delays if (low <= d < high if low > 0 else low <= d <= high))
        time_bins.append({
            "label": label,
            "count": count,
            "percentage": round((count / total_nodes) * 100.0, 2) if total_nodes > 0 else 0.0,
        })

    # Cumulative timeline (sample up to 30 curve points for smooth charting)
    cumulative_points = []
    step = max(1, len(node_timeline) // 25)
    for i in range(0, len(node_timeline), step):
        cumulative_points.append({
            "adopters": i + 1,
            "percentage": round(((i + 1) / total_nodes) * 100.0, 1),
            "delay_minutes": node_timeline[i]["delay"],
            "delay_hours": round(node_timeline[i]["delay"] / 60.0, 2),
        })
    # Always include the final point
    if cumulative_points[-1]["adopters"] != total_nodes:
        cumulative_points.append({
            "adopters": total_nodes,
            "percentage": 100.0,
            "delay_minutes": node_timeline[-1]["delay"],
            "delay_hours": round(node_timeline[-1]["delay"] / 60.0, 2),
        })

    # Propagation velocity indicators
    # How fast did the first 50% and 80% adopt?
    half_index = int(math.ceil(total_nodes * 0.5)) - 1
    eighty_index = int(math.ceil(total_nodes * 0.8)) - 1
    time_to_50_pct = node_timeline[half_index]["delay"] if 0 <= half_index < len(node_timeline) else 0.0
    time_to_80_pct = node_timeline[eighty_index]["delay"] if 0 <= eighty_index < len(node_timeline) else 0.0

    return {
        "post_id": G.graph.get("post_id", ""),
        "dataset": G.graph.get("dataset", ""),
        "label": G.graph.get("label", "unknown"),
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "max_depth": max_depth,
        "depth_levels": depth_levels,
        "delay_stats": {
            "min_minutes": round(min_delay, 2),
            "max_minutes": round(max_delay, 2),
            "mean_minutes": round(mean_delay, 2),
            "median_minutes": round(median_delay, 2),
            "time_to_50_pct_minutes": round(time_to_50_pct, 2),
            "time_to_80_pct_minutes": round(time_to_80_pct, 2),
        },
        "time_bins": time_bins,
        "cumulative_timeline": cumulative_points,
        "disclaimer": PROPAGATION_DISCLAIMER_TEXT,
    }
