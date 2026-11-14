"""
Social Network Analysis (SNA) and Propagation Package for Rumor Cascades.
Features:
1. Social Network / Propagation Graph
2. Graph-Based Influence Analysis (Degree, Betweenness, PageRank)
3. Misinformation Propagation Analysis (Cascade size, depth levels, timing)
4. Propagation Mitigation Simulation (Independent Cascade model)
"""

from .dataset_parser import (
    build_propagation_graph,
    get_dataset_path,
    get_dataset_root,
    get_post_summary,
    list_source_posts,
    load_dataset_metadata,
)
from .centrality_engine import (
    CENTRALITY_DISCLAIMER_TEXT,
    calculate_centrality_metrics,
)
from .propagation_engine import (
    PROPAGATION_DISCLAIMER_TEXT,
    analyze_propagation,
)
from .simulation_engine import (
    SIMULATION_DISCLAIMER_TEXT,
    simulate_mitigation,
)
from .graph_export import (
    GRAPH_DISCLAIMER_TEXT,
    export_graph_for_visualization,
)

__all__ = [
    "build_propagation_graph",
    "get_dataset_path",
    "get_dataset_root",
    "get_post_summary",
    "list_source_posts",
    "load_dataset_metadata",
    "calculate_centrality_metrics",
    "CENTRALITY_DISCLAIMER_TEXT",
    "analyze_propagation",
    "PROPAGATION_DISCLAIMER_TEXT",
    "simulate_mitigation",
    "SIMULATION_DISCLAIMER_TEXT",
    "export_graph_for_visualization",
    "GRAPH_DISCLAIMER_TEXT",
]
