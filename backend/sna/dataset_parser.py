"""
Dataset parser for Twitter15 and Twitter16 rumor detection benchmarks.
Parses:
- label.txt: label:source_tweet_id (e.g., non-rumor, false, true, unverified)
- source_tweets.txt: source_tweet_id \t source_tweet_content
- tree/<source_tweet_id>.txt: propagation trees formatted as:
    ['parent_uid', 'parent_tweet_id', 'parent_time_delay']->['child_uid', 'child_tweet_id', 'child_time_delay']
"""

import ast
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

logger = logging.getLogger(__name__)

# Base path to dataset
DEFAULT_DATASET_ROOT = (
    Path(__file__).resolve().parent.parent.parent
    / "dataset"
    / "rumor_detection_acl2017"
)

# In-memory caches for fast API responses
_DATASET_CACHE: Dict[str, Dict[str, Any]] = {}
_GRAPH_CACHE: Dict[str, nx.DiGraph] = {}


def get_dataset_root() -> Path:
    return DEFAULT_DATASET_ROOT


def get_dataset_path(dataset_name: str) -> Path:
    norm = dataset_name.lower().strip()
    if norm not in ("twitter15", "twitter16"):
        raise ValueError(f"Unknown dataset '{dataset_name}'. Supported: 'twitter15', 'twitter16'")
    path = get_dataset_root() / norm
    if not path.exists():
        raise FileNotFoundError(f"Dataset path not found: {path}")
    return path


def load_dataset_metadata(dataset_name: str) -> Dict[str, Any]:
    """Loads labels, source tweets, and tree file names for a dataset."""
    norm = dataset_name.lower().strip()
    if norm in _DATASET_CACHE:
        return _DATASET_CACHE[norm]

    base_path = get_dataset_path(norm)
    label_path = base_path / "label.txt"
    source_path = base_path / "source_tweets.txt"
    tree_dir = base_path / "tree"

    labels: Dict[str, str] = {}
    if label_path.exists():
        with open(label_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if ":" in line:
                    lbl, tid = line.split(":", 1)
                    labels[tid.strip()] = lbl.strip().lower()

    source_tweets: Dict[str, str] = {}
    if source_path.exists():
        with open(source_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if "\t" in line:
                    parts = line.split("\t", 1)
                    source_tweets[parts[0].strip()] = parts[1].strip()

    available_trees = set()
    if tree_dir.exists():
        for p in tree_dir.glob("*.txt"):
            available_trees.add(p.stem)

    _DATASET_CACHE[norm] = {
        "labels": labels,
        "source_tweets": source_tweets,
        "available_trees": available_trees,
        "tree_dir": tree_dir,
    }
    return _DATASET_CACHE[norm]


def parse_tree_file(tree_file_path: Path) -> List[Tuple[Tuple[str, str, float], Tuple[str, str, float]]]:
    """
    Parses a single tree text file into list of parent-child records:
    [((p_uid, p_tid, p_delay), (c_uid, c_tid, c_delay)), ...]
    """
    records = []
    if not tree_file_path.exists():
        return records

    with open(tree_file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or "->" not in line:
                continue
            left, right = line.split("->", 1)
            try:
                p = ast.literal_eval(left.strip())
                c = ast.literal_eval(right.strip())
                # Format: ['uid', 'tweet_id', 'time_delay']
                p_uid = str(p[0])
                p_tid = str(p[1])
                p_delay = float(p[2]) if p[2] is not None else 0.0

                c_uid = str(c[0])
                c_tid = str(c[1])
                c_delay = float(c[2]) if c[2] is not None else 0.0

                records.append(((p_uid, p_tid, p_delay), (c_uid, c_tid, c_delay)))
            except Exception as e:
                logger.debug(f"Failed to parse line '{line}': {e}")
                continue
    return records


def build_propagation_graph(dataset_name: str, post_id: str) -> nx.DiGraph:
    """
    Builds a NetworkX DiGraph representing the observed user-to-user propagation cascade.
    Nodes are users (uid), with attributes:
    - uid: User identifier
    - tweet_id: Associated tweet ID
    - delay: Time delay in minutes from the source post
    - is_root: True if this is the root source author
    - depth: Shortest path distance from root user
    Edges are directed parent -> child propagation events.
    """
    cache_key = f"{dataset_name.lower()}:{post_id.strip()}"
    if cache_key in _GRAPH_CACHE:
        return _GRAPH_CACHE[cache_key]

    meta = load_dataset_metadata(dataset_name)
    tree_path = meta["tree_dir"] / f"{post_id.strip()}.txt"
    if not tree_path.exists():
        raise FileNotFoundError(f"Propagation tree for post {post_id} not found in {dataset_name}.")

    records = parse_tree_file(tree_path)
    G = nx.DiGraph(
        post_id=post_id,
        dataset=dataset_name,
        label=meta["labels"].get(post_id, "unknown"),
        source_text=meta["source_tweets"].get(post_id, ""),
    )

    root_uid = None

    for (p_uid, p_tid, p_delay), (c_uid, c_tid, c_delay) in records:
        if p_uid == "ROOT":
            # This identifies the source tweet's initial author
            root_uid = c_uid
            G.add_node(
                c_uid,
                uid=c_uid,
                tweet_id=c_tid,
                delay=c_delay,
                is_root=True,
            )
        else:
            # Add parent node if not present
            if not G.has_node(p_uid):
                G.add_node(
                    p_uid,
                    uid=p_uid,
                    tweet_id=p_tid,
                    delay=p_delay,
                    is_root=(p_uid == root_uid),
                )
            # Add child node
            if not G.has_node(c_uid):
                G.add_node(
                    c_uid,
                    uid=c_uid,
                    tweet_id=c_tid,
                    delay=c_delay,
                    is_root=False,
                )
            # Add directed edge if not a self-loop
            if p_uid != c_uid:
                delay_delta = max(0.0, c_delay - p_delay)
                G.add_edge(p_uid, c_uid, delay_delta=delay_delta)

    # Set root if not explicitly detected from 'ROOT' line
    if not root_uid and len(G.nodes) > 0:
        # Find node with in-degree 0 and smallest delay
        in_zeros = [n for n, d in G.in_degree() if d == 0]
        if in_zeros:
            root_uid = min(in_zeros, key=lambda n: G.nodes[n].get("delay", 0.0))
        else:
            root_uid = min(G.nodes, key=lambda n: G.nodes[n].get("delay", 0.0))
        G.nodes[root_uid]["is_root"] = True

    G.graph["root_uid"] = root_uid

    # Compute depth (level) for each node using shortest path from root
    if root_uid and G.has_node(root_uid):
        try:
            lengths = nx.single_source_shortest_path_length(G, root_uid)
            for node, dist in lengths.items():
                G.nodes[node]["depth"] = dist
        except Exception:
            pass

    # For any disconnected nodes, assign depth based on closest ancestor or fallback to 1
    for node in G.nodes:
        if "depth" not in G.nodes[node]:
            G.nodes[node]["depth"] = 0 if node == root_uid else 1

    _GRAPH_CACHE[cache_key] = G
    return G


def get_post_summary(dataset_name: str, post_id: str) -> Optional[Dict[str, Any]]:
    """Returns quick metadata summary for a single post."""
    meta = load_dataset_metadata(dataset_name)
    if post_id not in meta["available_trees"]:
        return None

    tree_path = meta["tree_dir"] / f"{post_id}.txt"
    line_count = 0
    with open(tree_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                line_count += 1

    return {
        "post_id": post_id,
        "dataset": dataset_name,
        "label": meta["labels"].get(post_id, "unknown"),
        "text": meta["source_tweets"].get(post_id, ""),
        "tree_lines": line_count,
    }


def list_source_posts(
    dataset_name: str,
    label_filter: Optional[str] = None,
    query: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> Dict[str, Any]:
    """Returns paginated list of source posts for UI dropdown/search."""
    meta = load_dataset_metadata(dataset_name)
    posts = []

    for post_id in meta["available_trees"]:
        label = meta["labels"].get(post_id, "unknown")
        if label_filter and label_filter.lower() != "all" and label.lower() != label_filter.lower():
            continue

        text = meta["source_tweets"].get(post_id, "")
        if query:
            q = query.lower()
            if q not in post_id and q not in text.lower() and q not in label.lower():
                continue

        posts.append({
            "post_id": post_id,
            "label": label,
            "text": text,
            "text_preview": (text[:95] + "...") if len(text) > 95 else text,
        })

    # Sort stably by post_id
    posts.sort(key=lambda x: x["post_id"])
    total = len(posts)
    paginated = posts[offset : offset + limit]

    return {
        "dataset": dataset_name,
        "total": total,
        "limit": limit,
        "offset": offset,
        "posts": paginated,
    }
