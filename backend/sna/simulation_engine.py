"""
Independent Cascade (IC) simulation engine for rumor propagation and mitigation analysis.

Evaluates two scenarios under controlled, reproducible conditions:
- Scenario A: Normal propagation over the observed graph structure with probability p
- Scenario B: Mitigation intervention at structurally important nodes (selected by
  Betweenness Centrality, Degree Centrality, or PageRank), preventing them from spreading onward.
"""

import random
from typing import Any, Dict, List, Optional, Set
import networkx as nx

SIMULATION_DISCLAIMER_TEXT = (
    "Hypothetical simulation based on recorded propagation structure and assumed probabilities. "
    "These results illustrate structural cascade vulnerability and theoretical mitigation effects; "
    "they do not represent guaranteed real-world outcomes or actual historical intervention events."
)


def run_single_ic(
    G: nx.DiGraph,
    seed_node: str,
    p: float,
    blocked_nodes: Set[str],
    rng: random.Random,
) -> Tuple[Set[str], List[int]]:
    """
    Executes one Independent Cascade simulation from seed_node.
    Nodes in blocked_nodes cannot activate their downstream neighbors.
    Returns:
    - (set of all activated nodes, list of new activations per diffusion step)
    """
    if not G.has_node(seed_node):
        return set(), []

    activated: Set[str] = {seed_node}
    newly_activated: List[str] = [seed_node]
    step_counts: List[int] = [1]

    while newly_activated:
        next_step: List[str] = []
        for u in newly_activated:
            # If node u is intervened / blocked, it does not forward onward
            if u in blocked_nodes:
                continue

            for v in G.successors(u):
                if v not in activated:
                    # Attempt activation with probability p
                    if rng.random() <= p:
                        activated.add(v)
                        next_step.append(v)

        if next_step:
            step_counts.append(len(next_step))
        newly_activated = next_step

    return activated, step_counts


def simulate_mitigation(
    G: nx.DiGraph,
    p: float = 0.3,
    num_runs: int = 50,
    seed: int = 42,
    intervention_metric: str = "betweenness",
    intervention_top_k: int = 3,
) -> Dict[str, Any]:
    """
    Runs paired Monte Carlo Independent Cascade simulations comparing
    Scenario A (Normal spread) vs Scenario B (Targeted node intervention).
    """
    total_nodes = G.number_of_nodes()
    if total_nodes == 0:
        return {
            "total_nodes": 0,
            "scenario_a": {"mean_reached": 0, "reached_percentage": 0.0},
            "scenario_b": {"mean_reached": 0, "reached_percentage": 0.0},
            "reduction_nodes": 0,
            "reduction_percentage": 0.0,
            "intervention_nodes": [],
            "disclaimer": SIMULATION_DISCLAIMER_TEXT,
        }

    root_uid = G.graph.get("root_uid")
    if not root_uid or not G.has_node(root_uid):
        # Fallback to node with minimum delay
        root_uid = min(G.nodes, key=lambda n: G.nodes[n].get("delay", 0.0))

    # Identify candidate intervention nodes based on selected metric
    # Avoid selecting the root itself as the ONLY intervened node if other nodes exist,
    # because fact-checking typically intervenes on prominent amplifiers / early spreaders.
    metric_norm = intervention_metric.lower().strip()
    if metric_norm == "degree":
        scores = nx.degree_centrality(G)
        metric_title = "Degree Centrality (Direct Amplifiers)"
        metric_reason = "Selected as top direct hubs with the highest number of propagation connections."
    elif metric_norm == "pagerank":
        try:
            scores = nx.pagerank(G, alpha=0.85, max_iter=200)
        except Exception:
            scores = {n: 1.0 for n in G.nodes()}
        metric_title = "PageRank (Structural Link Authority)"
        metric_reason = "Selected as nodes with high link authority in the cascade network."
    else:
        # Default to betweenness centrality
        if total_nodes > 2:
            scores = nx.betweenness_centrality(G, normalized=True)
        else:
            scores = {n: 0.0 for n in G.nodes()}
        metric_title = "Betweenness Centrality (Information Bridges)"
        metric_reason = "Selected as critical bridges controlling information flow between cascade branches."

    # Sort descending by score
    sorted_nodes = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    
    # Exclude root from intervention if possible, so simulation demonstrates stopping forward spread
    candidate_nodes = [n for n, s in sorted_nodes if n != root_uid]
    if len(candidate_nodes) < intervention_top_k:
        candidate_nodes = [n for n, s in sorted_nodes]

    chosen_intervention_uids = candidate_nodes[:intervention_top_k]
    blocked_set = set(chosen_intervention_uids)

    # Detailed info for intervened nodes
    intervention_nodes_info = []
    for rank_idx, uid in enumerate(chosen_intervention_uids):
        node_data = G.nodes.get(uid, {})
        in_d = G.in_degree(uid)
        out_d = G.out_degree(uid)
        intervention_nodes_info.append({
            "rank": rank_idx + 1,
            "uid": uid,
            "score": round(float(scores.get(uid, 0.0)), 5),
            "in_degree": in_d,
            "out_degree": out_d,
            "delay_minutes": round(float(node_data.get("delay", 0.0)), 2),
            "depth": int(node_data.get("depth", 0)),
            "reason": f"Top #{rank_idx + 1} by {metric_title}. Out-degree: {out_d} direct retweets.",
        })

    # Run Paired Monte Carlo runs with controlled seeds
    runs_a_reached = []
    runs_b_reached = []
    max_steps = 1
    step_history_a = []
    step_history_b = []

    for run_idx in range(num_runs):
        run_seed = seed + run_idx * 10007

        # Scenario A (Normal propagation, no blocked nodes)
        rng_a = random.Random(run_seed)
        act_a, steps_a = run_single_ic(G, root_uid, p, blocked_nodes=set(), rng=rng_a)
        runs_a_reached.append(len(act_a))
        step_history_a.append(steps_a)
        if len(steps_a) > max_steps:
            max_steps = len(steps_a)

        # Scenario B (Mitigation intervention, blocked_set cannot forward)
        rng_b = random.Random(run_seed)
        act_b, steps_b = run_single_ic(G, root_uid, p, blocked_nodes=blocked_set, rng=rng_b)
        runs_b_reached.append(len(act_b))
        step_history_b.append(steps_b)
        if len(steps_b) > max_steps:
            max_steps = len(steps_b)

    mean_a = sum(runs_a_reached) / len(runs_a_reached)
    mean_b = sum(runs_b_reached) / len(runs_b_reached)
    diff = max(0.0, mean_a - mean_b)
    pct_reduction = (diff / mean_a * 100.0) if mean_a > 0 else 0.0

    # Calculate average cumulative adoption per diffusion step for charting
    step_comparison = []
    for step_idx in range(max_steps):
        # Average newly activated at this step
        vals_a = [s[step_idx] if step_idx < len(s) else 0 for s in step_history_a]
        vals_b = [s[step_idx] if step_idx < len(s) else 0 for s in step_history_b]
        step_comparison.append({
            "step": step_idx,
            "step_label": f"Step {step_idx}" if step_idx > 0 else "Source Post (Step 0)",
            "scenario_a_active": round(sum(vals_a) / len(vals_a), 2),
            "scenario_b_active": round(sum(vals_b) / len(vals_b), 2),
        })

    return {
        "parameters": {
            "propagation_probability": p,
            "monte_carlo_runs": num_runs,
            "random_seed": seed,
            "intervention_metric": metric_norm,
            "intervention_metric_title": metric_title,
            "intervention_top_k": intervention_top_k,
            "metric_reason": metric_reason,
            "source_node": root_uid,
        },
        "total_cascade_nodes": total_nodes,
        "scenario_a": {
            "name": "Scenario A: Normal Propagation",
            "mean_reached_nodes": round(mean_a, 2),
            "reach_percentage": round((mean_a / total_nodes) * 100.0, 2),
            "min_reached": min(runs_a_reached),
            "max_reached": max(runs_a_reached),
            "description": "Baseline spread across observed propagation tree with no intervention.",
        },
        "scenario_b": {
            "name": f"Scenario B: Targeted Intervention ({intervention_top_k} nodes)",
            "mean_reached_nodes": round(mean_b, 2),
            "reach_percentage": round((mean_b / total_nodes) * 100.0, 2),
            "min_reached": min(runs_b_reached),
            "max_reached": max(runs_b_reached),
            "description": f"Intervention on top {intervention_top_k} {metric_title} nodes prevents outward forwarding.",
        },
        "comparison": {
            "difference_nodes": round(diff, 2),
            "percentage_reduction": round(pct_reduction, 2),
            "protection_summary": (
                f"Hypothetical intervention on {len(chosen_intervention_uids)} structural nodes reduced average simulated spread "
                f"from {round(mean_a, 1)} to {round(mean_b, 1)} nodes ({round(pct_reduction, 1)}% reduction)."
            ),
        },
        "step_comparison": step_comparison,
        "intervention_nodes": intervention_nodes_info,
        "disclaimer": SIMULATION_DISCLAIMER_TEXT,
    }
