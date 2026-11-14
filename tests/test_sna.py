"""
Unit and Integration Tests for Social Network Analysis and Propagation Modules.
Validates:
- Twitter15 and Twitter16 dataset parsing and tree construction
- Feature 1: Social Network / Propagation Graph API
- Feature 2: Graph-Based Influence Analysis (Degree, Betweenness, PageRank)
- Feature 3: Misinformation Propagation Analysis (Cascades, Depth, Time Delays)
- Feature 4: Propagation Mitigation Simulation (Independent Cascade, Scenarios A & B)
- Error handling for invalid datasets and posts
"""

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.sna import (
    analyze_propagation,
    build_propagation_graph,
    calculate_centrality_metrics,
    export_graph_for_visualization,
    list_source_posts,
    load_dataset_metadata,
    simulate_mitigation,
)

client = TestClient(app)


def test_dataset_metadata_loading():
    meta15 = load_dataset_metadata("twitter15")
    assert len(meta15["available_trees"]) == 1490
    assert len(meta15["labels"]) == 1490
    assert len(meta15["source_tweets"]) == 1490

    meta16 = load_dataset_metadata("twitter16")
    assert len(meta16["available_trees"]) == 818
    assert len(meta16["labels"]) == 818
    assert len(meta16["source_tweets"]) == 818


def test_list_source_posts():
    res = list_source_posts("twitter15", limit=10, offset=0)
    assert res["total"] == 1490
    assert len(res["posts"]) == 10
    first = res["posts"][0]
    assert "post_id" in first
    assert "label" in first
    assert "text" in first


def test_propagation_graph_construction():
    posts = list_source_posts("twitter15", limit=1)
    post_id = posts["posts"][0]["post_id"]
    G = build_propagation_graph("twitter15", post_id)

    assert G.number_of_nodes() > 0
    assert G.number_of_edges() > 0
    assert G.graph.get("root_uid") is not None

    # Check root attributes
    root_uid = G.graph.get("root_uid")
    assert G.nodes[root_uid]["is_root"] is True
    assert G.nodes[root_uid]["depth"] == 0


def test_centrality_metrics():
    posts = list_source_posts("twitter15", limit=1)
    post_id = posts["posts"][0]["post_id"]
    G = build_propagation_graph("twitter15", post_id)

    cent = calculate_centrality_metrics(G, top_k=5)
    assert len(cent["degree_centrality"]) <= 5
    assert len(cent["betweenness_centrality"]) <= 5
    assert len(cent["pagerank"]) <= 5

    # Check metric explanations and disclaimers exist
    assert "degree_centrality" in cent["explanations"]
    assert "betweenness_centrality" in cent["explanations"]
    assert "pagerank" in cent["explanations"]
    assert "disclaimer" in cent


def test_propagation_analysis():
    posts = list_source_posts("twitter15", limit=1)
    post_id = posts["posts"][0]["post_id"]
    G = build_propagation_graph("twitter15", post_id)

    prop = analyze_propagation(G)
    assert prop["total_nodes"] == G.number_of_nodes()
    assert prop["total_edges"] == G.number_of_edges()
    assert prop["max_depth"] >= 0
    assert len(prop["depth_levels"]) > 0
    assert "median_minutes" in prop["delay_stats"]
    assert len(prop["time_bins"]) > 0
    assert len(prop["cumulative_timeline"]) > 0


def test_mitigation_simulation_reproducibility():
    posts = list_source_posts("twitter15", limit=1)
    post_id = posts["posts"][0]["post_id"]
    G = build_propagation_graph("twitter15", post_id)

    sim1 = simulate_mitigation(G, p=0.3, num_runs=10, seed=42, intervention_top_k=3)
    sim2 = simulate_mitigation(G, p=0.3, num_runs=10, seed=42, intervention_top_k=3)

    assert sim1["scenario_a"]["mean_reached_nodes"] == sim2["scenario_a"]["mean_reached_nodes"]
    assert sim1["scenario_b"]["mean_reached_nodes"] == sim2["scenario_b"]["mean_reached_nodes"]
    assert sim1["comparison"]["difference_nodes"] >= 0
    assert "disclaimer" in sim1


def test_api_datasets_endpoint():
    resp = client.get("/api/sna/datasets")
    assert resp.status_code == 200
    data = resp.json()
    assert "datasets" in data
    assert len(data["datasets"]) == 2


def test_api_posts_endpoint():
    resp = client.get("/api/sna/posts?dataset=twitter15&limit=5")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1490
    assert len(data["posts"]) == 5


def test_api_graph_endpoint():
    resp = client.get("/api/sna/posts?dataset=twitter15&limit=1")
    post_id = resp.json()["posts"][0]["post_id"]

    resp_graph = client.get(f"/api/sna/graph/twitter15/{post_id}?max_nodes=100")
    assert resp_graph.status_code == 200
    data = resp_graph.json()
    assert "nodes" in data
    assert "edges" in data
    assert "disclaimer" in data


def test_api_centrality_endpoint():
    resp = client.get("/api/sna/posts?dataset=twitter15&limit=1")
    post_id = resp.json()["posts"][0]["post_id"]

    resp_cent = client.get(f"/api/sna/centrality/twitter15/{post_id}?top_k=5")
    assert resp_cent.status_code == 200
    data = resp_cent.json()
    assert len(data["degree_centrality"]) <= 5
    assert len(data["betweenness_centrality"]) <= 5
    assert len(data["pagerank"]) <= 5


def test_api_propagation_endpoint():
    resp = client.get("/api/sna/posts?dataset=twitter15&limit=1")
    post_id = resp.json()["posts"][0]["post_id"]

    resp_prop = client.get(f"/api/sna/propagation/twitter15/{post_id}")
    assert resp_prop.status_code == 200
    data = resp_prop.json()
    assert data["total_nodes"] > 0
    assert "delay_stats" in data


def test_api_simulation_endpoint():
    resp = client.get("/api/sna/posts?dataset=twitter15&limit=1")
    post_id = resp.json()["posts"][0]["post_id"]

    resp_sim = client.post("/api/sna/simulation", json={
        "dataset": "twitter15",
        "post_id": post_id,
        "p": 0.25,
        "num_runs": 10,
        "seed": 99,
        "intervention_metric": "betweenness",
        "intervention_top_k": 2,
    })
    assert resp_sim.status_code == 200
    data = resp_sim.json()
    assert "scenario_a" in data
    assert "scenario_b" in data
    assert "comparison" in data
    assert "intervention_nodes" in data


def test_api_error_handling():
    # Non-existent dataset
    resp1 = client.get("/api/sna/posts?dataset=invalid_dataset")
    assert resp1.status_code == 400

    # Non-existent post
    resp2 = client.get("/api/sna/graph/twitter15/non_existent_post_99999999")
    assert resp2.status_code == 404
