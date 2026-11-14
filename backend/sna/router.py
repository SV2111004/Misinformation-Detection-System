"""
FastAPI Router for Social Network Analysis (SNA) and Rumor Propagation.
Exposes endpoints for:
1. Social Network / Propagation Graph
2. Graph-Based Influence Analysis
3. Misinformation Propagation Analysis
4. Propagation Mitigation Simulation
"""

from typing import Any, Dict, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from .dataset_parser import (
    build_propagation_graph,
    list_source_posts,
    load_dataset_metadata,
)
from .centrality_engine import calculate_centrality_metrics
from .propagation_engine import analyze_propagation
from .simulation_engine import simulate_mitigation
from .graph_export import export_graph_for_visualization

router = APIRouter(prefix="/api/sna", tags=["Social Network Analysis & Propagation"])


class SimulationRequest(BaseModel):
    dataset: str = Field(
        "twitter15",
        description="Dataset name: 'twitter15' or 'twitter16'",
    )
    post_id: str = Field(
        ...,
        description="Source post ID of the rumor cascade",
        min_length=1,
    )
    p: float = Field(
        0.3,
        description="Propagation transmission probability under Independent Cascade (0.01 to 1.0)",
        ge=0.01,
        le=1.0,
    )
    num_runs: int = Field(
        50,
        description="Number of Monte Carlo simulation runs (5 to 200)",
        ge=5,
        le=200,
    )
    seed: int = Field(
        42,
        description="Random seed for reproducible Monte Carlo execution",
    )
    intervention_metric: str = Field(
        "betweenness",
        description="Metric used to select intervention targets: 'betweenness', 'degree', or 'pagerank'",
    )
    intervention_top_k: int = Field(
        3,
        description="Number of top structurally important nodes to intervene on (1 to 10)",
        ge=1,
        le=10,
    )


@router.get("/datasets")
def get_datasets() -> Dict[str, Any]:
    """
    Returns available propagation benchmarks (Twitter15, Twitter16) and summary counts.
    """
    results = []
    for name in ("twitter15", "twitter16"):
        try:
            meta = load_dataset_metadata(name)
            label_counts: Dict[str, int] = {}
            for lbl in meta["labels"].values():
                label_counts[lbl] = label_counts.get(lbl, 0) + 1
            results.append({
                "dataset": name,
                "name": name.upper(),
                "total_trees": len(meta["available_trees"]),
                "total_source_tweets": len(meta["source_tweets"]),
                "labels": label_counts,
                "description": (
                    f"{name.upper()} rumor detection benchmark containing observed "
                    f"propagation cascades, post timestamps, and rumor verification labels."
                ),
            })
        except Exception as e:
            results.append({
                "dataset": name,
                "name": name.upper(),
                "error": str(e),
            })
    return {"datasets": results}


@router.get("/posts")
def get_posts(
    dataset: str = Query("twitter15", description="Dataset name: 'twitter15' or 'twitter16'"),
    label: Optional[str] = Query("all", description="Filter by rumor label: all, non-rumor, false, true, unverified"),
    query: Optional[str] = Query(None, description="Search keyword or post ID"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> Dict[str, Any]:
    """
    Lists source posts for the selected dataset with search and pagination support.
    """
    try:
        return list_source_posts(
            dataset_name=dataset,
            label_filter=label,
            query=query,
            limit=limit,
            offset=offset,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load posts: {str(e)}")


@router.get("/graph/{dataset}/{post_id}")
def get_propagation_graph(
    dataset: str,
    post_id: str,
    max_nodes: int = Query(150, ge=10, le=500, description="Display limit for frontend rendering"),
) -> Dict[str, Any]:
    """
    Feature 1: Returns directed propagation graph nodes, edges, and metadata for the selected rumor.
    """
    try:
        G = build_propagation_graph(dataset, post_id)
        return export_graph_for_visualization(G, max_nodes=max_nodes)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph construction failed: {str(e)}")


@router.get("/centrality/{dataset}/{post_id}")
def get_centrality_analysis(
    dataset: str,
    post_id: str,
    top_k: int = Query(5, ge=1, le=20, description="Number of top structurally important nodes to return"),
) -> Dict[str, Any]:
    """
    Feature 2: Returns Degree Centrality, Betweenness Centrality, and PageRank rankings for the cascade.
    """
    try:
        G = build_propagation_graph(dataset, post_id)
        return calculate_centrality_metrics(G, top_k=top_k)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Centrality calculation failed: {str(e)}")


@router.get("/propagation/{dataset}/{post_id}")
def get_propagation_analysis(
    dataset: str,
    post_id: str,
) -> Dict[str, Any]:
    """
    Feature 3: Returns cascade statistics, depth levels hierarchy, and time-delay kinetics.
    """
    try:
        G = build_propagation_graph(dataset, post_id)
        return analyze_propagation(G)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Propagation analysis failed: {str(e)}")


@router.post("/simulation")
def post_mitigation_simulation(
    request: SimulationRequest,
) -> Dict[str, Any]:
    """
    Feature 4: Runs Independent Cascade simulation comparing Scenario A (Normal) vs Scenario B (Targeted Intervention).
    """
    try:
        G = build_propagation_graph(request.dataset, request.post_id)
        return simulate_mitigation(
            G=G,
            p=request.p,
            num_runs=request.num_runs,
            seed=request.seed,
            intervention_metric=request.intervention_metric,
            intervention_top_k=request.intervention_top_k,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")
