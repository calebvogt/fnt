"""Social networks from the GBI: SRI association, then igraph metrics.

Association is the Simple Ratio Index, which on group-by-individual data is the
Jaccard index: the events a pair shared over the events either of them attended.
Two animals that are always seen together score 1; two that never overlap score
0. Nothing is normalised by time, so a pair that co-occurs in many brief
intervals scores like a pair that co-occurs in a few long ones - use the
co-presence edge list when duration is what matters.

Everything downstream is igraph, matching the R original's library so the
numbers are comparable rather than merely similar.

Two of the network-level metrics the R pipeline recorded were misnamed, and the
names are corrected here:

* ``net_spectral_radius`` - R stored this as ``net_eigen_centrality``, but
  ``centr_eigen()$value`` returns the leading eigenvalue of the UNWEIGHTED
  adjacency matrix, not a centralisation score. Typical values are 3-6.
* ``net_mean_dist_weighted`` - R stored this as ``net_mean_dist`` and described
  it as "avg. # of edges between any two nodes". It is the WEIGHTED
  shortest-path length; with SRI weights it lands near 0.02, and an edge count
  can never be below 1.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .gbi_generator import GBI_META

NODE_COLUMNS = ["trial", "day", "name", "code", "sex", "phase", "group",
                "node_strength", "node_degree", "node_eigen_centrality",
                "node_betweenness", "node_closeness", "node_page_rank",
                "node_oppsex_strength"]
NET_COLUMNS = ["trial", "day", "n_nodes", "n_edges", "net_degree_centralization",
               "net_components", "net_spectral_radius", "net_mean_dist_weighted",
               "net_edge_density", "net_transitivity",
               "net_modularity_infomap", "net_modularity_infomap_groups",
               "net_modularity_fast_greedy", "net_modularity_fast_greedy_groups"]


def _import_igraph():
    """Import igraph, working around a missing MSVC OpenMP runtime.

    The python-igraph wheels link against ``VCOMP140.DLL``, which is part of the
    Visual C++ redistributable and is often simply absent on a Windows box that
    has never had Visual Studio on it. The import then fails with a bare "DLL
    load failed" that says nothing about which DLL. scikit-learn ships its own
    copy, so if one is already on disk we can point the loader at it rather
    than making the user chase an installer.
    """
    try:
        import igraph
        return igraph
    except (ImportError, OSError) as first:
        if not hasattr(os, "add_dll_directory"):
            raise
        import sysconfig
        site = sysconfig.get_paths().get("purelib", "")
        candidate = os.path.join(site, "sklearn", ".libs")
        if os.path.isdir(candidate):
            try:
                os.add_dll_directory(candidate)
                import igraph
                return igraph
            except Exception:
                pass
        raise ImportError(
            "python-igraph could not be loaded. On Windows this is usually a "
            "missing VCOMP140.DLL (the MSVC OpenMP runtime): install the "
            "Microsoft Visual C++ Redistributable, or use a conda-forge build "
            f"of python-igraph. Original error: {first}") from first


def sri_matrix(gbi_block: pd.DataFrame) -> np.ndarray:
    """Simple Ratio Index for every pair of columns in a 0/1 GBI block."""
    mat = gbi_block.to_numpy().astype(float)
    shared = mat.T @ mat
    attended = np.diag(shared).astype(float)
    union = attended[:, None] + attended[None, :] - shared
    with np.errstate(divide="ignore", invalid="ignore"):
        sri = np.where(union > 0, shared / union, 0.0)
    np.fill_diagonal(sri, 0.0)
    return sri


def build_graph(sri: np.ndarray, names: list[str]):
    igraph = _import_igraph()
    g = igraph.Graph.Weighted_Adjacency(sri.tolist(), mode="undirected",
                                        attr="weight", loops=False)
    g.vs["name"] = names
    return g


def network_for_day(gbi: pd.DataFrame, day: int, animals: list[str],
                    meta: pd.DataFrame | None = None
                    ) -> tuple[pd.DataFrame, dict] | tuple[None, None]:
    """Node table and network summary for one day's GBI."""
    block = gbi.loc[gbi["day"] == day, animals]
    if block.empty or not animals:
        return None, None

    sri = sri_matrix(block)
    g = build_graph(sri, animals)
    n = g.vcount()
    if n == 0:
        return None, None

    sex = {}
    if meta is not None and "sex" in meta.columns:
        sex = meta.drop_duplicates("name").set_index("name")["sex"].to_dict()

    strength = g.strength(weights="weight")
    degree = g.degree()
    eigen = g.eigenvector_centrality(directed=False, scale=True, weights="weight")
    between = g.betweenness(directed=False, weights=None)
    close = g.closeness(mode="all", normalized=True, weights=None)
    rank = g.pagerank(weights="weight")

    oppsex = []
    for v in g.vs:
        mine = sex.get(v["name"])
        total = 0.0
        for eid in g.incident(v):
            edge = g.es[eid]
            other = edge.source if edge.target == v.index else edge.target
            if mine is not None and sex.get(g.vs[other]["name"]) not in (None, mine):
                total += edge["weight"]
        oppsex.append(total)

    trial = gbi["trial"].iloc[0] if "trial" in gbi.columns else ""
    info = (meta.drop_duplicates("name").set_index("name")
            if meta is not None else None)

    def look(name, field):
        if info is None or field not in info.columns or name not in info.index:
            return ""
        return info.loc[name, field]

    nodes = pd.DataFrame({
        "trial": trial, "day": day, "name": animals,
        "code": [look(a, "code") for a in animals],
        "sex": [sex.get(a, "") for a in animals],
        "phase": [look(a, "phase") for a in animals],
        "group": [look(a, "group") for a in animals],
        "node_strength": strength, "node_degree": degree,
        "node_eigen_centrality": eigen, "node_betweenness": between,
        "node_closeness": close, "node_page_rank": rank,
        "node_oppsex_strength": oppsex})

    deg = np.asarray(degree, float)
    adjacency = np.array(g.get_adjacency(eids=False).data, float)
    infomap = g.community_infomap(edge_weights="weight")
    greedy = g.community_fastgreedy(weights="weight").as_clustering()
    summary = {
        "trial": trial, "day": day, "n_nodes": n, "n_edges": g.ecount(),
        # normalised by n(n-1), igraph's theoretical maximum for an undirected
        # graph with loops permitted - which is what the R original used
        "net_degree_centralization": (float((deg.max() - deg).sum()) / (n * (n - 1))
                                      if n > 1 else np.nan),
        "net_components": len(g.connected_components()),
        "net_spectral_radius": float(np.linalg.eigvalsh(adjacency).max()),
        "net_mean_dist_weighted": g.average_path_length(directed=False,
                                                        weights="weight"),
        "net_edge_density": g.density(loops=False),
        "net_transitivity": g.transitivity_undirected(),
        "net_modularity_infomap": infomap.modularity,
        "net_modularity_infomap_groups": len(infomap),
        "net_modularity_fast_greedy": greedy.modularity,
        "net_modularity_fast_greedy_groups": len(greedy)}
    return nodes, summary


def social_networks(gbi: pd.DataFrame, meta: pd.DataFrame | None = None,
                    animals: list[str] | None = None,
                    days: tuple[int, int] | None = None,
                    progress=None) -> dict[str, pd.DataFrame]:
    """Per-day node and network statistics for one trial."""
    if gbi.empty:
        return {"node_stats": pd.DataFrame(columns=NODE_COLUMNS),
                "net_stats": pd.DataFrame(columns=NET_COLUMNS)}
    say = progress or (lambda _m: None)

    animals = animals or [c for c in gbi.columns if c not in GBI_META]
    day_values = sorted(gbi["day"].unique())
    if days is not None:
        day_values = [d for d in day_values if days[0] <= d <= days[1]]

    node_rows, net_rows = [], []
    for day in day_values:
        say(f"Building network for day {day}...")
        nodes, summary = network_for_day(gbi, day, animals, meta)
        if nodes is None:
            continue
        node_rows.append(nodes)
        net_rows.append(summary)

    node_stats = (pd.concat(node_rows, ignore_index=True)
                  if node_rows else pd.DataFrame(columns=NODE_COLUMNS))
    net_stats = (pd.DataFrame(net_rows) if net_rows
                 else pd.DataFrame(columns=NET_COLUMNS))
    say(f"{len(net_stats)} daily networks")
    return {"node_stats": node_stats[NODE_COLUMNS] if len(node_stats) else node_stats,
            "net_stats": net_stats[NET_COLUMNS] if len(net_stats) else net_stats}


class SocialNetworkAnalyzer:
    """Config-driven wrapper around :func:`social_networks`."""

    def __init__(self, config=None):
        self.config = config

    def run(self, gbi: pd.DataFrame, meta=None, animals=None, progress=None):
        days = getattr(self.config, "analysis_days", None) if self.config else None
        return social_networks(gbi, meta, animals, days, progress)
