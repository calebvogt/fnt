"""PoE daisy-chain wiring optimiser for the UWB antenna layouts.

Each gateway is a depot with two arms (A/B); each arm is an open daisy chain of
up to ``cap_arm`` antennas, and a gateway carries at most ``cap_gw`` antennas
across both arms. Gateways never connect to each other. We minimise total
Euclidean PoE cable length with a multi-start heuristic: randomised
cheapest-append construction + per-arm 2-opt + relocate/swap local search,
keeping the best of many restarts — near-optimal for these small grids.
"""
from __future__ import annotations

import math
import random
from copy import deepcopy

from .config import Cable

# distinct, readable colours keyed by gateway order (rgba 0–1)
GATEWAY_COLORS = [
    (0.20, 0.80, 0.95, 1.0),   # cyan
    (0.97, 0.47, 0.80, 1.0),   # pink
    (0.98, 0.67, 0.23, 1.0),   # orange
    (0.51, 0.86, 0.46, 1.0),   # green
    (0.75, 0.55, 0.98, 1.0),   # violet
]


def gateway_color_map(cables):
    """Map each gateway label to a colour, by first appearance in ``cables``."""
    labels = []
    for c in cables:
        if c.gateway not in labels:
            labels.append(c.gateway)
    return {lab: GATEWAY_COLORS[i % len(GATEWAY_COLORS)]
            for i, lab in enumerate(labels)}


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _arm_len(gpos, nodes):
    """Length of the open path gateway -> nodes[0] -> … -> nodes[-1]."""
    if not nodes:
        return 0.0
    total = _dist(gpos, nodes[0][1])
    for a, b in zip(nodes, nodes[1:]):
        total += _dist(a[1], b[1])
    return total


def _two_opt(gpos, nodes):
    """In-place 2-opt on an open path with a fixed start (the gateway)."""
    improved = True
    while improved and len(nodes) > 2:
        improved = False
        for i in range(len(nodes) - 1):
            for k in range(i + 1, len(nodes)):
                a = gpos if i == 0 else nodes[i - 1][1]
                before = _dist(a, nodes[i][1]) + (
                    _dist(nodes[k][1], nodes[k + 1][1])
                    if k + 1 < len(nodes) else 0.0)
                after = _dist(a, nodes[k][1]) + (
                    _dist(nodes[i][1], nodes[k + 1][1])
                    if k + 1 < len(nodes) else 0.0)
                if after + 1e-12 < before:
                    nodes[i:k + 1] = nodes[i:k + 1][::-1]
                    improved = True


def _gcount(arms, gi):
    return sum(len(a["nodes"]) for a in arms if a["gi"] == gi)


def _total(arms):
    return sum(_arm_len(a["gpos"], a["nodes"]) for a in arms)


def _local_search(arms, cap_arm, cap_gw):
    for a in arms:
        _two_opt(a["gpos"], a["nodes"])

    def try_relocate():
        for src in arms:
            for ni in range(len(src["nodes"])):
                node = src["nodes"][ni]
                for dst in arms:
                    if dst is src or len(dst["nodes"]) >= cap_arm:
                        continue
                    if dst["gi"] != src["gi"] and _gcount(arms, dst["gi"]) >= cap_gw:
                        continue
                    st = src["nodes"][:ni] + src["nodes"][ni + 1:]
                    dt = dst["nodes"] + [node]
                    _two_opt(src["gpos"], st)
                    _two_opt(dst["gpos"], dt)
                    base = _arm_len(src["gpos"], src["nodes"]) \
                        + _arm_len(dst["gpos"], dst["nodes"])
                    new = _arm_len(src["gpos"], st) + _arm_len(dst["gpos"], dt)
                    if new + 1e-9 < base:
                        src["nodes"], dst["nodes"] = st, dt
                        return True
        return False

    def try_swap():
        for i in range(len(arms)):
            for j in range(i + 1, len(arms)):
                A, B = arms[i], arms[j]
                for ai in range(len(A["nodes"])):
                    for bi in range(len(B["nodes"])):
                        at = A["nodes"][:ai] + [B["nodes"][bi]] + A["nodes"][ai + 1:]
                        bt = B["nodes"][:bi] + [A["nodes"][ai]] + B["nodes"][bi + 1:]
                        _two_opt(A["gpos"], at)
                        _two_opt(B["gpos"], bt)
                        base = _arm_len(A["gpos"], A["nodes"]) \
                            + _arm_len(B["gpos"], B["nodes"])
                        new = _arm_len(A["gpos"], at) + _arm_len(B["gpos"], bt)
                        if new + 1e-9 < base:
                            A["nodes"], B["nodes"] = at, bt
                            return True
        return False

    for _ in range(200):
        if not try_relocate() and not try_swap():
            break


def solve_poe_wiring(gateways, leaves, z=1.8288, cap_arm=8, cap_gw=9,
                     restarts=80, seed=7):
    """Return (cables, total_length_m).

    gateways : list of (label, (x, y));  leaves : list of (label, (x, y))
    """
    rng = random.Random(seed)
    best, best_len = None, float("inf")
    for r in range(restarts):
        arms = [{"gi": gi, "glabel": gl, "arm": arm, "gpos": gp, "nodes": []}
                for gi, (gl, gp) in enumerate(gateways) for arm in ("A", "B")]
        order = list(range(len(leaves)))
        if r > 0:
            rng.shuffle(order)
        ok = True
        for li in order:
            llabel, lpos = leaves[li]
            pick = None
            for arm in arms:
                if len(arm["nodes"]) >= cap_arm or _gcount(arms, arm["gi"]) >= cap_gw:
                    continue
                last = arm["nodes"][-1][1] if arm["nodes"] else arm["gpos"]
                d = _dist(last, lpos)
                if pick is None or d < pick[0]:
                    pick = (d, arm)
            if pick is None:            # capacity exhausted — infeasible seeding
                ok = False
                break
            pick[1]["nodes"].append((llabel, lpos))
        if not ok:
            continue
        _local_search(arms, cap_arm, cap_gw)
        L = _total(arms)
        if L < best_len:
            best_len, best = L, deepcopy(arms)

    cables, total = [], 0.0
    for arm in best:
        if not arm["nodes"]:
            continue
        pts = [[arm["gpos"][0], arm["gpos"][1], z]]
        pts += [[p[0], p[1], z] for _, p in arm["nodes"]]
        cables.append(Cable(gateway=arm["glabel"], arm=arm["arm"], nodes=pts))
        total += _arm_len(arm["gpos"], arm["nodes"])
    return cables, total
