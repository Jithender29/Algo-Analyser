"""Dijkstra's shortest-path algorithm implementation.

Function signature:
    dijkstra(graph, start)

- graph: mapping node -> iterable of neighbors; each neighbor can be a node or (node, weight)
- start: starting node
- returns: dict mapping node -> shortest distance from start
"""
import heapq
from typing import Any, Dict, Iterable


def _iter_neighbors(neighs: Iterable[Any]):
    for n in neighs:
        if isinstance(n, tuple):
            yield n[0], n[1]
        else:
            yield n, 1.0


def dijkstra(graph: Dict[Any, Iterable[Any]], start: Any) -> Dict[Any, float]:
    dist = {start: 0.0}
    pq = [(0.0, start)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float("inf")):
            continue
        for v, w in _iter_neighbors(graph.get(u, [])):
            nd = d + float(w)
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))

    return dist
