"""Breadth-First Search (BFS) implementation.

Function signature:
    bfs(graph, start)

- graph: mapping from node -> iterable of neighbors. Neighbors may be `node` or `(node, weight)` tuples.
- start: starting node
- returns: list of nodes in BFS visitation order
"""
from collections import deque
from typing import Any, Dict, Iterable, List


def _normalize_neighbors(neighs: Iterable[Any]):
    for n in neighs:
        yield n[0] if isinstance(n, tuple) and len(n) >= 1 else n


def bfs(graph: Dict[Any, Iterable[Any]], start: Any) -> List[Any]:
    visited = set()
    q = deque([start])
    order: List[Any] = []

    while q:
        node = q.popleft()
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        for nbr in _normalize_neighbors(graph.get(node, [])):
            if nbr not in visited:
                q.append(nbr)

    return order
