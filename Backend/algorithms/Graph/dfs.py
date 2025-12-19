"""Depth-First Search (DFS) implementation (iterative).

Function signature:
    dfs(graph, start)

- graph: mapping from node -> iterable of neighbors. Neighbors may be `node` or `(node, weight)` tuples.
- start: starting node
- returns: list of nodes in DFS visitation order (preorder)
"""
from typing import Any, Dict, Iterable, List


def _normalize_neighbors(neighs: Iterable[Any]):
    for n in neighs:
        yield n[0] if isinstance(n, tuple) and len(n) >= 1 else n


def dfs(graph: Dict[Any, Iterable[Any]], start: Any) -> List[Any]:
    visited = set()
    stack = [start]
    order: List[Any] = []

    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        # push neighbors in reverse so left-most neighbor is visited first
        neighs = list(_normalize_neighbors(graph.get(node, [])))
        for nbr in reversed(neighs):
            if nbr not in visited:
                stack.append(nbr)

    return order
