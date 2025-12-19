"""Graph algorithms package.

Exposes a GRAPH_ALGOS mapping for potential benchmark use.
"""

from .bfs import bfs
from .dfs import dfs
from .dijkstra import dijkstra

GRAPH_ALGOS = {
    "bfs": bfs,
    "dfs": dfs,
    "dijkstra": dijkstra,
}
