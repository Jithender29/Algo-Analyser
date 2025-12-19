"""
Top-level Algorithms package.

Subpackages:
- Sorting: implementations of sorting algorithms.
- Searching: implementations of basic search algorithms.
"""

"""
Algorithm implementations grouped by type.

Currently exposed:
- sorting: bubble_sort, insertion_sort, selection, merge, quick, heap
- searching: linear, binary
"""

from .Searching import SEARCHING_ALGOS  # type: ignore
from .Sorting import SORTING_ALGOS  # type: ignore
from .Graph import GRAPH_ALGOS  # type: ignore


