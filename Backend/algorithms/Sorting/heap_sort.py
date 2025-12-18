from typing import List, Any
import heapq


def heap_sort(arr: List[Any]) -> List[Any]:
    """Heap sort implemented using Python's heapq. Returns a new sorted list."""
    if not arr:
        return arr
    h = list(arr)
    heapq.heapify(h)
    return [heapq.heappop(h) for _ in range(len(h))]
