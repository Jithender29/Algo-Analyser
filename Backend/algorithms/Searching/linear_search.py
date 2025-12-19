from typing import List, Any


def linear_search(arr: List[Any], target: Any) -> int:
    """Return the first index of target in arr or -1 if not found."""
    for i, v in enumerate(arr):
        if v == target:
            return i
    return -1
