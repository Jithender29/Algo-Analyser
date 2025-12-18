from typing import List, Any


def binary_search(arr: List[Any], target: Any) -> int:
    """Iterative binary search. Assumes arr is sorted. Returns index or -1."""
    lo = 0
    hi = len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        if arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
