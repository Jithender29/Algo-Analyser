from typing import List, Any


def counting_sort(arr: List[int]) -> List[int]:
    """Counting sort that supports negative integers.

    - Works with integers only and returns a new sorted list.
    - Raises ValueError on non-integer elements.
    """
    if not arr:
        return arr

    # Validate and find min/max
    try:
        min_val = min(arr)
        max_val = max(arr)
    except TypeError as exc:
        raise ValueError("counting_sort only supports integers") from exc

    if not all(isinstance(x, int) for x in arr):
        raise ValueError("counting_sort only supports integers")

    offset = -min_val if min_val < 0 else 0
    range_len = max_val - min_val + 1
    counts = [0] * range_len

    for x in arr:
        counts[x + offset] += 1

    out: List[int] = []
    for i, c in enumerate(counts):
        out.extend([i - offset] * c)

    return out
