from typing import List, Any


def bucket_sort(arr: List[float]) -> List[float]:
    """Bucket sort supporting numeric data (ints or floats).

    - Distributes elements into `n` buckets (n = len(arr)), sorts each bucket with Python's
      built-in `sorted()` and concatenates results.
    - Works for arbitrary numeric ranges (handles negative values).
    - Raises ValueError for non-numeric elements.
    """
    if not arr:
        return arr

    # Validate numeric types
    try:
        min_val = min(arr)
        max_val = max(arr)
    except TypeError as exc:
        raise ValueError("bucket_sort only supports numeric values") from exc

    n = len(arr)
    if n == 1 or min_val == max_val:
        return list(arr)

    buckets: List[List[float]] = [[] for _ in range(n)]
    range_span = max_val - min_val

    for x in arr:
        # scale to bucket index in [0, n-1]
        idx = int((x - min_val) / range_span * (n - 1))
        buckets[idx].append(x)

    out: List[float] = []
    for b in buckets:
        if b:
            out.extend(sorted(b))
    return out
