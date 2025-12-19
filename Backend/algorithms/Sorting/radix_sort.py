from typing import List, Any


def _radix_sort_positive(arr: List[int]) -> List[int]:
    """Helper: radix sort for non-negative integers (LSD base 10)."""
    if not arr:
        return arr
    max_val = max(arr)
    exp = 1
    out = list(arr)
    while max_val // exp > 0:
        buckets = [[] for _ in range(10)]
        for num in out:
            buckets[(num // exp) % 10].append(num)
        out = [n for bucket in buckets for n in bucket]
        exp *= 10
    return out


def radix_sort(arr: List[int]) -> List[int]:
    """Radix sort supporting integers (handles negatives).

    - Splits negatives and positives, uses LSD radix on absolute values for positives.
    - Raises ValueError for non-integer elements.
    """
    if not arr:
        return arr

    if not all(isinstance(x, int) for x in arr):
        raise ValueError("radix_sort only supports integers")

    negatives = [abs(x) for x in arr if x < 0]
    positives = [x for x in arr if x >= 0]

    pos_sorted = _radix_sort_positive(positives)
    if negatives:
        neg_sorted = _radix_sort_positive(negatives)
        # Convert back to negatives and reverse to keep proper ordering
        neg_sorted = [-x for x in reversed(neg_sorted)]
    else:
        neg_sorted = []

    return neg_sorted + pos_sorted
