from typing import List, Any


def quick_sort(arr: List[Any]) -> List[Any]:
    """Quick sort (in-place, uses middle pivot) and returns the array."""
    def _quick(a, lo, hi):
        if lo >= hi:
            return
        pivot = a[(lo + hi) // 2]
        i, j = lo, hi
        while i <= j:
            while a[i] < pivot:
                i += 1
            while a[j] > pivot:
                j -= 1
            if i <= j:
                a[i], a[j] = a[j], a[i]
                i += 1
                j -= 1
        if lo < j:
            _quick(a, lo, j)
        if i < hi:
            _quick(a, i, hi)

    _quick(arr, 0, len(arr) - 1)
    return arr
