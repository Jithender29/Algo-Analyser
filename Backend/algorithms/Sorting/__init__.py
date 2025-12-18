"""
Sorting algorithms package.

Exposes a SORTING_ALGOS mapping used by run_from_configs.py.
"""

from .bubble_sort import bubble_sort
from .insertion_sort import insertion_sort
from .selection_sort import selection_sort
from .merge_sort import merge_sort
from .quick_sort import quick_sort
from .heap_sort import heap_sort

SORTING_ALGOS = {
    "bubble": bubble_sort,
    "insertion": insertion_sort,
    "selection": selection_sort,
    "merge": merge_sort,
    "quick": quick_sort,
    "heap": heap_sort,
}


