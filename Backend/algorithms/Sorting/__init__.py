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
from .counting_sort import counting_sort
from .radix_sort import radix_sort
from .bucket_sort import bucket_sort

SORTING_ALGOS = {
    "bubble": bubble_sort,
    "insertion": insertion_sort,
    "selection": selection_sort,
    "merge": merge_sort,
    "quick": quick_sort,
    "heap": heap_sort,
    "counting": counting_sort,
    "radix": radix_sort,
    "bucket": bucket_sort,
}


