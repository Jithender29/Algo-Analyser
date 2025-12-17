"""
Sorting algorithms package.

Exposes a SORTING_ALGOS mapping used by run_from_configs.py.
"""

from .bubble_sort import bubble_sort
from .insertion_sort import insertion_sort

SORTING_ALGOS = {
    "bubble": bubble_sort,
    "insertion": insertion_sort,
}


