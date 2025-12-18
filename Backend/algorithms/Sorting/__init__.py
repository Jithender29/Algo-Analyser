"""
Sorting algorithms package.

Exposes a SORTING_ALGOS mapping used by run_from_configs.py.
"""

from .bubble_sort import bubble_sort
from .insertion_sort import insertion_sort
from .counting_sort import counting_sort
from .radix_sort import radix_sort
from .bucket_sort import bucket_sort

SORTING_ALGOS = {
    "bubble": bubble_sort,
    "insertion": insertion_sort,
    "counting": counting_sort,
    "radix": radix_sort,
    "bucket": bucket_sort,
}


