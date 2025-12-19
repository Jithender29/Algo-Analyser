"""Searching algorithms package.

Exposes a SEARCHING_ALGOS mapping for potential benchmark use.
"""

from .linear_search import linear_search
from .binary_search import binary_search

SEARCHING_ALGOS = {
    "linear": linear_search,
    "binary": binary_search,
}
