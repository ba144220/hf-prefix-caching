"""
HF Prefix Caching - Efficient prefix caching for HuggingFace models.

This package provides efficient prefix caching mechanisms for HuggingFace transformers,
allowing for faster inference by caching and reusing computations from common prefixes
in text generation tasks.
"""

from .prefix_cache import PrefixCache
from .types import CacheHash, CacheBlock

__version__ = "0.1.0"
__author__ = "Yuchi Hsu"
__email__ = "yuchihsu@stanford.edu"

__all__ = ["PrefixCache", "CacheHash", "CacheBlock"] 