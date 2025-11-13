"""
Retrieve package - DEPRECATED

This module has been moved to 'vectordb'.
This file provides backward compatibility for existing code.

Please update your imports:
    OLD: from retrieve import VectorDB
    NEW: from vectordb import VectorDB
"""

import warnings
from vectordb import VectorDB

warnings.warn(
    "The 'retrieve' module is deprecated. Please use 'vectordb' instead: "
    "from vectordb import VectorDB",
    DeprecationWarning,
    stacklevel=2
)

__all__ = ['VectorDB']
