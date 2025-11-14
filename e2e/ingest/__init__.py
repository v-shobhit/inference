"""
Data ingestion module for creating and managing vector stores.
Handles passage ingestion and vector database creation.
"""

from .store import VectorDB
from .create import create_vector_store_from_passages

__all__ = ['VectorDB', 'create_vector_store_from_passages']

