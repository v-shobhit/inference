"""
Text chunking module for creating passages from articles.
Provides semantic and simple chunking strategies.
"""

from .chunker import TextChunker
from .passages import PassageBuilder

__all__ = [
    'TextChunker',
    'PassageBuilder',
]

