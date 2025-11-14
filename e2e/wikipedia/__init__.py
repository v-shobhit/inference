"""
Wikipedia article downloading and processing modules.
"""

from .extractor import WikipediaExtractor
from .chunker import TextChunker
from .passages import PassageBuilder
from .downloader import WikipediaDownloader
from .utils import extract_wikipedia_urls

__all__ = [
    'WikipediaExtractor',
    'TextChunker',
    'PassageBuilder',
    'WikipediaDownloader',
    'extract_wikipedia_urls'
]

