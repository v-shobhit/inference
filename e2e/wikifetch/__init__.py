"""
Wikipedia article fetching module.
Handles downloading and extraction of Wikipedia articles.
"""

from .extractor import WikipediaExtractor
from .downloader import WikipediaDownloader
from .utils import extract_wikipedia_urls

__all__ = [
    'WikipediaExtractor',
    'WikipediaDownloader',
    'extract_wikipedia_urls'
]

