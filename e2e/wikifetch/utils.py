"""
Utility functions for Wikipedia URL handling.
"""

from typing import List, Dict, Any, Optional, Tuple


def extract_wikipedia_urls(
    prompts: List[Dict[str, Any]],
    max_urls: Optional[int] = None,
    filter_non_articles: bool = True
) -> Tuple[List[str], List[str]]:
    """
    Extract unique Wikipedia URLs from loaded prompts.
    
    Works with prompts loaded by retrieval.PromptLoader.
    
    Args:
        prompts: List of prompt dicts with 'wiki_links' field
        max_urls: Maximum URLs to return (None = all)
        filter_non_articles: Filter out Category/File/Template pages
    
    Returns:
        Tuple of (valid_urls, excluded_urls)
    
    Example:
        >>> from retrieval import PromptLoader
        >>> prompts = PromptLoader.load(tsv_path="data.tsv")
        >>> urls, excluded = extract_wikipedia_urls(prompts)
        >>> print(f"Found {len(urls)} valid URLs")
    """
    urls = set()
    excluded_urls = []
    
    for prompt in prompts:
        wiki_links = prompt.get('wiki_links', [])
        
        # wiki_links is already normalized to a list by PromptLoader
        for url in wiki_links:
            # Ensure it's a Wikipedia URL
            if not isinstance(url, str) or not url.startswith('https://en.wikipedia.org/wiki/'):
                continue
            
            # Filter non-article pages if requested
            if filter_non_articles and _is_non_article_page(url):
                excluded_urls.append(url)
                continue
            
            urls.add(url)
    
    # Sort for consistency
    sorted_urls = sorted(list(urls))
    
    # Limit if requested
    if max_urls is not None:
        sorted_urls = sorted_urls[:max_urls]
    
    return sorted_urls, excluded_urls


def _is_non_article_page(url: str) -> bool:
    """
    Check if URL is a non-article page.
    
    Filters out:
    - Category pages (Category:...)
    - File pages (File:...)
    - Template pages (Template:...)
    
    Args:
        url: Wikipedia URL to check
    
    Returns:
        True if non-article page, False if regular article
    """
    non_article_patterns = [
        '/wiki/Category:',
        '/wiki/File:',
        '/wiki/Template:',
        '/wiki/Wikipedia:',
        '/wiki/Help:',
        '/wiki/Portal:',
        '/wiki/Special:',
    ]
    
    return any(pattern in url for pattern in non_article_patterns)

