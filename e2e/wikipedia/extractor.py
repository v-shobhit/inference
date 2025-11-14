"""
Wikipedia content extraction via Wikipedia API.
"""

import re
import time
import urllib.parse
from typing import Optional, Dict, Any
import requests


class WikipediaExtractor:
    """
    Extract clean Wikipedia article content using the Wikipedia API.
    Produces clean text similar to WikiExtractor output.
    """
    
    def __init__(self, language: str = 'en', timeout: int = 30, max_retries: int = 3):
        """
        Initialize the Wikipedia extractor.
        
        Args:
            language: Wikipedia language code (default: 'en')
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.language = language
        self.timeout = timeout
        self.max_retries = max_retries
        self.api_url = f"https://{language}.wikipedia.org/w/api.php"
        
        # Configure connection pooling to avoid overwhelming the network
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=10,
            max_retries=3
        )
        self.session.mount('https://', adapter)
        self.session.mount('http://', adapter)
    
    def extract_article_title_from_url(self, url: str) -> Optional[str]:
        """
        Extract article title from Wikipedia URL.
        
        Args:
            url: Wikipedia URL (e.g., https://en.wikipedia.org/wiki/Article_Title)
        
        Returns:
            Decoded article title or None if URL format is invalid
        """
        # Handle URLs like: https://en.wikipedia.org/wiki/Article_Title
        pattern = r'https?://(?:www\.)?([a-z]{2})\.wikipedia\.org/wiki/(.+)'
        match = re.match(pattern, url)
        if match:
            # Decode URL encoding
            title = urllib.parse.unquote(match.group(2))
            return title
        return None
    
    def get_article_content(self, title: str) -> Optional[Dict[str, Any]]:
        """
        Fetch clean article content from Wikipedia API.
        
        Args:
            title: Article title
        
        Returns:
            Dictionary with article metadata and clean text, or None if failed
            {
                'title': str,
                'pageid': str,
                'url': str,
                'text': str,  # Clean text with headings
                'length': int
            }
        """
        # Parameters for Wikipedia API
        params = {
            'action': 'query',
            'format': 'json',
            'titles': title,
            'prop': 'extracts|info',
            'explaintext': True,  # Get plain text, not HTML
            'exsectionformat': 'plain',
            'inprop': 'url',
            'redirects': 1,  # Follow redirects
        }
        
        # Retry logic with exponential backoff
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(
                    self.api_url,
                    params=params,
                    timeout=self.timeout,
                    headers={'User-Agent': 'FRAMES-RAG-Benchmark/1.0'}
                )
                response.raise_for_status()
                
                data = response.json()
                
                # Extract page data
                pages = data.get('query', {}).get('pages', {})
                if not pages:
                    return None
                
                # Get the first (and should be only) page
                page_id = list(pages.keys())[0]
                
                # Check if page exists
                if page_id == '-1':
                    return None
                
                page = pages[page_id]
                
                # Extract content
                extract = page.get('extract', '')
                if not extract or len(extract.strip()) < 100:
                    return None
                
                # Clean the extracted text
                clean_text = self.clean_wikipedia_text(extract)
                
                # Add article title as the main heading at the beginning
                article_title = page.get('title', title)
                full_text = f"# {article_title}\n\n{clean_text}"
                
                return {
                    'title': article_title,
                    'pageid': page_id,
                    'url': page.get('fullurl', ''),
                    'text': full_text,
                    'length': len(full_text),
                }
                
            except (requests.exceptions.Timeout, 
                    requests.exceptions.ConnectionError,
                    requests.exceptions.RequestException):
                if attempt < self.max_retries - 1:
                    # Exponential backoff: wait 1s, 2s, 4s...
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    continue
                else:
                    # Final attempt failed
                    return None
            except Exception:
                # Non-retryable error
                return None
        
        return None
    
    def download_from_url(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Download and extract clean content from a Wikipedia URL.
        
        Args:
            url: Wikipedia URL
        
        Returns:
            Content dictionary with 'source_url' field added, or None if failed
        """
        title = self.extract_article_title_from_url(url)
        if not title:
            return None
        
        content = self.get_article_content(title)
        if content:
            content['source_url'] = url
        
        return content
    
    def clean_wikipedia_text(self, text: str) -> str:
        """
        Clean Wikipedia text to remove common artifacts.
        Similar to WikiExtractor's cleaning approach.
        
        Args:
            text: Raw Wikipedia text
        
        Returns:
            Cleaned text
        """
        # Remove citation needed markers
        text = re.sub(r'\[citation needed\]', '', text, flags=re.IGNORECASE)
        
        # Remove edit markers
        text = re.sub(r'\[edit\]', '', text, flags=re.IGNORECASE)
        
        # Remove coordinate information
        text = re.sub(r'Coordinates:\s*\d+°[^\n]*\n', '', text)
        
        # Remove "See also", "References", "External links" sections and everything after
        # These sections typically don't contain useful information for RAG
        section_patterns = [
            r'\n+See also\s*\n.*$',
            r'\n+References\s*\n.*$',
            r'\n+External links\s*\n.*$',
            r'\n+Further reading\s*\n.*$',
            r'\n+Notes\s*\n.*$',
            r'\n+Bibliography\s*\n.*$',
            r'\n+Sources\s*\n.*$',
            r'\n+Cited literature\s*\n.*$',
        ]
        for pattern in section_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove stub messages
        text = re.sub(
            r'This\s+(?:article|section)\s+(?:about|on|related to).*?is a stub.*?$',
            '',
            text,
            flags=re.IGNORECASE | re.MULTILINE
        )
        
        # Remove "You can help Wikipedia" messages
        text = re.sub(
            r'You can help Wikipedia by expanding it\.?',
            '',
            text,
            flags=re.IGNORECASE
        )
        
        # Format headings with delimiters for better readability
        text = self.format_headings(text)
        
        # Clean up whitespace
        text = re.sub(r'\n\n+', '\n\n', text)  # Multiple newlines to double
        text = re.sub(r' +', ' ', text)  # Multiple spaces to single
        text = text.strip()
        
        return text
    
    def format_headings(self, text: str) -> str:
        """
        Add delimiters to section headings for better readability.
        Detects likely headings (short lines, title case, no ending punctuation).
        
        Args:
            text: Text to format
        
        Returns:
            Text with formatted headings
        """
        lines = text.split('\n')
        formatted_lines = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Skip empty lines
            if not stripped:
                formatted_lines.append(line)
                continue
            
            # Detect heading characteristics:
            # - Short line (< 80 chars)
            # - Doesn't end with sentence punctuation
            # - Not all lowercase
            # - Previous line is empty (or first line)
            # - Next line is empty or exists
            is_short = len(stripped) < 80
            no_punctuation = not stripped.endswith(('.', ',', ';', ':', '!', '?', '"', "'", ')'))
            not_all_lower = not stripped.islower()
            prev_empty = (i == 0) or (i > 0 and not lines[i-1].strip())
            next_empty = (i == len(lines) - 1) or (i < len(lines) - 1 and not lines[i+1].strip())
            
            # If it looks like a heading, add delimiter
            if is_short and no_punctuation and not_all_lower and (prev_empty or next_empty):
                # Check if it's not a list item or sentence fragment
                if not stripped.startswith(('•', '-', '*', '1.', '2.', '3.', '(')) and ' ' in stripped:
                    formatted_lines.append(f"\n### {stripped}\n")
                else:
                    formatted_lines.append(line)
            else:
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)

