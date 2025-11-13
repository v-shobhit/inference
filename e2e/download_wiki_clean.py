#!/usr/bin/env python3
"""
Clean Wikipedia Content Downloader for FRAMES
Uses Wikipedia API to fetch clean article content, avoiding HTML/PDF processing issues.
Inspired by WikiExtractor's approach to produce clean, RAG-ready text.

Requirements:
pip install wikipedia-api mwparserfromhell requests tqdm datasets pandas

Usage:
# Download clean Wikipedia articles from FRAMES dataset
python download_wiki_clean.py --output_dir wiki_clean_articles

# Process with chunking
python download_wiki_clean.py --output_dir wiki_clean_articles --create_chunks --chunk_size 512
"""

import argparse
import json
import re
import os
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Any, Tuple, Optional
import ast

import requests
from tqdm import tqdm
import pandas as pd
from datasets import load_dataset

# For semantic chunking
import spacy
from sentence_transformers import SentenceTransformer, util
import numpy as np

class WikipediaCleanExtractor:
    """
    Extract clean Wikipedia article content using the Wikipedia API.
    Produces clean text similar to WikiExtractor output.
    """
    
    def __init__(self, language='en'):
        self.language = language
        self.api_url = f"https://{language}.wikipedia.org/w/api.php"
        self.session = requests.Session()
        
        # Configure connection pooling to avoid overwhelming the network
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=10,
            max_retries=3
        )
        self.session.mount('https://', adapter)
        self.session.mount('http://', adapter)
        
    def extract_article_title_from_url(self, url: str) -> Optional[str]:
        """Extract article title from Wikipedia URL."""
        # Handle URLs like: https://en.wikipedia.org/wiki/Article_Title
        pattern = r'https?://(?:www\.)?([a-z]{2})\.wikipedia\.org/wiki/(.+)'
        match = re.match(pattern, url)
        if match:
            # Decode URL encoding
            import urllib.parse
            title = urllib.parse.unquote(match.group(2))
            return title
        return None
    
    def get_article_content(self, title: str, timeout: int = 30, max_retries: int = 3) -> Optional[Dict[str, Any]]:
        """
        Fetch clean article content from Wikipedia API.
        Returns a dictionary with article metadata and clean text.
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
        for attempt in range(max_retries):
            try:
                response = self.session.get(
                    self.api_url,
                    params=params,
                    timeout=timeout,
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
                    requests.exceptions.RequestException) as e:
                if attempt < max_retries - 1:
                    # Exponential backoff: wait 1s, 2s, 4s...
                    import time
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    continue
                else:
                    # Final attempt failed
                    return None
            except Exception as e:
                # Non-retryable error
                return None
        
        return None
    
    def format_headings(self, text: str) -> str:
        """
        Add delimiters to section headings for better readability.
        Detects likely headings (short lines, title case, no ending punctuation).
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
    
    def clean_wikipedia_text(self, text: str) -> str:
        """
        Clean Wikipedia text to remove common artifacts.
        Similar to WikiExtractor's cleaning approach.
        """
        # Remove citation needed markers
        text = re.sub(r'\[citation needed\]', '', text, flags=re.IGNORECASE)
        
        # Remove edit markers
        text = re.sub(r'\[edit\]', '', text, flags=re.IGNORECASE)
        
        # Remove coordinate information
        text = re.sub(r'Coordinates:\s*\d+°[^\n]*\n', '', text)
        
        # Remove "See also", "References", "External links" sections and everything after
        # These sections typically don't contain useful information for RAG
        # Note: API returns plain text, so sections appear as plain headings, not wiki markup
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
    
    def download_from_url(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Download and extract clean content from a Wikipedia URL.
        """
        title = self.extract_article_title_from_url(url)
        if not title:
            return None
        
        content = self.get_article_content(title)
        if content:
            content['source_url'] = url
        
        return content


def process_single_url(args_tuple) -> Tuple[bool, str, Optional[Dict], str]:
    """
    Process a single Wikipedia URL.
    Designed for multiprocessing.
    
    Returns: (success, url, content_dict, error_message)
    """
    url, output_dir = args_tuple
    
    # Create extractor for this process (heading formatting always enabled)
    extractor = WikipediaCleanExtractor()
    
    # Create safe filename from URL
    title = extractor.extract_article_title_from_url(url)
    if not title:
        return False, url, None, "Could not extract title from URL"
    
    # Create safe filename
    safe_filename = re.sub(r'[^\w\s-]', '_', title)
    safe_filename = re.sub(r'[-\s]+', '_', safe_filename)
    if len(safe_filename) > 200:
        safe_filename = safe_filename[:200]
    
    output_path = output_dir / f"{safe_filename}.txt"
    json_path = output_dir / f"{safe_filename}.json"
    
    # Skip if file already exists
    if output_path.exists() and json_path.exists():
        return True, url, None, "Already exists"
    
    # Download content
    content = extractor.download_from_url(url)
    
    if not content:
        return False, url, None, "Failed to download or extract content"
    
    # Check if content is substantial
    if len(content['text']) < 100:
        return False, url, None, f"Content too short ({len(content['text'])} chars)"
    
    try:
        # Save as plain text
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content['text'])
        
        # Save metadata as JSON
        metadata = {
            'title': content['title'],
            'pageid': content['pageid'],
            'url': content['url'],
            'source_url': content['source_url'],
            'length': content['length'],
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return True, url, content, "Success"
        
    except Exception as e:
        # Clean up partial files
        if output_path.exists():
            output_path.unlink()
        if json_path.exists():
            json_path.unlink()
        return False, url, None, f"Error saving: {str(e)}"


def semantic_chunking(
    text: str,
    nlp,
    embedding_model,
    threshold: float = 0.75,
    max_chunk_size: int = 400,
    verbose: bool = False
) -> List[str]:
    """
    Perform semantic chunking based on sentence similarity.
    Groups semantically similar sentences together.
    
    Args:
        text: Input text to chunk
        nlp: spaCy model for sentence splitting
        embedding_model: SentenceTransformer model for embeddings
        threshold: Similarity threshold (0-1) for grouping sentences
        max_chunk_size: Maximum words per chunk
        verbose: Print progress
    
    Returns:
        List of semantically coherent text chunks
    """
    if verbose:
        print("  Performing semantic chunking...")
    
    # Split into sentences using spaCy
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 15]
    
    if not sentences:
        return [text] if text else []
    
    # Skip very short texts
    if len(sentences) <= 2:
        return [text]
    
    if verbose:
        print(f"  Processing {len(sentences)} sentences...")
    
    # Encode sentences to embeddings
    sentence_embeddings = embedding_model.encode(sentences, show_progress_bar=False)
    
    chunks = []
    current_chunk = [sentences[0]]
    current_embedding = sentence_embeddings[0]
    
    for i in range(1, len(sentences)):
        sentence = sentences[i]
        sentence_embedding = sentence_embeddings[i]
        
        # Calculate similarity with current chunk
        similarity = util.cos_sim(current_embedding, sentence_embedding).item()
        
        # Check if we should add to current chunk
        potential_chunk_text = ' '.join(current_chunk + [sentence])
        potential_word_count = len(potential_chunk_text.split())
        
        # Group if similar AND doesn't exceed size
        if similarity >= threshold and potential_word_count < max_chunk_size:
            current_chunk.append(sentence)
            # Update chunk embedding (running average)
            current_embedding = (current_embedding + sentence_embedding) / 2
        else:
            # Finalize current chunk and start new one
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_embedding = sentence_embedding
    
    # Add the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    if verbose:
        print(f"  Created {len(chunks)} semantic chunks")
    
    return chunks


def add_chunk_overlap(
    chunks: List[str],
    overlap_percentage: float = 0.15,
    max_chunk_size: int = 400
) -> List[str]:
    """
    Add overlap between chunks to preserve context.
    
    Args:
        chunks: List of text chunks
        overlap_percentage: Percentage of previous chunk to overlap
        max_chunk_size: Maximum words per chunk after overlap
    
    Returns:
        List of chunks with overlap added
    """
    if len(chunks) <= 1:
        return chunks
    
    overlapped_chunks = []
    
    for i, chunk in enumerate(chunks):
        if i == 0:
            # First chunk - check if needs truncation
            chunk_words = chunk.split()
            if len(chunk_words) > max_chunk_size:
                overlapped_chunks.append(' '.join(chunk_words[:max_chunk_size]))
            else:
                overlapped_chunks.append(chunk)
        else:
            # Calculate overlap from previous chunk
            prev_words = chunks[i - 1].split()
            overlap_size = max(1, int(len(prev_words) * overlap_percentage))
            overlap_text = ' '.join(prev_words[-overlap_size:])
            
            # Combine overlap with current chunk
            combined_chunk = overlap_text + ' ' + chunk
            combined_words = combined_chunk.split()
            
            # Enforce size limit
            if len(combined_words) > max_chunk_size:
                overlapped_chunks.append(' '.join(combined_words[:max_chunk_size]))
            else:
                overlapped_chunks.append(combined_chunk)
    
    return overlapped_chunks


def merge_small_chunks(
    chunks: List[str],
    min_chunk_size: int = 50,
    max_chunk_size: int = 400
) -> List[str]:
    """
    Merge chunks smaller than min_chunk_size with adjacent chunks.
    
    Args:
        chunks: List of text chunks
        min_chunk_size: Minimum words per chunk
        max_chunk_size: Maximum words per chunk
    
    Returns:
        List of chunks with small ones merged
    """
    if not chunks:
        return chunks
    
    merged_chunks = []
    i = 0
    
    while i < len(chunks):
        current_chunk = chunks[i]
        current_words = len(current_chunk.split())
        
        # If current chunk is too small, try to merge it
        if current_words < min_chunk_size:
            # Try to merge with next chunk first
            if i + 1 < len(chunks):
                next_chunk = chunks[i + 1]
                combined = current_chunk + ' ' + next_chunk
                combined_words = len(combined.split())
                
                if combined_words <= max_chunk_size:
                    merged_chunks.append(combined)
                    i += 2  # Skip both chunks
                    continue
            
            # If can't merge with next, try to merge with previous
            if merged_chunks:
                prev_chunk = merged_chunks[-1]
                combined = prev_chunk + ' ' + current_chunk
                combined_words = len(combined.split())
                
                if combined_words <= max_chunk_size:
                    merged_chunks[-1] = combined
                    i += 1
                    continue
            
            # If can't merge with either, keep as is
            merged_chunks.append(current_chunk)
        else:
            # Chunk is already good size
            merged_chunks.append(current_chunk)
        
        i += 1
    
    # Filter out very small chunks (< 10 words) as last resort
    final_chunks = [chunk for chunk in merged_chunks if len(chunk.split()) >= 10]
    
    return final_chunks


def split_by_sections(text: str) -> List[Tuple[str, str]]:
    """
    Split text by section headings (# and ###).
    
    Args:
        text: Input text with markdown headings
    
    Returns:
        List of (section_heading, section_content) tuples
    """
    sections = []
    lines = text.split('\n')
    current_heading = ""
    current_content = []
    
    for line in lines:
        # Check if it's a heading
        if line.startswith('# ') or line.startswith('### '):
            # Save previous section if it exists
            if current_heading or current_content:
                content = '\n'.join(current_content).strip()
                if content:  # Only add if there's actual content
                    sections.append((current_heading, content))
            
            # Start new section
            current_heading = line
            current_content = []
        else:
            current_content.append(line)
    
    # Add the last section
    if current_heading or current_content:
        content = '\n'.join(current_content).strip()
        if content:
            sections.append((current_heading, content))
    
    return sections


def create_chunks_from_text(
    text: str,
    max_length: int = 512,
    overlap: int = 50,
    use_semantic: bool = True,
    threshold: float = 0.75,
    verbose: bool = False
) -> List[Dict[str, str]]:
    """
    Split text into chunks respecting section boundaries.
    Chunks within sections using semantic chunking if available.
    
    Args:
        text: Input text to chunk
        max_length: Maximum characters per chunk (converted to ~words/2.5 for semantic)
        overlap: Overlap in characters (converted to percentage for semantic)
        use_semantic: Whether to use semantic chunking (requires spacy & transformers)
        threshold: Semantic similarity threshold (0-1)
        verbose: Print progress
    
    Returns:
        List of dicts with 'text' and 'section' keys
    """
    if not text:
        return []
    
    # Split text by sections first
    sections = split_by_sections(text)
    
    if verbose:
        print(f"  Split into {len(sections)} sections")
    
    # Convert character-based sizes to word-based for semantic chunking
    max_words = int(max_length / 2.5)  # Rough conversion: ~2.5 chars per word
    overlap_pct = min(0.3, overlap / max_length)  # Convert to percentage
    min_words = max(10, int(max_words * 0.15))  # Min 15% of max or 10 words
    
    all_chunks = []
    
    # Chunk each section separately
    for heading, content in sections:
        if not content.strip():
            continue
        
        section_chunks = []
        
        # Use semantic chunking if available and requested
        if use_semantic:
            try:
                # Load models (cached after first call)
                if not hasattr(create_chunks_from_text, 'nlp'):
                    if verbose:
                        print("  Loading spaCy and embedding models...")
                    create_chunks_from_text.nlp = spacy.load("en_core_web_sm")
                    create_chunks_from_text.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                
                nlp = create_chunks_from_text.nlp
                embedding_model = create_chunks_from_text.embedding_model
                
                # Perform semantic chunking on this section
                section_chunks = semantic_chunking(
                    content, nlp, embedding_model,
                    threshold=threshold,
                    max_chunk_size=max_words,
                    verbose=False
                )
                
                # Add overlap within section
                section_chunks = add_chunk_overlap(section_chunks, overlap_percentage=overlap_pct, max_chunk_size=max_words)
                
                # Merge small chunks within section
                section_chunks = merge_small_chunks(section_chunks, min_chunk_size=min_words, max_chunk_size=max_words)
                
            except Exception as e:
                if verbose:
                    print(f"  Semantic chunking failed for section ({e}), falling back to simple chunking")
                section_chunks = []
        
        # Fallback to simple sentence-based chunking for this section
        if not section_chunks:
            if len(content) <= max_length:
                section_chunks = [content]
            else:
                section_chunks = []
                sentences = re.split(r'(?<=[.!?])\s+', content)
                
                current_chunk = []
                current_length = 0
                
                for sentence in sentences:
                    sentence_len = len(sentence)
                    
                    if current_length + sentence_len > max_length and current_chunk:
                        section_chunks.append(' '.join(current_chunk))
                        
                        # Add overlap
                        overlap_sentences = []
                        overlap_length = 0
                        for s in reversed(current_chunk):
                            if overlap_length + len(s) <= overlap:
                                overlap_sentences.insert(0, s)
                                overlap_length += len(s)
                            else:
                                break
                        
                        current_chunk = overlap_sentences
                        current_length = overlap_length
                    
                    current_chunk.append(sentence)
                    current_length += sentence_len
                
                if current_chunk:
                    section_chunks.append(' '.join(current_chunk))
        
        # Create chunk dicts with section information
        for chunk_text in section_chunks:
            chunk_dict = {
                'text': f"{heading}\n\n{chunk_text}" if heading else chunk_text,
                'section': heading.strip() if heading else "Introduction"
            }
            all_chunks.append(chunk_dict)
    
    return all_chunks


def chunk_article_worker(args_tuple: Tuple) -> Tuple[Path, List[Dict[str, str]], Dict]:
    """
    Worker function for parallel article chunking.
    Loads models once per worker (cached in worker process).
    
    Args:
        args_tuple: (txt_file, max_length, overlap, use_semantic, threshold)
    
    Returns:
        Tuple of (txt_file, chunks_with_sections, metadata)
        where chunks_with_sections is a list of dicts with 'text' and 'section' keys
    """
    txt_file, max_length, overlap, use_semantic, threshold = args_tuple
    
    try:
        # Read article text
        with open(txt_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Read metadata
        json_file = txt_file.with_suffix('.json')
        metadata = {}
        if json_file.exists():
            with open(json_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        
        # Create chunks
        chunks = create_chunks_from_text(
            text,
            max_length=max_length,
            overlap=overlap,
            use_semantic=use_semantic,
            threshold=threshold,
            verbose=False
        )
        
        return (txt_file, chunks, metadata)
        
    except Exception as e:
        print(f"\nError processing {txt_file.name}: {e}")
        return (txt_file, [], {})


def create_passages_from_articles(
    articles_dir: Path,
    output_json: Path,
    max_length: int = 512,
    overlap: int = 50,
    use_semantic: bool = True,
    threshold: float = 0.75,
    workers: int = 1
) -> Dict[str, Any]:
    """
    Create passages JSON from downloaded articles using semantic chunking.
    
    Args:
        articles_dir: Directory containing article .txt files
        output_json: Output path for passages.json
        max_length: Maximum characters per chunk
        overlap: Overlap in characters
        use_semantic: Whether to use semantic chunking
        threshold: Semantic similarity threshold (0-1)
        workers: Number of parallel workers for chunking (1=sequential)
    
    Returns:
        Dictionary with statistics
    """
    articles_dir = Path(articles_dir)
    
    # Find all text files
    txt_files = list(articles_dir.glob("*.txt"))
    
    if not txt_files:
        print(f"No text files found in {articles_dir}")
        return {"error": "No text files found"}
    
    print(f"\nCreating passages from {len(txt_files)} articles...")
    print(f"Chunk size: {max_length} characters, Overlap: {overlap} characters")
    
    if use_semantic:
        print(f"Using SEMANTIC chunking (threshold: {threshold})")
    else:
        print("Using SIMPLE sentence-based chunking")
    
    if workers > 1:
        print(f"Parallel processing: {workers} workers")
    else:
        print("Sequential processing")
    
    passages = []
    passage_index = 0
    
    # Prepare arguments for workers
    worker_args = [
        (txt_file, max_length, overlap, use_semantic, threshold)
        for txt_file in txt_files
    ]
    
    # Process articles (parallel or sequential)
    if workers > 1:
        # Parallel processing with multiprocessing
        with Pool(processes=workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(chunk_article_worker, worker_args),
                total=len(txt_files),
                desc="Chunking articles"
            ))
    else:
        # Sequential processing
        results = []
        for args in tqdm(worker_args, desc="Chunking articles"):
            results.append(chunk_article_worker(args))
    
    # Convert results to passages
    for txt_file, chunks, metadata in results:
        for chunk in chunks:
            # chunk is a dict with 'text' and 'section' keys
            chunk_text = chunk['text']
            chunk_section = chunk['section']
            
            passage = {
                'index': passage_index,
                'article_filename': txt_file.name,
                'article_title': metadata.get('title', txt_file.stem),
                'article_url': metadata.get('url', ''),
                'source_url': metadata.get('source_url', ''),
                'section': chunk_section,
                'passage': chunk_text,
                'passage_length': len(chunk_text)
            }
            passages.append(passage)
            passage_index += 1
    
    # Save passages JSON
    print(f"\nSaving {len(passages)} passages to {output_json}")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(passages, f, indent=2, ensure_ascii=False)
    
    # Calculate statistics
    stats = {
        'total_articles': len(txt_files),
        'total_passages': len(passages),
        'avg_passages_per_article': len(passages) / len(txt_files) if txt_files else 0,
        'chunk_size': max_length,
        'overlap': overlap,
        'semantic_chunking': use_semantic,
        'similarity_threshold': threshold if use_semantic else None
    }
    
    return stats


def download_frames_dataset(output_dir: str) -> Optional[str]:
    """Download the FRAMES dataset from Hugging Face and save as TSV."""
    print("Downloading FRAMES dataset from Hugging Face...")
    
    try:
        # Load the dataset
        dataset = load_dataset("google/frames-benchmark", split="test")
        
        # Convert to pandas DataFrame
        df = dataset.to_pandas()
        
        # Create output directory if it doesn't exist
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save as TSV
        tsv_path = output_dir / "frames_dataset.tsv"
        df.to_csv(tsv_path, sep='\t', index=False)
        
        print(f"✅ FRAMES dataset downloaded and saved to: {tsv_path}")
        print(f"Dataset contains {len(df)} rows")
        print(f"Columns: {list(df.columns)}")
        
        return str(tsv_path)
        
    except Exception as e:
        print(f"❌ Error downloading FRAMES dataset: {e}")
        return None


def extract_wikipedia_links(item) -> List[str]:
    """
    Extract Wikipedia links from FRAMES dataset wiki_links field.
    
    The wiki_links field is a Python list string that should be properly formatted.
    """
    links = []
    
    if isinstance(item, list):
        # Already a list
        links = [url for url in item if isinstance(url, str) and url.startswith('https://en.wikipedia.org/wiki/')]
    elif isinstance(item, str):
        try:
            # Parse as Python list
            parsed = ast.literal_eval(item)
            if isinstance(parsed, list):
                links = [url for url in parsed if isinstance(url, str) and url.startswith('https://en.wikipedia.org/wiki/')]
            elif isinstance(parsed, str) and parsed.startswith('https://en.wikipedia.org/wiki/'):
                links = [parsed]
        except (ValueError, SyntaxError) as e:
            print(f"Warning: Failed to parse wiki_links: {item[:100]}... Error: {e}")
            return []
    
    return links


def main():
    parser = argparse.ArgumentParser(
        description='Download clean Wikipedia articles from FRAMES dataset using Wikipedia API'
    )
    parser.add_argument(
        '--tsv_path',
        default=None,
        help='Input TSV file with FRAMES data (default: download from Hugging Face)'
    )
    parser.add_argument(
        '--output_dir',
        default='wiki_clean_articles',
        help='Output directory for clean article text files (default: wiki_clean_articles)'
    )
    parser.add_argument(
        '--output_data',
        default='/tmp/data',
        help='Output directory for dataset file, if downloaded from Hugging Face (default: data)'
    )
    parser.add_argument(
        '--max_urls',
        type=int,
        default=None,
        help='Maximum number of URLs to process (default: all)'
    )
    parser.add_argument(
        '--download-workers',
        type=int,
        default=10,
        help='Number of parallel workers for downloading articles (default: 10, max recommended: 20)'
    )
    parser.add_argument(
        '--create_chunks',
        action='store_true',
        help='Create passages JSON file with chunked content'
    )
    parser.add_argument(
        '--chunk_size',
        type=int,
        default=512,
        help='Maximum characters per chunk (default: 512)'
    )
    parser.add_argument(
        '--overlap',
        type=int,
        default=50,
        help='Overlap between chunks in characters (default: 50)'
    )
    parser.add_argument(
        '--semantic',
        action='store_true',
        default=True,
        help='Use semantic chunking based on sentence similarity (default: True, requires spacy)'
    )
    parser.add_argument(
        '--no-semantic',
        dest='semantic',
        action='store_false',
        help='Disable semantic chunking, use simple sentence-based chunking'
    )
    parser.add_argument(
        '--similarity-threshold',
        type=float,
        default=0.75,
        help='Semantic similarity threshold 0-1 for grouping sentences (default: 0.75)'
    )
    parser.add_argument(
        '--chunk-workers',
        type=int,
        default=1,
        help='Number of parallel workers for chunking articles (default: 1, sequential). Recommended: 2-8 for large datasets'
    )
    
    args = parser.parse_args()
    
    # Validate chunk workers count
    max_workers = cpu_count()
    if args.chunk_workers > max_workers:
        print(f"⚠️  WARNING: {args.chunk_workers} chunk workers exceeds CPU count ({max_workers})!")
        print(f"   Reducing to {max_workers} workers...")
        args.chunk_workers = max_workers
    elif args.chunk_workers > 8:
        print(f"⚠️  Note: {args.chunk_workers} chunk workers may use significant memory (~500MB per worker)")
        print(f"   Expected memory usage: ~{args.chunk_workers * 0.5:.1f}GB")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Determine TSV file path
    if args.tsv_path is None:
        print("=== DOWNLOADING FRAMES DATASET ===")
        tsv_path = download_frames_dataset(args.output_data)
        if tsv_path is None:
            print("❌ Failed to download FRAMES dataset. Exiting.")
            return
        args.tsv_path = tsv_path
    else:
        print(f"Using provided TSV file: {args.tsv_path}")
    
    # Load dataset
    print("\nLoading FRAMES dataset...")
    df = pd.read_csv(args.tsv_path, sep='\t')
    
    # Extract all unique Wikipedia URLs
    print("Extracting Wikipedia URLs from dataset...")
    urls = set()
    excluded_urls = []
    
    for item in tqdm(df.wiki_links, desc="Processing wiki_links"):
        for link in extract_wikipedia_links(item):
            # Filter out non-article pages
            if '/wiki/Category:' in link or '/wiki/File:' in link or '/wiki/Template:' in link:
                excluded_urls.append(link)
                continue
            urls.add(link)
    
    urls = sorted(list(urls))
    
    if excluded_urls:
        print(f"\nℹ️  Excluded {len(set(excluded_urls))} non-article pages (Categories, Files, Templates)")
    
    if args.max_urls:
        urls = urls[:args.max_urls]
    
    if not urls:
        print("No URLs found to process")
        return
    
    print(f"\n=== DOWNLOADING CLEAN WIKIPEDIA ARTICLES ===")
    print(f"Total unique Wikipedia URLs: {len(urls)}")
    print(f"Output directory: {output_dir}")
    print(f"Download workers: {args.download_workers}")
    
    # Prepare arguments for multiprocessing
    process_args = [(url, output_dir) for url in urls]
    
    # Process URLs in parallel with progress bar
    start_time = time.time()
    
    results = []
    failed_urls = []
    
    with Pool(processes=args.download_workers) as pool:
        with tqdm(total=len(urls), desc="Downloading articles", unit="article") as pbar:
            # Use imap_unordered for immediate progress updates as workers complete
            for result in pool.imap_unordered(process_single_url, process_args):
                success, url, content, message = result
                results.append((success, url))
                
                if success:
                    if message == "Already exists":
                        pbar.set_description(f"⊙ Skipped (exists)")
                    else:
                        pbar.set_description(f"✓ Downloaded")
                else:
                    pbar.set_description(f"✗ Failed")
                    failed_urls.append((url, message))
                    if message not in ["Already exists"]:
                        print(f"\n❌ FAILED: {url}")
                        print(f"   Error: {message}")
                
                pbar.update(1)
    
    # Count results
    successful = sum(1 for success, _ in results if success)
    failed = len(results) - successful
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n=== DOWNLOAD COMPLETE ===")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {duration:.2f} seconds")
    print(f"Average time per URL: {duration/len(urls):.2f} seconds")
    
    # Print detailed failure report if there were failures
    if failed_urls:
        print(f"\n=== FAILED DOWNLOADS ===")
        for i, (url, message) in enumerate(failed_urls, 1):
            print(f"{i:2d}. {url}")
            print(f"    Error: {message}")
    
    # Create passages JSON if requested
    if args.create_chunks:
        print(f"\n=== CREATING PASSAGES ===")
        
        # Create filename with chunking parameters
        chunk_method = f"semantic{args.similarity_threshold}" if args.semantic else "simple"
        passages_filename = f"passages_chunk{args.chunk_size}_overlap{args.overlap}_{chunk_method}.json"
        passages_json = output_dir / passages_filename
        
        stats = create_passages_from_articles(
            output_dir,
            passages_json,
            max_length=args.chunk_size,
            overlap=args.overlap,
            use_semantic=args.semantic,
            threshold=args.similarity_threshold,
            workers=args.chunk_workers
        )
        
        if "error" not in stats:
            print(f"\n=== PASSAGES CREATED ===")
            print(f"Total articles: {stats['total_articles']}")
            print(f"Total passages: {stats['total_passages']}")
            print(f"Average passages per article: {stats['avg_passages_per_article']:.1f}")
            
            # Show chunking method used
            if stats.get('semantic_chunking'):
                print(f"Chunking method: SEMANTIC (similarity threshold: {stats['similarity_threshold']})")
            else:
                print(f"Chunking method: SIMPLE sentence-based")
            
            print(f"Passages saved to: {passages_json}")
        else:
            print(f"Error creating passages: {stats['error']}")
    else:
        print(f"\nTo create passages JSON, run again with --create_chunks flag:")
        print(f"  python {os.path.basename(__file__)} --output_dir {output_dir} --create_chunks")


if __name__ == "__main__":
    main()

