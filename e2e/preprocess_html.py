#!/usr/bin/env python3
"""
Wikipedia HTML RAG Text Processor
Complete pipeline for extracting meaningful content from Wikipedia HTML files for RAG applications.
Filters out navigation, references, captions, and other non-informational content.

Requirements:
pip install beautifulsoup4 lxml spacy sentence-transformers numpy trafilatura justext requests tqdm

Download spaCy model:
python -m spacy download en_core_web_sm

Usage:
# Process a single HTML file
python preprocess_html.py path/to/your/file.html

# Process all HTML files in a folder
python preprocess_html.py path/to/html/folder --output path/to/output/folder

# Use different extraction methods
python preprocess_html.py path/to/html/folder --method basic --output output_folder
"""

import argparse
import json
import multiprocessing
import os
import queue
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Queue, Manager
from pathlib import Path
from typing import List, Dict, Any, Optional

# Core libraries
import numpy as np
from bs4 import BeautifulSoup, Comment
import spacy
from sentence_transformers import SentenceTransformer, util

# Content extraction libraries
import trafilatura
import justext
from tqdm import tqdm


class PersistentWorker:
    """Persistent worker that initializes models once and processes multiple files."""

    def __init__(self, worker_id: int, log_dir: str = None, progress_queue=None):
        self.worker_id = worker_id
        self.log_dir = log_dir
        self.progress_queue = progress_queue
        self.stdout_file = None
        self.stderr_file = None
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

        # Setup logging
        self._setup_logging()

        # Initialize models once
        self._init_models()

    def _setup_logging(self):
        """Setup output redirection for this worker."""
        try:
            if self.log_dir:
                log_dir = Path(self.log_dir)
                log_dir.mkdir(parents=True, exist_ok=True)
                self.stdout_file = open(log_dir / f"worker_{self.worker_id}.out", 'w', encoding='utf-8')
                self.stderr_file = open(log_dir / f"worker_{self.worker_id}.err", 'w', encoding='utf-8')

                # Redirect stdout and stderr
                sys.stdout = self.stdout_file
                sys.stderr = self.stderr_file

                print(f"Worker {self.worker_id} starting - initializing models...")
        except Exception as e:
            print(f"Warning: Could not setup logging for worker {self.worker_id}: {e}")

    def _init_models(self):
        """Initialize models once for this worker."""
        try:
            print(f"Worker {self.worker_id}: Loading spaCy model...")
            import spacy
            self.nlp = spacy.load("en_core_web_sm")

            print(f"Worker {self.worker_id}: Loading embedding model...")
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

            # Create processor instance
            self.processor = WikipediaHTMLProcessor()
            self.processor.nlp = self.nlp
            self.processor.embedding_model = self.embedding_model

            print(f"Worker {self.worker_id}: Models loaded successfully!")

        except Exception as e:
            print(f"Worker {self.worker_id}: Error loading models: {e}")
            raise

    def process_file(self, file_args):
        """Process a single file."""
        (html_file_path, output_path, extraction_method, threshold, max_chunk_size,
         overlap_percentage, min_chunk_size, store_embeddings, preprocessing_args) = file_args

        try:
            print(f"Worker {self.worker_id}: Processing {Path(html_file_path).name}")

            # Process the file
            chunks = self.processor.process_document(
                str(html_file_path), extraction_method, threshold, max_chunk_size,
                overlap_percentage, min_chunk_size, verbose=True,
                store_embeddings=store_embeddings, preprocessing_args=preprocessing_args
            )

            if chunks:
                # Create output filename
                html_file = Path(html_file_path)
                output_filename = html_file.stem + "_rag_chunks.json"
                output_file_path = output_path / output_filename

                # Save chunks
                self.processor.save_chunks(chunks, str(output_file_path), verbose=True)

                print(f"Worker {self.worker_id}: Completed {html_file.name} - {len(chunks)} chunks")

                result = {
                    'status': 'success',
                    'file': html_file.name,
                    'chunks': len(chunks)
                }

                # Send progress update
                if self.progress_queue:
                    self.progress_queue.put(result)

                return result

            else:
                print(f"Worker {self.worker_id}: Failed {Path(html_file_path).name} - No chunks generated")
                result = {
                    'status': 'failed',
                    'file': Path(html_file_path).name,
                    'error': 'No chunks generated'
                }

                # Send progress update
                if self.progress_queue:
                    self.progress_queue.put(result)

                return result

        except Exception as e:
            print(f"Worker {self.worker_id}: Error processing {Path(html_file_path).name}: {str(e)}")
            result = {
                'status': 'failed',
                'file': Path(html_file_path).name,
                'error': str(e)
            }

            # Send progress update
            if self.progress_queue:
                self.progress_queue.put(result)

            return result

    def process_files(self, file_list):
        """Process a list of files."""
        results = []
        for file_args in file_list:
            result = self.process_file(file_args)
            results.append(result)
        return results

    def cleanup(self):
        """Clean up resources."""
        # Restore original stdout/stderr and close files
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

        if self.stdout_file:
            self.stdout_file.close()
        if self.stderr_file:
            self.stderr_file.close()


def persistent_worker_main(worker_args):
    """Main function for persistent worker process."""
    worker_id, file_batch, log_dir, progress_queue = worker_args

    worker = None
    try:
        # Create persistent worker
        worker = PersistentWorker(worker_id, log_dir, progress_queue)

        # Process all files in this batch
        results = worker.process_files(file_batch)

        return results

    except Exception as e:
        error_result = {
            'status': 'failed',
            'file': 'worker_initialization',
            'error': f'Worker {worker_id} failed to initialize: {str(e)}'
        }

        # Send error update
        if progress_queue:
            progress_queue.put(error_result)

        return [error_result]
    finally:
        if worker:
            worker.cleanup()


class WikipediaHTMLProcessor:
    def __init__(self):
        """Initialize the processor with required models."""
        print("Loading models...")

        # Load spaCy model - let it fail if not available
        self.nlp = spacy.load("en_core_web_sm")

        # Load embedding model - let it fail if not available
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        print("Models loaded successfully!")

    def extract_html_content(self, html_path: str, verbose: bool = True) -> str:
        """Extract HTML content from file."""
        if verbose:
            print(f"Reading HTML file: {html_path}")
        try:
            with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
                html_content = f.read()
            if verbose:
                print(f"Read {len(html_content)} characters from HTML file")
            return html_content
        except Exception as e:
            print(f"Error reading HTML file: {e}")
            return ""

    def advanced_content_extraction(self, html_content: str, verbose: bool = True) -> str:
        """Use trafilatura for advanced content extraction."""
        if verbose:
            print("Using trafilatura for advanced content extraction...")

        # Extract main content using trafilatura
        extracted = trafilatura.extract(html_content,
                                      include_comments=False,
                                      include_tables=True,
                                      include_images=False,
                                      include_links=False,
                                      favor_precision=True)
        if extracted:
            if verbose:
                print(f"Trafilatura extracted {len(extracted)} characters")
            return extracted
        else:
            raise RuntimeError("Trafilatura failed to extract any content from the HTML")

    def basic_content_extraction(self, html_content: str, verbose: bool = True) -> str:
        """Basic content extraction with Wikipedia-specific filtering."""
        if verbose:
            print("Using basic content extraction with Wikipedia filtering...")

        soup = BeautifulSoup(html_content, 'lxml')

        # Remove unwanted elements that don't contain meaningful content
        unwanted_elements = [
            # Scripts and styles
            'script', 'style', 'noscript',

            # Navigation and UI elements
            'nav', 'header', 'footer', 'aside',

            # Wikipedia-specific navigation and metadata
            '.navbox', '.infobox', '.mbox', '.sidebar', '.vertical-navbox',
            '.navigation-not-searchable', '.noprint',

            # References and citations (these are typically just numbers/links)
            '.reference', '.references', '.reflist', '.cite',

            # Edit links and administrative content
            '.editsection', '.mw-editsection', '.edit-page',

            # Disambiguation and maintenance templates
            '.hatnote', '.dablink', '.rellink',

            # Category and template information
            '.catlinks', '.printfooter', '.mw-normal-catlinks',

            # Image galleries and media metadata
            '.gallery', '.thumbcaption', '.thumbinner',

            # Tables of contents (keep headings but remove TOC)
            '.toc', '.toccolours',

            # Stub and maintenance messages
            '.ambox', '.tmbox', '.cmbox', '.fmbox', '.imbox', '.ombox',
        ]

        # Remove unwanted elements
        for selector in unwanted_elements:
            if selector.startswith('.'):
                # CSS class selector
                class_name = selector[1:]
                elements = soup.find_all(class_=class_name)
            else:
                # Tag selector
                elements = soup.find_all(selector)

            for element in elements:
                element.decompose()

        # Remove comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

        # Focus on main content areas (Wikipedia specific)
        main_content_selectors = [
            '#mw-content-text',  # Main Wikipedia content
            '.mw-parser-output',  # Parsed content
            '#content',  # Generic content
            'main',  # HTML5 main element
            '.content',  # Generic content class
        ]

        main_content = None
        for selector in main_content_selectors:
            if selector.startswith('#'):
                # ID selector
                element_id = selector[1:]
                main_content = soup.find(id=element_id)
            elif selector.startswith('.'):
                # Class selector
                class_name = selector[1:]
                main_content = soup.find(class_=class_name)
            else:
                # Tag selector
                main_content = soup.find(selector)

            if main_content:
                break

        # If no main content area found, use body
        if not main_content:
            main_content = soup.find('body') or soup

        # Extract meaningful text content
        text_content = self.extract_meaningful_text(main_content)

        # Clean the extracted text
        cleaned_text = self.clean_extracted_text(text_content, verbose=verbose)

        if verbose:
            print(f"Basic extraction yielded {len(cleaned_text)} characters")
        return cleaned_text

    def extract_meaningful_text(self, element) -> str:
        """Extract meaningful text while filtering out low-value content."""

        # Tags that typically contain meaningful content
        meaningful_tags = [
            'p', 'div', 'article', 'section', 'main',
            'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
            'li', 'td', 'th', 'blockquote', 'dd', 'dt'
        ]

        # Tags to skip entirely
        skip_tags = [
            'script', 'style', 'nav', 'header', 'footer', 'aside',
            'button', 'input', 'select', 'textarea', 'form'
        ]

        text_parts = []

        def process_element(elem):
            if elem.name in skip_tags:
                return

            # Skip elements with certain classes
            elem_classes = elem.get('class', [])
            skip_classes = [
                'reference', 'cite', 'navbox', 'infobox', 'mbox',
                'editsection', 'mw-editsection', 'noprint', 'printfooter',
                'catlinks', 'thumbcaption', 'gallery', 'toc'
            ]

            if any(skip_class in elem_classes for skip_class in skip_classes):
                return

            # Skip elements with certain IDs
            elem_id = elem.get('id', '')
            skip_ids = ['toc', 'references', 'external-links', 'see-also', 'notes']
            if elem_id.lower() in skip_ids:
                return

            if elem.name in meaningful_tags:
                # Get direct text content
                if elem.name in ['p', 'div', 'li', 'td', 'th', 'blockquote', 'dd', 'dt']:
                    text = elem.get_text(strip=True)
                    if text and self.is_meaningful_text(text):
                        text_parts.append(text)
                elif elem.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    # Include headings as they provide structure
                    heading_text = elem.get_text(strip=True)
                    if heading_text and len(heading_text) > 3:  # Skip very short headings
                        text_parts.append(f"\n{heading_text}\n")

            # Recursively process children for container elements
            if elem.name in ['div', 'section', 'article', 'main']:
                for child in elem.children:
                    if hasattr(child, 'name') and child.name:  # Skip text nodes
                        process_element(child)

        # Process top-level elements (recursive processing handled inside process_element)
        for elem in element.children:
            if hasattr(elem, 'name') and elem.name:  # Skip text nodes
                process_element(elem)

        # Join and clean up
        full_text = ' '.join(text_parts)
        return full_text

    def is_meaningful_text(self, text: str) -> bool:
        """Determine if text contains meaningful information."""
        if not text or len(text.strip()) < 20:
            return False

        # Skip if it's mostly numbers or special characters
        if len(re.sub(r'[^a-zA-Z\s]', '', text)) < len(text) * 0.5:
            return False

        # Skip common non-informational patterns
        skip_patterns = [
            r'^(edit|source|cite|reference)$',
            r'^\[\d+\]$',  # Reference numbers
            r'^(category:|file:|image:)',  # Wikipedia namespace prefixes
            r'^(coordinates?:)',
            r'^(see also|references|external links|further reading)$',
            r'^\d+\s*(px|em|%|\w+\s*=)',  # Styling information
            r'^(thumb|left|right|center|\d+px)$',  # Image positioning
            r'missing image',
            r'page \d+',
            r'^\w+\s*\|\s*\w+$',  # Simple key-value pairs
        ]

        text_lower = text.lower().strip()
        for pattern in skip_patterns:
            if re.match(pattern, text_lower, re.IGNORECASE):
                return False

        # Check if text has reasonable sentence structure
        sentences = re.split(r'[.!?]+', text)
        meaningful_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        return len(meaningful_sentences) > 0

    def clean_extracted_text(self, text: str, verbose: bool = True) -> str:
        """Clean and normalize extracted text."""
        if verbose:
            print("Cleaning extracted text...")

        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)

        # Remove Wikipedia-specific artifacts
        text = re.sub(r'\[\d+\]', '', text)  # Reference numbers
        text = re.sub(r'\[edit\]', '', text, flags=re.IGNORECASE)  # Edit links
        text = re.sub(r'\[citation needed\]', '', text, flags=re.IGNORECASE)

        # Remove coordinates
        text = re.sub(r'Coordinates:\s*\d+°[^.]*\.', '', text)

        # Remove common Wikipedia footer text
        footer_patterns = [
            r'This article about.*is a stub',
            r'You can help Wikipedia by expanding it',
            r'Categories?:\s*[A-Za-z\s,]+$',
            r'Retrieved from.*$',
            r'This page was last edited.*$'
        ]

        for pattern in footer_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # Clean up multiple spaces again
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def semantic_chunking(self, text: str, threshold: float = 0.75, max_chunk_size: int = 400, verbose: bool = True) -> List[str]:
        """Perform semantic chunking based on sentence similarity."""
        if verbose:
            print("Performing semantic chunking...")

        # Split into sentences
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 15]

        if not sentences:
            return [text] if text else []

        if verbose:
            print(f"Processing {len(sentences)} sentences...")

        # Skip very short texts
        if len(sentences) <= 2:
            return [text]

        # Encode sentences in batches for efficiency
        sentence_embeddings = self.embedding_model.encode(sentences, show_progress_bar=False)

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
            potential_token_count = len(potential_chunk_text.split())

            # Allow chunks to grow up to max_chunk_size during semantic chunking
            # Final size enforcement happens after overlap is added
            if similarity >= threshold and potential_token_count < max_chunk_size:
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
            print(f"Created {len(chunks)} semantic chunks")
        return chunks

    def add_chunk_overlap(self, chunks: List[str], overlap_percentage: float = 0.15, max_chunk_size: int = 400, verbose: bool = True) -> List[str]:
        """Add overlap between chunks to preserve context while enforcing size limits."""
        if len(chunks) <= 1:
            return chunks

        if verbose:
            print("Adding chunk overlap...")

        overlapped_chunks = []

        for i, chunk in enumerate(chunks):
            if i == 0:
                # First chunk - check if it needs truncation
                chunk_words = chunk.split()
                if len(chunk_words) > max_chunk_size:
                    truncated_chunk = ' '.join(chunk_words[:max_chunk_size])
                    overlapped_chunks.append(truncated_chunk)
                else:
                    overlapped_chunks.append(chunk)
            else:
                # Calculate overlap size
                prev_words = chunks[i-1].split()
                overlap_size = max(1, int(len(prev_words) * overlap_percentage))
                overlap_text = ' '.join(prev_words[-overlap_size:])

                # Combine overlap with current chunk
                combined_chunk = overlap_text + ' ' + chunk
                combined_words = combined_chunk.split()

                # Enforce size limit after adding overlap
                if len(combined_words) > max_chunk_size:
                    # Truncate to fit within size limit
                    truncated_chunk = ' '.join(combined_words[:max_chunk_size])
                    overlapped_chunks.append(truncated_chunk)
                else:
                    overlapped_chunks.append(combined_chunk)

        if verbose:
            print(f"Added overlap to {len(overlapped_chunks)} chunks (enforcing max {max_chunk_size} words)")
        return overlapped_chunks

    def merge_small_chunks(self, chunks: List[str], min_chunk_size: int = 50, max_chunk_size: int = 400, verbose: bool = True) -> List[str]:
        """Merge small chunks with adjacent chunks to ensure meaningful content."""
        if not chunks:
            return chunks

        if verbose:
            print(f"Merging chunks smaller than {min_chunk_size} words...")

        merged_chunks = []
        i = 0

        while i < len(chunks):
            current_chunk = chunks[i]
            current_words = len(current_chunk.split())

            # If current chunk is too small, try to merge it
            if current_words < min_chunk_size:
                # Try to merge with next chunk first (better for context flow)
                if i + 1 < len(chunks):
                    next_chunk = chunks[i + 1]
                    combined = current_chunk + ' ' + next_chunk
                    combined_words = len(combined.split())

                    # If combined chunk fits within max size, merge them
                    if combined_words <= max_chunk_size:
                        merged_chunks.append(combined)
                        i += 2  # Skip both chunks as they're now merged
                        continue

                # If can't merge with next, try to merge with previous
                if merged_chunks:
                    prev_chunk = merged_chunks[-1]
                    combined = prev_chunk + ' ' + current_chunk
                    combined_words = len(combined.split())

                    # If combined chunk fits within max size, merge with previous
                    if combined_words <= max_chunk_size:
                        merged_chunks[-1] = combined  # Replace last chunk with merged version
                        i += 1
                        continue

                # If can't merge with either, keep as is (better than losing content)
                merged_chunks.append(current_chunk)
            else:
                # Chunk is already good size
                merged_chunks.append(current_chunk)

            i += 1

        # Filter out any remaining very small chunks (< 10 words) as last resort
        final_chunks = [chunk for chunk in merged_chunks if len(chunk.split()) >= 10]

        if verbose:
            print(f"Merged {len(chunks)} chunks into {len(final_chunks)} chunks (min {min_chunk_size} words)")
        return final_chunks

    def process_document(self, file_path: str, extraction_method: str = "advanced",
                        threshold: float = 0.75, max_chunk_size: int = 400,
                        overlap_percentage: float = 0.15, min_chunk_size: int = 50,
                        verbose: bool = True, store_embeddings: bool = False,
                        preprocessing_args: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Complete processing pipeline for a Wikipedia HTML file.

        Args:
            file_path: Path to the HTML file
            extraction_method: 'advanced', 'basic', or 'justext'
            threshold: Semantic similarity threshold for chunking
            max_chunk_size: Maximum words per chunk after overlap
            overlap_percentage: Chunk overlap percentage
            min_chunk_size: Minimum words per chunk (smaller chunks will be merged)
            verbose: Whether to print processing details
            store_embeddings: Whether to include embeddings in output JSON
            preprocessing_args: Dictionary of preprocessing arguments to include in metadata

        Returns:
            List of processed chunks with optional embeddings and metadata
        """
        if verbose:
            print(f"\n=== Processing Wikipedia HTML: {file_path} ===")

        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            return []

        # Step 1: Read HTML content
        html_content = self.extract_html_content(file_path, verbose=verbose)
        if not html_content:
            if verbose:
                print("No content read from HTML file")
            return []

        # Step 2: Extract meaningful content
        if extraction_method == "advanced":
            extracted_text = self.advanced_content_extraction(html_content, verbose=verbose)
        elif extraction_method == "justext":
            extracted_text = self.justext_extraction(html_content, verbose=verbose)
        elif extraction_method == "basic":
            extracted_text = self.basic_content_extraction(html_content, verbose=verbose)
        else:
            raise ValueError(f"Unknown extraction method: {extraction_method}. Must be 'advanced', 'basic', or 'justext'")

        if not extracted_text:
            if verbose:
                print("No meaningful text extracted")
            return []

        if verbose:
            print(f"Text length after extraction: {len(extracted_text)} characters")

        # Step 3: Semantic chunking
        chunks = self.semantic_chunking(extracted_text, threshold=threshold, max_chunk_size=max_chunk_size, verbose=verbose)

        if not chunks:
            if verbose:
                print("No chunks created")
            return []

        # Step 4: Add overlap (enforcing final chunk size limits)
        overlapped_chunks = self.add_chunk_overlap(chunks, overlap_percentage=overlap_percentage, max_chunk_size=max_chunk_size, verbose=verbose)

        # Step 5: Merge small chunks to ensure meaningful content
        final_chunks = self.merge_small_chunks(overlapped_chunks, min_chunk_size=min_chunk_size, max_chunk_size=max_chunk_size, verbose=verbose)

        # Step 6: Create final chunks (optionally with embeddings)
        if store_embeddings and verbose:
            print("Generating embeddings for chunks...")
        elif verbose:
            print("Creating chunks (without embeddings)...")
        processed_chunks = []

        for i, chunk in enumerate(final_chunks):

            # Create base chunk data
            chunk_data = {
                'chunk_id': i,
                'text': chunk,
                'metadata': {
                    'source': file_path,
                    'word_count': len(chunk.split()),
                    'char_count': len(chunk),
                    'type': 'html',
                    'extraction_method': extraction_method
                }
            }

            # Add preprocessing arguments to metadata if provided
            if preprocessing_args:
                chunk_data['metadata']['preprocessing_args'] = preprocessing_args

            # Conditionally add embeddings
            if store_embeddings:
                chunk_embedding = self.embedding_model.encode(chunk)
                chunk_data['embedding'] = chunk_embedding.tolist()  # Convert to list for JSON serialization

            processed_chunks.append(chunk_data)

        if verbose:
            print(f"Successfully processed {len(processed_chunks)} chunks")
        return processed_chunks

    def justext_extraction(self, html_content: str, verbose: bool = True) -> str:
        """Use justext for content extraction."""
        if verbose:
            print("Using justext for content extraction...")

        paragraphs = justext.justext(html_content, justext.get_stoplist("English"))
        text_parts = []

        for paragraph in paragraphs:
            if not paragraph.is_boilerplate and len(paragraph.text.strip()) > 20:
                text_parts.append(paragraph.text.strip())

        extracted_text = '\n\n'.join(text_parts)
        if not extracted_text.strip():
            raise RuntimeError("Justext failed to extract any meaningful content from the HTML")

        if verbose:
            print(f"Justext extracted {len(extracted_text)} characters")
        return extracted_text

    def save_chunks(self, chunks: List[Dict[str, Any]], output_path: str, verbose: bool = True):
        """Save processed chunks to JSON file."""
        if verbose:
            print(f"Saving chunks to: {output_path}")

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False)
            if verbose:
                print(f"Successfully saved {len(chunks)} chunks")
        except Exception as e:
            print(f"Error saving chunks: {e}")


    def process_folder(self, input_folder: str, output_folder: str, extraction_method: str = "advanced",
                      threshold: float = 0.75, max_chunk_size: int = 400,
                      overlap_percentage: float = 0.15, min_chunk_size: int = 50,
                      store_embeddings: bool = False, preprocessing_args: Dict[str, Any] = None,
                      max_workers: int = 4, limit_files: int = None) -> Dict[str, Any]:
        """
        Process HTML files in a folder using multiprocessing.

        Args:
            input_folder: Path to folder containing HTML files
            output_folder: Path to folder where output JSON files will be saved
            extraction_method: Content extraction method to use
            threshold: Semantic similarity threshold for chunking
            max_chunk_size: Maximum words per chunk after overlap
            overlap_percentage: Chunk overlap percentage
            min_chunk_size: Minimum words per chunk (smaller chunks will be merged)
            store_embeddings: Whether to include embeddings in output JSON
            preprocessing_args: Dictionary of preprocessing arguments to include in metadata
            max_workers: Number of worker processes to use
            limit_files: Maximum number of files to process (None = process all)

        Returns:
            Dictionary with processing statistics
        """
        input_path = Path(input_folder)
        output_path = Path(output_folder)

        if not input_path.exists():
            print(f"Error: Input folder not found: {input_folder}")
            return {"error": "Input folder not found"}

        # Create output folder if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)

        # Find all HTML files
        html_files = []
        for pattern in ['*.html', '*.htm']:
            html_files.extend(input_path.glob(pattern))

        if not html_files:
            print(f"No HTML files found in {input_folder}")
            return {"error": "No HTML files found"}

        # Apply file limit if specified
        total_files_found = len(html_files)
        if limit_files is not None and limit_files > 0:
            if limit_files < len(html_files):
                html_files = html_files[:limit_files]
                print(f"Limiting to first {limit_files} files out of {total_files_found} found")
            elif limit_files >= len(html_files):
                print(f"Limit ({limit_files}) is greater than or equal to files found ({total_files_found}), processing all files")

        print(f"\n=== PROCESSING FOLDER ===")
        print(f"Input folder: {input_folder}")
        print(f"Output folder: {output_folder}")
        print(f"Found {total_files_found} HTML files")
        if limit_files is not None:
            print(f"Processing {len(html_files)} files (limited)")
        else:
            print(f"Processing all {len(html_files)} files")
        print(f"Extraction method: {extraction_method}")
        print(f"Using {max_workers} worker processes")

        # Initialize results
        results = {
            "total_files": len(html_files),
            "successful": 0,
            "failed": 0,
            "total_chunks": 0,
            "failed_files": []
        }

        # Create logs directory for worker output
        logs_dir = output_path / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        print(f"Worker logs will be saved to: {logs_dir}")

        # Split files among workers - each worker gets a batch of files
        def chunk_list(lst, n):
            """Split list into n roughly equal chunks."""
            k, m = divmod(len(lst), n)
            return [lst[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]

        # Create file batches for workers
        file_batches = chunk_list(html_files, max_workers)

        # Create progress queue for real-time updates
        with Manager() as manager:
            progress_queue = manager.Queue()

            # Prepare arguments for each worker with their file batch
            worker_tasks = []
            for worker_id, file_batch in enumerate(file_batches):
                if file_batch:  # Only create workers that have files to process
                    # Convert file batch to argument tuples
                    file_args_batch = [
                        (str(html_file), output_path, extraction_method, threshold, max_chunk_size,
                         overlap_percentage, min_chunk_size, store_embeddings, preprocessing_args)
                        for html_file in file_batch
                    ]

                    worker_tasks.append((worker_id, file_args_batch, str(logs_dir), progress_queue))

            print(f"Distributing {len(html_files)} files among {len(worker_tasks)} workers")
            for i, (worker_id, file_batch, _, _) in enumerate(worker_tasks):
                print(f"  Worker {worker_id}: {len(file_batch)} files")

            # Use ProcessPoolExecutor with persistent workers
            with ProcessPoolExecutor(max_workers=len(worker_tasks)) as executor:
                # Submit worker tasks (each worker processes its batch of files)
                futures = [executor.submit(persistent_worker_main, worker_task) for worker_task in worker_tasks]

                # Process real-time updates from workers
                with tqdm(total=len(html_files), desc="Processing HTML files", unit="file") as pbar:
                    completed_files = 0
                    active_workers = len(worker_tasks)

                    while completed_files < len(html_files) and active_workers > 0:
                        try:
                            # Check for progress updates (non-blocking with timeout)
                            result = progress_queue.get(timeout=0.1)

                            if result['status'] == 'success':
                                results["successful"] += 1
                                results["total_chunks"] += result['chunks']
                            else:
                                results["failed"] += 1
                                results["failed_files"].append(result['file'])

                            completed_files += 1

                            # Update progress bar for each file
                            pbar.set_postfix({
                                'success': results["successful"],
                                'failed': results["failed"],
                                'chunks': results["total_chunks"]
                            })
                            pbar.update(1)

                        except queue.Empty:
                            # No updates available, check if workers are still running
                            finished_workers = sum(1 for f in futures if f.done())
                            active_workers = len(worker_tasks) - finished_workers

                            # Small sleep to avoid busy waiting
                            time.sleep(0.1)

                    # Ensure all workers have completed
                    for future in as_completed(futures):
                        future.result()  # This will raise any exceptions

        # Add logs information to results
        results["logs_directory"] = str(logs_dir)

        return results

    def display_summary(self, chunks: List[Dict[str, Any]]):
        """Display a summary of the processed chunks."""
        if not chunks:
            print("No chunks to display")
            return

        print(f"\n=== PROCESSING SUMMARY ===")
        print(f"Total chunks: {len(chunks)}")

        word_counts = [chunk['metadata']['word_count'] for chunk in chunks]
        print(f"Average words per chunk: {np.mean(word_counts):.1f}")
        print(f"Min words per chunk: {min(word_counts)}")
        print(f"Max words per chunk: {max(word_counts)}")

        print(f"\n=== SAMPLE CHUNKS ===")
        for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
            print(f"\nChunk {i+1}:")
            print(f"Words: {chunk['metadata']['word_count']}")
            text_preview = chunk['text'][:250] + "..." if len(chunk['text']) > 250 else chunk['text']
            print(f"Text: {text_preview}")

    def display_folder_summary(self, results: Dict[str, Any]):
        """Display a summary of folder processing results."""
        print(f"\n=== FOLDER PROCESSING SUMMARY ===")
        print(f"Total files processed: {results['total_files']}")
        print(f"Successful: {results['successful']}")
        print(f"Failed: {results['failed']}")
        print(f"Total chunks generated: {results['total_chunks']}")

        if results['successful'] > 0:
            avg_chunks = results['total_chunks'] / results['successful']
            print(f"Average chunks per successful file: {avg_chunks:.1f}")

        if results['failed_files']:
            print(f"\n=== FAILED FILES ===")
            for i, filename in enumerate(results['failed_files'], 1):
                print(f"{i:2d}. {filename}")

        # Show logs directory if available
        if 'logs_directory' in results:
            print(f"\n=== WORKER LOGS ===")
            print(f"Worker output logs saved to: {results['logs_directory']}")


def main():
    parser = argparse.ArgumentParser(description='Process Wikipedia HTML files for RAG application')
    parser.add_argument('input_path', help='Path to HTML file or folder containing HTML files')
    parser.add_argument('--output', '-o', help='Output file path (for single file) or output folder path (for folder input)')
    parser.add_argument('--method', '-m', choices=['advanced', 'basic', 'justext'],
                       default='advanced', help='Content extraction method (default: advanced)')
    parser.add_argument('--threshold', '-t', type=float, default=0.75,
                       help='Semantic similarity threshold (default: 0.75)')
    parser.add_argument('--chunk-size', '-s', type=int, default=400,
                       help='Maximum words per chunk after overlap is added (default: 400)')
    parser.add_argument('--overlap', type=float, default=0.15,
                       help='Chunk overlap percentage (default: 0.15)')
    parser.add_argument('--store-embeddings', action='store_true',
                       help='Store embeddings in output JSON (default: False, embeddings not stored)')
    parser.add_argument('--min-chunk-size', type=int, default=50,
                       help='Minimum words per chunk - smaller chunks will be merged with adjacent chunks (default: 50)')
    parser.add_argument('--workers', '-w', type=int, default=multiprocessing.cpu_count(),
                       help=f'Number of worker processes for folder processing (default: {multiprocessing.cpu_count()}), max: 16')
    parser.add_argument('--limit-files', type=int, default=None,
                       help='Limit the number of files to process from folder (default: process all files)')

    args = parser.parse_args()

    # Limit workers to reasonable number to avoid overwhelming the system
    if args.workers > 16:
        print(f"Warning: Limiting workers from {args.workers} to 16 to avoid overwhelming the system")
        args.workers = 16

    # Create preprocessing arguments dictionary for metadata
    preprocessing_args = {
        'extraction_method': args.method,
        'threshold': args.threshold,
        'max_chunk_size': args.chunk_size,
        'overlap_percentage': args.overlap,
        'min_chunk_size': args.min_chunk_size,
    }

    # Initialize processor
    processor = WikipediaHTMLProcessor()

    input_path = Path(args.input_path)

    # Check if input is a file or folder
    if input_path.is_file():
        # Process single file
        print("Processing single HTML file...")
        chunks = processor.process_document(str(input_path), extraction_method=args.method,
                                          threshold=args.threshold, max_chunk_size=args.chunk_size,
                                          overlap_percentage=args.overlap, min_chunk_size=args.min_chunk_size,
                                          store_embeddings=args.store_embeddings,
                                          preprocessing_args=preprocessing_args)

        if not chunks:
            print("Failed to process document")
            return

        # Display summary
        processor.display_summary(chunks)

        # Save results
        if args.output:
            output_path = args.output
        else:
            # Create output filename based on input
            output_path = f"{input_path.stem}_rag_chunks.json"

        processor.save_chunks(chunks, output_path)

        print(f"\n=== COMPLETE ===")
        print(f"Processed chunks saved to: {output_path}")

    elif input_path.is_dir():
        # Process folder
        print("Processing folder of HTML files...")

        # Determine output folder
        if args.output:
            output_folder = args.output
        else:
            output_folder = f"{input_path.name}_processed"

        # Process all HTML files in folder
        results = processor.process_folder(str(input_path), output_folder, args.method,
                                         threshold=args.threshold, max_chunk_size=args.chunk_size,
                                         overlap_percentage=args.overlap, min_chunk_size=args.min_chunk_size,
                                         store_embeddings=args.store_embeddings,
                                         preprocessing_args=preprocessing_args, max_workers=args.workers,
                                         limit_files=args.limit_files)

        if "error" in results:
            print(f"Error: {results['error']}")
            return

        # Display folder summary
        processor.display_folder_summary(results)

        print(f"\n=== COMPLETE ===")
        print(f"Processed {results['successful']} files successfully")
        print(f"Output saved to folder: {output_folder}")

    else:
        print(f"Error: Input path '{args.input_path}' is neither a file nor a directory")
        return

if __name__ == "__main__":
    # Required for multiprocessing on Windows and some other platforms
    multiprocessing.freeze_support()
    multiprocessing.set_start_method('spawn')
    main()