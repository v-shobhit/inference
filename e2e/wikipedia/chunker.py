"""
Text chunking strategies for Wikipedia articles.
"""

import re
from typing import List, Dict, Tuple, Optional
import numpy as np


class TextChunker:
    """
    Handles text chunking with multiple strategies.
    Supports semantic chunking (with embeddings) and simple sentence-based chunking.
    """
    
    def __init__(
        self,
        use_semantic: bool = True,
        similarity_threshold: float = 0.75
    ):
        """
        Initialize the text chunker.
        
        Args:
            use_semantic: Whether to use semantic chunking (requires spacy + transformers)
            similarity_threshold: Similarity threshold (0-1) for grouping sentences
        """
        self.use_semantic = use_semantic
        self.similarity_threshold = similarity_threshold
        
        # Lazy-loaded models (loaded on first use)
        self._nlp = None
        self._embedding_model = None
    
    def chunk_text(
        self,
        text: str,
        max_length: int = 512,
        overlap: int = 50,
        verbose: bool = False
    ) -> List[Dict[str, str]]:
        """
        Split text into chunks respecting section boundaries.
        
        Args:
            text: Input text to chunk
            max_length: Maximum characters per chunk
            overlap: Overlap in characters
            verbose: Print progress information
        
        Returns:
            List of dicts with 'text' and 'section' keys
        """
        if not text:
            return []
        
        # Split text by sections first
        sections = self._split_by_sections(text)
        
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
            if self.use_semantic:
                try:
                    self._ensure_models_loaded(verbose)
                    
                    # Perform semantic chunking on this section
                    section_chunks = self._semantic_chunking(
                        content,
                        max_chunk_size=max_words,
                        verbose=False
                    )
                    
                    # Add overlap within section
                    section_chunks = self._add_overlap(
                        section_chunks,
                        overlap_percentage=overlap_pct,
                        max_chunk_size=max_words
                    )
                    
                    # Merge small chunks within section
                    section_chunks = self._merge_small_chunks(
                        section_chunks,
                        min_chunk_size=min_words,
                        max_chunk_size=max_words
                    )
                    
                except Exception as e:
                    if verbose:
                        print(f"  Semantic chunking failed for section ({e}), falling back to simple chunking")
                    section_chunks = []
            
            # Fallback to simple sentence-based chunking for this section
            if not section_chunks:
                section_chunks = self._simple_chunking(content, max_length, overlap)
            
            # Create chunk dicts with section information
            for chunk_text in section_chunks:
                chunk_dict = {
                    'text': f"{heading}\n\n{chunk_text}" if heading else chunk_text,
                    'section': heading.strip() if heading else "Introduction"
                }
                all_chunks.append(chunk_dict)
        
        return all_chunks
    
    def _semantic_chunking(
        self,
        text: str,
        max_chunk_size: int = 400,
        verbose: bool = False
    ) -> List[str]:
        """
        Perform semantic chunking based on sentence similarity.
        Groups semantically similar sentences together.
        
        Args:
            text: Input text to chunk
            max_chunk_size: Maximum words per chunk
            verbose: Print progress
        
        Returns:
            List of semantically coherent text chunks
        """
        if verbose:
            print("  Performing semantic chunking...")
        
        # Import here to avoid loading if not needed
        from sentence_transformers import util
        
        # Split into sentences using spaCy
        doc = self._nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 15]
        
        if not sentences:
            return [text] if text else []
        
        # Skip very short texts
        if len(sentences) <= 2:
            return [text]
        
        if verbose:
            print(f"  Processing {len(sentences)} sentences...")
        
        # Encode sentences to embeddings
        sentence_embeddings = self._embedding_model.encode(sentences, show_progress_bar=False)
        
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
            if similarity >= self.similarity_threshold and potential_word_count < max_chunk_size:
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
    
    def _simple_chunking(
        self,
        text: str,
        max_length: int,
        overlap: int
    ) -> List[str]:
        """
        Simple sentence-based chunking fallback.
        
        Args:
            text: Text to chunk
            max_length: Maximum characters per chunk
            overlap: Overlap in characters
        
        Returns:
            List of text chunks
        """
        if len(text) <= max_length:
            return [text]
        
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            if current_length + sentence_len > max_length and current_chunk:
                chunks.append(' '.join(current_chunk))
                
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
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _split_by_sections(self, text: str) -> List[Tuple[str, str]]:
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
    
    def _add_overlap(
        self,
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
    
    def _merge_small_chunks(
        self,
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
    
    def _ensure_models_loaded(self, verbose: bool = False):
        """
        Lazy-load spaCy and embedding models on first use.
        
        Args:
            verbose: Print loading messages
        """
        if self._nlp is None or self._embedding_model is None:
            if verbose:
                print("  Loading spaCy and embedding models...")
            
            import spacy
            from sentence_transformers import SentenceTransformer
            
            self._nlp = spacy.load("en_core_web_sm")
            self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            
            if verbose:
                print("  Models loaded successfully")

