"""
Passage creation and management from Wikipedia articles.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Union
from multiprocessing import Pool
from tqdm import tqdm
from .chunker import TextChunker


class PassageBuilder:
    """
    Builds passage JSON files from downloaded Wikipedia articles.
    Handles parallel processing and metadata management.
    """
    
    def __init__(self, chunker: TextChunker):
        """
        Initialize the passage builder.
        
        Args:
            chunker: TextChunker instance for chunking articles
        """
        self.chunker = chunker
    
    def create_passages_from_articles(
        self,
        articles_dir: Union[Path, str],
        output_json: Union[Path, str],
        max_length: int = 512,
        overlap: int = 50,
        workers: int = 1,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Create passages JSON from downloaded articles.
        
        Args:
            articles_dir: Directory containing article .txt files
            output_json: Output path for passages.json
            max_length: Maximum characters per chunk
            overlap: Overlap in characters
            workers: Number of parallel workers for chunking (1=sequential)
            verbose: Print progress information
        
        Returns:
            Dictionary with statistics
        """
        if isinstance(output_json, str):
            output_json = Path(output_json)
        if isinstance(articles_dir, str):
            articles_dir = Path(articles_dir)
        
        # Find all text files
        txt_files = list(articles_dir.glob("*.txt"))
        
        if not txt_files:
            return {"error": f"No text files found in {articles_dir}"}
        
        if verbose:
            print(f"\nCreating passages from {len(txt_files)} articles...")
            print(f"Chunk size: {max_length} characters, Overlap: {overlap} characters")
            print(f"Chunking method: {'SEMANTIC' if self.chunker.use_semantic else 'SIMPLE'}")
            if self.chunker.use_semantic:
                print(f"Similarity threshold: {self.chunker.similarity_threshold}")
            if workers > 1:
                print(f"Parallel processing: {workers} workers")
                if self.chunker.use_semantic:
                    print(f"⚠️  Note: First few articles may be slow (loading models in each worker)")
            else:
                print("Sequential processing")
        
        passages = []
        passage_index = 0
        
        # Prepare arguments for workers
        worker_args = [
            (txt_file, max_length, overlap)
            for txt_file in txt_files
        ]
        
        # Process articles (parallel or sequential)
        if workers > 1:
            # Parallel processing with multiprocessing
            with Pool(processes=workers) as pool:
                results = list(tqdm(
                    pool.imap_unordered(self._chunk_article_worker_wrapper, worker_args),
                    total=len(txt_files),
                    desc="Chunking articles",
                    disable=not verbose
                ))
        else:
            # Sequential processing
            results = []
            iterator = tqdm(worker_args, desc="Chunking articles", disable=not verbose)
            for args in iterator:
                results.append(self._process_single_article(args))
        
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
        if verbose:
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
            'semantic_chunking': self.chunker.use_semantic,
            'similarity_threshold': self.chunker.similarity_threshold if self.chunker.use_semantic else None
        }
        
        return stats
    
    def _process_single_article(
        self,
        args_tuple: Tuple[Path, int, int]
    ) -> Tuple[Path, List[Dict[str, str]], Dict]:
        """
        Process a single article file.
        
        Args:
            args_tuple: (txt_file, max_length, overlap)
        
        Returns:
            Tuple of (txt_file, chunks_with_sections, metadata)
        """
        txt_file, max_length, overlap = args_tuple
        
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
            chunks = self.chunker.chunk_text(
                text,
                max_length=max_length,
                overlap=overlap,
                verbose=False
            )
            
            return (txt_file, chunks, metadata)
            
        except Exception as e:
            print(f"\nError processing {txt_file.name}: {e}")
            return (txt_file, [], {})
    
    def _chunk_article_worker_wrapper(self, args_tuple: Tuple) -> Tuple:
        """
        Wrapper for multiprocessing that handles model initialization per worker.
        
        Args:
            args_tuple: Arguments for _process_single_article
        
        Returns:
            Result from _process_single_article
        """
        # Each worker process will have its own chunker instance with lazy-loaded models
        return self._process_single_article(args_tuple)

