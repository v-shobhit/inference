#!/usr/bin/env python3
"""
FRAMES Wikipedia Fetcher

Downloads Wikipedia articles from FRAMES dataset using the Wikipedia API and 
creates chunked passages for RAG retrieval. Uses modular architecture with 
reusable components.

Requirements:
    pip install wikipedia-api mwparserfromhell requests tqdm datasets pandas
    pip install spacy sentence-transformers  # For semantic chunking

Usage:
    # Download articles only
    python frames_wiki_fetch.py --output-dir wiki_clean_articles
    
    # Download and create chunked passages
    python frames_wiki_fetch.py --output-dir wiki_clean_articles \\
        --create-chunks --chunk-size 512 --overlap 50
    
    # Use semantic chunking (recommended)
    python frames_wiki_fetch.py --output-dir wiki_clean_articles \\
        --create-chunks --semantic --similarity-threshold 0.75
    
    # From custom TSV file
    python frames_wiki_fetch.py --tsv-path data/custom.tsv \\
        --output-dir wiki_clean_articles
"""

import argparse
import os
from pathlib import Path
from multiprocessing import cpu_count

# Import from our new modular structure
from retrieval import PromptLoader
from wikipedia import (
    WikipediaExtractor,
    TextChunker,
    PassageBuilder,
    WikipediaDownloader,
    extract_wikipedia_urls
)


def main():
    parser = argparse.ArgumentParser(
        description='Download clean Wikipedia articles from FRAMES dataset',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Input arguments
    parser.add_argument(
        '--tsv-path',
        default=None,
        help='Input TSV file with FRAMES data (default: download from Hugging Face)'
    )
    parser.add_argument(
        '--dataset-split',
        default='test',
        help='HuggingFace dataset split to use (default: test). Ignored if --tsv-path provided.'
    )
    
    # Output arguments
    parser.add_argument(
        '--output-dir',
        default='wiki_clean_articles',
        help='Output directory for article text files (default: wiki_clean_articles)'
    )
    parser.add_argument(
        '--max-urls',
        type=int,
        default=None,
        help='Maximum number of URLs to process (default: all)'
    )
    
    # Download arguments
    parser.add_argument(
        '--download-workers',
        type=int,
        default=10,
        help='Number of parallel workers for downloading (default: 10, max recommended: 20)'
    )
    
    # Chunking arguments
    parser.add_argument(
        '--create-chunks',
        action='store_true',
        help='Create passages JSON file with chunked content'
    )
    parser.add_argument(
        '--chunk-size',
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
        help='Use semantic chunking (default: True, requires spacy)'
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
        help='Semantic similarity threshold 0-1 (default: 0.75)'
    )
    parser.add_argument(
        '--chunk-workers',
        type=int,
        default=1,
        help='Number of parallel workers for chunking (default: 1). Recommended: 2-8'
    )
    parser.add_argument(
        '--chunk-device',
        type=str,
        default=None,
        choices=['cpu', 'cuda'],
        help='Device for semantic chunking embeddings (default: auto). '
             'Use "cpu" when using many workers to avoid GPU conflicts.'
    )
    
    args = parser.parse_args()
    
    # Validate worker counts
    max_workers = cpu_count()
    if args.chunk_workers > max_workers:
        print(f"⚠️  WARNING: {args.chunk_workers} chunk workers exceeds CPU count ({max_workers})")
        print(f"   Reducing to {max_workers} workers...")
        args.chunk_workers = max_workers
    elif args.chunk_workers > 8 and args.semantic:
        print(f"⚠️  WARNING: {args.chunk_workers} chunk workers with semantic chunking is NOT recommended!")
        print(f"   - Memory usage: ~{args.chunk_workers * 0.5:.1f}GB (each worker loads embedding models)")
        print(f"   - Model loading contention can cause hanging/deadlock")
        print(f"   - GPU conflicts if using CUDA (will auto-switch to CPU)")
        print(f"   STRONGLY RECOMMENDED: Use 4-8 workers for semantic, or --no-semantic for speed")
        print(f"   Proceeding anyway... (this may hang or crash)")
    
    if args.download_workers > 20:
        print(f"⚠️  WARNING: {args.download_workers} download workers may overwhelm Wikipedia API")
        print(f"   Reducing to 20 workers...")
        args.download_workers = 20
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # ===================================================================
    # STEP 1: LOAD FRAMES DATASET (using retrieval.PromptLoader)
    # ===================================================================
    print("=" * 80)
    print("STEP 1: LOADING FRAMES DATASET")
    print("=" * 80)
    
    prompts = PromptLoader.load(
        split=args.dataset_split,
        tsv_path=args.tsv_path
    )
    
    print(f"✅ Loaded {len(prompts)} prompts from FRAMES dataset")
    
    # ===================================================================
    # STEP 2: EXTRACT WIKIPEDIA URLs
    # ===================================================================
    print(f"\n{'=' * 80}")
    print("STEP 2: EXTRACTING WIKIPEDIA URLs")
    print("=" * 80)
    
    urls, excluded_urls = extract_wikipedia_urls(
        prompts,
        max_urls=args.max_urls,
        filter_non_articles=True
    )
    
    if excluded_urls:
        num_excluded = len(set(excluded_urls))
        print(f"ℹ️  Excluded {num_excluded} non-article pages (Categories, Files, Templates, etc.)")
    
    if not urls:
        print("❌ No URLs found to process. Exiting.")
        return 1
    
    print(f"✅ Found {len(urls)} unique Wikipedia URLs to download")
    
    # ===================================================================
    # STEP 3: DOWNLOAD WIKIPEDIA ARTICLES
    # ===================================================================
    print(f"\n{'=' * 80}")
    print("STEP 3: DOWNLOADING WIKIPEDIA ARTICLES")
    print("=" * 80)
    
    extractor = WikipediaExtractor(language='en')
    downloader = WikipediaDownloader(extractor, workers=args.download_workers)
    
    download_stats = downloader.download_articles(
        urls=urls,
        output_dir=output_dir,
        verbose=True
    )
    
    # Check if we have any successful downloads
    if download_stats['successful'] == 0:
        print("\n❌ No articles were successfully downloaded. Exiting.")
        return 1
    
    print(f"\n✅ Successfully downloaded {download_stats['successful']} articles")
    
    # ===================================================================
    # STEP 4: CREATE PASSAGES (if requested)
    # ===================================================================
    if args.create_chunks:
        print(f"\n{'=' * 80}")
        print("STEP 4: CREATING PASSAGES FROM ARTICLES")
        print("=" * 80)
        
        # Create chunker
        # Use CPU by default when using multiple workers to avoid GPU conflicts
        device = args.chunk_device
        if device is None and args.chunk_workers > 1:
            device = 'cpu'  # Force CPU for multi-worker to avoid GPU contention
            print(f"ℹ️  Using CPU for chunking (multiple workers: {args.chunk_workers})")
        
        chunker = TextChunker(
            use_semantic=args.semantic,
            similarity_threshold=args.similarity_threshold,
            device=device
        )
        
        # Create passage builder
        builder = PassageBuilder(chunker)
        
        # Create filename with chunking parameters
        chunk_method = f"semantic{args.similarity_threshold}" if args.semantic else "simple"
        passages_filename = f"passages_chunk{args.chunk_size}_overlap{args.overlap}_{chunk_method}.json"
        passages_json = output_dir / passages_filename
        
        # Build passages
        passages_stats = builder.create_passages_from_articles(
            articles_dir=output_dir,
            output_json=passages_json,
            max_length=args.chunk_size,
            overlap=args.overlap,
            workers=args.chunk_workers,
            verbose=True
        )
        
        if "error" not in passages_stats:
            print(f"\n{'=' * 80}")
            print("PASSAGES CREATED SUCCESSFULLY")
            print("=" * 80)
            print(f"Total articles: {passages_stats['total_articles']}")
            print(f"Total passages: {passages_stats['total_passages']}")
            print(f"Average passages per article: {passages_stats['avg_passages_per_article']:.1f}")
            
            # Show chunking method
            if passages_stats.get('semantic_chunking'):
                print(f"Chunking method: SEMANTIC (threshold: {passages_stats['similarity_threshold']})")
            else:
                print(f"Chunking method: SIMPLE sentence-based")
            
            print(f"\n✅ Passages saved to: {passages_json}")
            print(f"\nTo create vector store, run:")
            print(f"  python create_vector_store.py --passages {passages_json} --output vector_store.pkl")
        else:
            print(f"\n❌ Error creating passages: {passages_stats['error']}")
            return 1
    else:
        print(f"\n{'=' * 80}")
        print("DOWNLOAD COMPLETE")
        print("=" * 80)
        print(f"\nTo create passages, run again with --create_chunks:")
        print(f"  python {os.path.basename(__file__)} --output_dir {output_dir} --create_chunks")
    
    return 0


if __name__ == "__main__":
    exit(main())
