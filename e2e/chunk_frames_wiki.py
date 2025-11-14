#!/usr/bin/env python3
"""
Create Chunked Passages from Wikipedia Articles

Processes downloaded Wikipedia articles and creates chunked passages
for RAG retrieval. Supports both simple sentence-based chunking and
semantic chunking using embeddings.

Requirements:
    pip install tqdm
    pip install spacy sentence-transformers  # For semantic chunking
    python -m spacy download en_core_web_sm  # For semantic chunking

Usage:
    # Simple sentence-based chunking (fast, no GPU needed)
    python chunk_frames_wiki.py --articles-dir wiki_articles \\
        --output passages.json --no-semantic

    # Semantic chunking (recommended, better quality)
    python chunk_frames_wiki.py --articles-dir wiki_articles \\
        --output passages.json --semantic --similarity-threshold 0.75

    # Control chunk size and overlap
    python chunk_frames_wiki.py --articles-dir wiki_articles \\
        --output passages.json --chunk-size 512 --overlap 50

    # Parallel processing (2-8 workers recommended for semantic)
    python chunk_frames_wiki.py --articles-dir wiki_articles \\
        --output passages.json --workers 4

    # Force CPU for semantic chunking (useful with many workers)
    python chunk_frames_wiki.py --articles-dir wiki_articles \\
        --output passages.json --semantic --device cpu --workers 8
"""

import argparse
from pathlib import Path
from multiprocessing import cpu_count

# Import from our modular structure
from wikipedia import TextChunker, PassageBuilder


def main():
    parser = argparse.ArgumentParser(
        description='Create chunked passages from Wikipedia articles',
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Input/Output arguments
    parser.add_argument(
        '--articles-dir',
        required=True,
        help='Directory containing downloaded Wikipedia articles (.txt files)'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output JSON file for passages (e.g., passages.json)'
    )

    # Chunking method arguments
    parser.add_argument(
        '--semantic',
        action='store_true',
        default=True,
        help='Use semantic chunking (default: True, requires spacy and sentence-transformers)'
    )
    parser.add_argument(
        '--no-semantic',
        dest='semantic',
        action='store_false',
        help='Disable semantic chunking, use simple sentence-based chunking (faster)'
    )
    parser.add_argument(
        '--similarity-threshold',
        type=float,
        default=0.75,
        help='Semantic similarity threshold 0-1 (default: 0.75). Lower = smaller chunks.'
    )

    # Chunk size arguments
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

    # Performance arguments
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of parallel workers for chunking (default: 1). Recommended: 2-8 for semantic.'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cpu', 'cuda'],
        help='Device for semantic chunking embeddings (default: auto). '
             'Use "cpu" when using many workers to avoid GPU conflicts.'
    )

    args = parser.parse_args()

    # Validate inputs
    articles_dir = Path(args.articles_dir)
    if not articles_dir.exists():
        print(f"❌ Error: Articles directory not found: {articles_dir}")
        print(f"\nDid you run download_frames_wiki.py first?")
        print(f"  python download_frames_wiki.py --output-dir {articles_dir}")
        return 1

    # Count article files
    article_files = list(articles_dir.glob("*.txt"))
    if not article_files:
        print(f"❌ Error: No .txt files found in {articles_dir}")
        print(f"\nMake sure you've downloaded articles first:")
        print(f"  python download_frames_wiki.py --output-dir {articles_dir}")
        return 1

    print(f"Found {len(article_files)} article files to process")

    # Validate worker counts
    max_workers = cpu_count()
    if args.workers > max_workers:
        print(
            f"⚠️  WARNING: {args.workers} workers exceeds CPU count ({max_workers})")
        print(f"   Reducing to {max_workers} workers...")
        args.workers = max_workers
    elif args.workers > 8 and args.semantic:
        print(
            f"⚠️  WARNING: {args.workers} workers with semantic chunking is NOT recommended!")
        print(
            f"   - Memory usage: ~{args.workers * 0.5:.1f}GB (each worker loads embedding models)")
        print(f"   - Model loading contention can cause hanging/deadlock")
        print(f"   - GPU conflicts if using CUDA (will auto-switch to CPU)")
        print(f"   STRONGLY RECOMMENDED: Use 2-8 workers for semantic")
        print(f"   Proceeding anyway... (this may hang or crash)")

    # ===================================================================
    # CREATE CHUNKER
    # ===================================================================
    print(f"\n{'=' * 80}")
    print("CONFIGURATION")
    print("=" * 80)

    # Use CPU by default when using multiple workers to avoid GPU conflicts
    device = args.device
    if device is None and args.workers > 1 and args.semantic:
        device = 'cpu'  # Force CPU for multi-worker to avoid GPU contention
        print(f"Device: CPU (auto-selected for multi-worker)")
    elif device:
        print(f"Device: {device}")
    else:
        print(f"Device: auto-detect")

    print(
        f"Chunking method: {'SEMANTIC' if args.semantic else 'SIMPLE sentence-based'}")
    if args.semantic:
        print(f"Similarity threshold: {args.similarity_threshold}")
    print(f"Chunk size: {args.chunk_size} characters")
    print(f"Overlap: {args.overlap} characters")
    print(f"Workers: {args.workers}")

    chunker = TextChunker(
        use_semantic=args.semantic,
        similarity_threshold=args.similarity_threshold,
        device=device
    )

    # ===================================================================
    # CREATE PASSAGES
    # ===================================================================
    print(f"\n{'=' * 80}")
    print("CREATING PASSAGES")
    print("=" * 80)

    builder = PassageBuilder(chunker)

    passages_stats = builder.create_passages_from_articles(
        articles_dir=articles_dir,
        output_json=args.output,
        max_length=args.chunk_size,
        overlap=args.overlap,
        workers=args.workers,
        verbose=True
    )

    # ===================================================================
    # SUMMARY
    # ===================================================================
    if "error" not in passages_stats:
        print(f"\n{'=' * 80}")
        print("PASSAGES CREATED SUCCESSFULLY")
        print("=" * 80)
        print(f"Total articles processed: {passages_stats['total_articles']}")
        print(f"Total passages created: {passages_stats['total_passages']}")
        print(
            f"Average passages per article: {passages_stats['avg_passages_per_article']:.1f}")

        # Show chunking method
        if passages_stats.get('semantic_chunking'):
            print(
                f"Chunking method: SEMANTIC (threshold: {passages_stats['similarity_threshold']})")
        else:
            print(f"Chunking method: SIMPLE sentence-based")

        print(f"\n✅ Passages saved to: {args.output}")
        print(f"\nNext step: Create vector store for retrieval")
        print(f"  python create_vector_store.py --passages {args.output} \\")
        print(f"      --output vector_store.pkl")

        return 0
    else:
        print(f"\n❌ Error creating passages: {passages_stats['error']}")
        return 1


if __name__ == "__main__":
    exit(main())
