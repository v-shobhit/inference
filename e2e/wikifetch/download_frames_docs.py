#!/usr/bin/env python3
"""
Download Wikipedia Articles from FRAMES Dataset

Downloads Wikipedia articles referenced in the FRAMES benchmark dataset
using the Wikipedia API and saves them as clean text files.

Requirements:
    pip install wikipedia-api mwparserfromhell requests tqdm datasets pandas

Usage:
    # Download from FRAMES test split (default)
    python download_frames_wiki.py --output-dir wiki_articles

    # From custom TSV file
    python download_frames_wiki.py --tsv-path data/custom.tsv --output-dir wiki_articles

    # Limit number of articles
    python download_frames_wiki.py --output-dir wiki_articles --max-urls 100

    # Control parallelism
    python download_frames_wiki.py --output-dir wiki_articles --workers 20
"""

import argparse
from pathlib import Path

# Import from our modular structure
from retrieval import PromptLoader
from wikifetch import (
    WikipediaExtractor,
    WikipediaDownloader,
    extract_wikipedia_urls
)


def main():
    parser = argparse.ArgumentParser(
        description='Download Wikipedia articles from FRAMES dataset',
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
        default='wiki_articles',
        help='Output directory for article text files (default: wiki_articles)'
    )
    parser.add_argument(
        '--max-urls',
        type=int,
        default=None,
        help='Maximum number of URLs to process (default: all)'
    )

    # Download arguments
    parser.add_argument(
        '--workers',
        type=int,
        default=10,
        help='Number of parallel workers for downloading (default: 10, max recommended: 20)'
    )

    args = parser.parse_args()

    # Validate worker count
    if args.workers > 20:
        print(
            f"⚠️  WARNING: {args.workers} workers may overwhelm Wikipedia API")
        print(f"   Reducing to 20 workers...")
        args.workers = 20

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # ===================================================================
    # STEP 1: LOAD FRAMES DATASET
    # ===================================================================
    print("=" * 80)
    print("LOADING FRAMES DATASET")
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
    print("EXTRACTING WIKIPEDIA URLs")
    print("=" * 80)

    urls, excluded_urls = extract_wikipedia_urls(
        prompts,
        max_urls=args.max_urls,
        filter_non_articles=True
    )

    if excluded_urls:
        num_excluded = len(set(excluded_urls))
        print(
            f"ℹ️  Excluded {num_excluded} non-article pages (Categories, Files, Templates, etc.)")

    if not urls:
        print("❌ No URLs found to process. Exiting.")
        return 1

    print(f"✅ Found {len(urls)} unique Wikipedia URLs to download")

    # ===================================================================
    # STEP 3: DOWNLOAD WIKIPEDIA ARTICLES
    # ===================================================================
    print(f"\n{'=' * 80}")
    print("DOWNLOADING WIKIPEDIA ARTICLES")
    print("=" * 80)

    extractor = WikipediaExtractor(language='en')
    downloader = WikipediaDownloader(extractor, workers=args.workers)

    download_stats = downloader.download_articles(
        urls=urls,
        output_dir=output_dir,
        verbose=True
    )

    # Check if we have any successful downloads
    if download_stats['successful'] == 0:
        print("\n❌ No articles were successfully downloaded. Exiting.")
        return 1

    # ===================================================================
    # SUMMARY
    # ===================================================================
    print(f"\n{'=' * 80}")
    print("DOWNLOAD COMPLETE")
    print("=" * 80)
    print(f"✅ Successfully downloaded {download_stats['successful']} articles")
    print(f"❌ Failed: {download_stats['failed']}")

    # Only show skipped count if it exists in stats
    if 'skipped' in download_stats:
        print(f"⏭️  Skipped (already exist): {download_stats['skipped']}")

    print(f"\nArticles saved to: {output_dir}")
    print(f"\nNext step: Create passages for RAG retrieval")
    print(f"  python chunk_frames_wiki.py --articles-dir {output_dir}")

    return 0


if __name__ == "__main__":
    exit(main())
