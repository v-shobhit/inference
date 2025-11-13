#!/usr/bin/env python3
"""
Create Vector Store from Passages

This script creates a serialized vector store (pickle file) from a passages JSON file.
The resulting vector store can be loaded quickly for retrieval without re-ingesting passages.

Usage:
    # Create vector store from passages
    python create_vector_store.py --passages passages.json --output vector_store.pkl

    # Specify custom models
    python create_vector_store.py --passages passages.json --output vector_store.pkl \
        --retriever_model intfloat/e5-base-v2 \
        --reranker_model colbert-ir/colbertv2.0

    # Process only first N passages
    python create_vector_store.py --passages passages.json --output vector_store.pkl \
        --passage_count 1000
"""

import argparse
import json
from pathlib import Path
from vectordb.create import create_vector_store_from_passages


def main():
    parser = argparse.ArgumentParser(
        description="Create a serialized vector store from passages JSON file",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # Create vector store from passages
  python create_vector_store.py --passages passages.json --output vector_store.pkl

  # Use custom models
  python create_vector_store.py \\
      --passages passages.json \\
      --output vector_store.pkl \\
      --retriever_model intfloat/e5-base-v2 \\
      --reranker_model colbert-ir/colbertv2.0

  # Process only first 1000 passages
  python create_vector_store.py \\
      --passages passages.json \\
      --output vector_store.pkl \\
      --passage_count 1000
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--passages",
        type=str,
        required=True,
        help="Path to passages JSON file\n"
             "Expected format: [{{'passage': 'text', 'metadata_key': 'value', ...}}, ...]\n"
             "The 'passage' key is required; all other keys become metadata"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for serialized vector store (e.g., vector_store.pkl)"
    )
    
    # Model arguments
    parser.add_argument(
        "--retriever_model",
        type=str,
        default="intfloat/e5-base-v2",
        help="HuggingFace model for retrieval embeddings (default: intfloat/e5-base-v2)"
    )
    parser.add_argument(
        "--reranker_model",
        type=str,
        default="colbert-ir/colbertv2.0",
        help="Model for reranking (default: colbert-ir/colbertv2.0)\n"
             "Note: Reranker is not loaded during vector store creation,\n"
             "but the model name is stored for consistency"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for embeddings (cuda/cpu)\n"
             "Auto-detects if not specified"
    )
    
    # Processing arguments
    parser.add_argument(
        "--passage_count",
        type=int,
        default=None,
        help="Number of passages to ingest (default: all)\n"
             "Useful for testing with a subset"
    )
    
    # Metadata output
    parser.add_argument(
        "--save_stats",
        type=str,
        default=None,
        help="Save statistics to JSON file (optional)"
    )
    
    args = parser.parse_args()
    
    try:
        # Create vector store
        stats = create_vector_store_from_passages(
            passages_path=args.passages,
            output_path=args.output,
            retriever_model=args.retriever_model,
            reranker_model=args.reranker_model,
            device=args.device,
            passage_count=args.passage_count,
            verbose=True
        )
        
        # Save statistics if requested
        if args.save_stats:
            stats_path = Path(args.save_stats)
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2)
            print(f"\n📊 Statistics saved to: {stats_path}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
