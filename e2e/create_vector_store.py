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
import time
from pathlib import Path
from retrieve import VectorDB


def create_vector_store_from_passages(
    passages_path: str,
    output_path: str,
    retriever_model: str = "intfloat/e5-base-v2",
    reranker_model: str = "colbert-ir/colbertv2.0",
    device: str = None,
    passage_count: int = None
) -> dict:
    """
    Create and serialize a vector store from passages JSON.
    
    Args:
        passages_path: Path to passages JSON file
        output_path: Path to save serialized vector store
        retriever_model: HuggingFace model for retrieval embeddings
        reranker_model: Model for reranking (loaded later when needed)
        device: Device to use (cuda/cpu), auto-detects if None
        passage_count: Number of passages to ingest (None = all)
    
    Returns:
        Dictionary with statistics
    """
    passages_path = Path(passages_path)
    output_path = Path(output_path)
    
    # Validate input
    if not passages_path.exists():
        raise FileNotFoundError(f"Passages file not found: {passages_path}")
    
    print("=" * 80)
    print("CREATING VECTOR STORE FROM PASSAGES")
    print("=" * 80)
    print(f"Input passages: {passages_path}")
    print(f"Output vector store: {output_path}")
    print(f"Retriever model: {retriever_model}")
    print(f"Reranker model: {reranker_model}")
    
    # Load passages
    print(f"\nLoading passages from {passages_path}...")
    load_start = time.time()
    with open(passages_path, 'r', encoding='utf-8') as f:
        passage_data = json.load(f)
    load_time = time.time() - load_start
    
    if not isinstance(passage_data, list):
        raise ValueError("Passages JSON must be an array of objects")
    
    if not passage_data:
        raise ValueError("Passages JSON is empty")
    
    print(f"✅ Loaded {len(passage_data)} passages in {load_time:.2f}s")
    
    # Validate passage format
    if 'passage' not in passage_data[0]:
        raise ValueError("Each passage object must have a 'passage' key")
    
    # Separate passage text from metadata
    # Note: We use .pop() to remove 'passage' key and keep everything else as metadata
    passage_list = [p.pop('passage') for p in passage_data]
    passage_metadata = [p for p in passage_data]
    
    # Limit passages if requested
    if passage_count is not None:
        passage_list = passage_list[:passage_count]
        passage_metadata = passage_metadata[:passage_count]
        print(f"ℹ️  Limited to {len(passage_list)} passages")
    
    # Show sample passage
    print(f"\nSample passage (first 200 chars):")
    print(f"  {passage_list[0][:200]}...")
    if passage_metadata[0]:
        print(f"  Metadata keys: {list(passage_metadata[0].keys())}")
    
    # Initialize vector store
    print(f"\nInitializing VectorDB...")
    init_start = time.time()
    vector_store = VectorDB(
        retriever_model=retriever_model,
        reranker_model=reranker_model,
        device=device
    )
    init_time = time.time() - init_start
    print(f"✅ VectorDB initialized in {init_time:.2f}s")
    
    # Ingest passages
    print(f"\nIngesting {len(passage_list)} passages...")
    ingest_start = time.time()
    vector_store.ingest(passage_list, passage_metadata)
    ingest_time = time.time() - ingest_start
    print(f"✅ Ingestion completed in {ingest_time:.2f}s")
    print(f"   ({ingest_time/len(passage_list)*1000:.2f}ms per passage)")
    
    # Serialize vector store
    print(f"\nSerializing vector store to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialize_start = time.time()
    vector_store.serialize(str(output_path))
    serialize_time = time.time() - serialize_start
    
    # Get file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    
    print(f"✅ Vector store saved successfully!")
    print(f"   File size: {file_size_mb:.2f} MB")
    print(f"   Serialization time: {serialize_time:.2f}s")
    
    # Calculate statistics
    total_time = load_time + init_time + ingest_time + serialize_time
    
    stats = {
        'passages_file': str(passages_path),
        'output_file': str(output_path),
        'num_passages': len(passage_list),
        'retriever_model': retriever_model,
        'reranker_model': reranker_model,
        'file_size_mb': file_size_mb,
        'timings': {
            'load_passages': load_time,
            'init_vectordb': init_time,
            'ingest_passages': ingest_time,
            'serialize': serialize_time,
            'total': total_time
        }
    }
    
    # Print summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print("=" * 80)
    print(f"Total passages ingested: {len(passage_list)}")
    print(f"Vector store size: {file_size_mb:.2f} MB")
    print(f"\nTiming breakdown:")
    print(f"  Load passages:     {load_time:8.2f}s ({load_time/total_time*100:5.1f}%)")
    print(f"  Initialize model:  {init_time:8.2f}s ({init_time/total_time*100:5.1f}%)")
    print(f"  Ingest passages:   {ingest_time:8.2f}s ({ingest_time/total_time*100:5.1f}%)")
    print(f"  Serialize:         {serialize_time:8.2f}s ({serialize_time/total_time*100:5.1f}%)")
    print(f"  {'─' * 40}")
    print(f"  Total:             {total_time:8.2f}s")
    
    print(f"\n{'=' * 80}")
    print("✅ VECTOR STORE CREATED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nTo use this vector store in batch_retrieval.py:")
    print(f"  python batch_retrieval.py --vector_store {output_path} [other args...]")
    
    return stats


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
            passage_count=args.passage_count
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

