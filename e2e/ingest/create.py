"""
Vector store creation utilities.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from .store import VectorDB


def create_vector_store_from_passages(
    passages_path: str,
    output_path: str,
    retriever_model: str = "intfloat/e5-base-v2",
    reranker_model: str = "colbert-ir/colbertv2.0",
    device: str = None,
    passage_count: int = None,
    verbose: bool = True
) -> Dict:
    """
    Create and serialize a vector store from passages JSON.
    
    Args:
        passages_path: Path to passages JSON file
        output_path: Path to save serialized vector store
        retriever_model: HuggingFace model for retrieval embeddings
        reranker_model: Model for reranking (loaded later when needed)
        device: Device to use (cuda/cpu), auto-detects if None
        passage_count: Number of passages to ingest (None = all)
        verbose: Whether to print progress information
    
    Returns:
        Dictionary with statistics
    """
    passages_path = Path(passages_path)
    output_path = Path(output_path)
    
    # Validate input
    if not passages_path.exists():
        raise FileNotFoundError(f"Passages file not found: {passages_path}")
    
    if verbose:
        print("=" * 80)
        print("CREATING VECTOR STORE FROM PASSAGES")
        print("=" * 80)
        print(f"Input passages: {passages_path}")
        print(f"Output vector store: {output_path}")
        print(f"Retriever model: {retriever_model}")
        print(f"Reranker model: {reranker_model}")
    
    # Load passages
    if verbose:
        print(f"\nLoading passages from {passages_path}...")
    load_start = time.time()
    with open(passages_path, 'r', encoding='utf-8') as f:
        passage_data = json.load(f)
    load_time = time.time() - load_start
    
    if not isinstance(passage_data, list):
        raise ValueError("Passages JSON must be an array of objects")
    
    if not passage_data:
        raise ValueError("Passages JSON is empty")
    
    if verbose:
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
        if verbose:
            print(f"ℹ️  Limited to {len(passage_list)} passages")
    
    # Show sample passage
    if verbose:
        print(f"\nSample passage (first 200 chars):")
        print(f"  {passage_list[0][:200]}...")
        if passage_metadata[0]:
            print(f"  Metadata keys: {list(passage_metadata[0].keys())}")
    
    # Initialize vector store
    if verbose:
        print(f"\nInitializing VectorDB...")
    init_start = time.time()
    vector_store = VectorDB(
        retriever_model=retriever_model,
        reranker_model=reranker_model,
        device=device
    )
    init_time = time.time() - init_start
    if verbose:
        print(f"✅ VectorDB initialized in {init_time:.2f}s")
    
    # Ingest passages
    if verbose:
        print(f"\nIngesting {len(passage_list)} passages...")
    ingest_start = time.time()
    vector_store.ingest(passage_list, passage_metadata)
    ingest_time = time.time() - ingest_start
    if verbose:
        print(f"✅ Ingestion completed in {ingest_time:.2f}s")
        print(f"   ({ingest_time/len(passage_list)*1000:.2f}ms per passage)")
    
    # Serialize vector store
    if verbose:
        print(f"\nSerializing vector store to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialize_start = time.time()
    vector_store.serialize(str(output_path))
    serialize_time = time.time() - serialize_start
    
    # Get file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    
    if verbose:
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
    if verbose:
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

