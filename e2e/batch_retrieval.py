#!/usr/bin/env python3
"""
Batch Retrieval on Frames-Benchmark Dataset

This script performs batch retrieval and reranking on the frames-benchmark dataset.
Supports both direct retrieval and rewriter-based multi-query retrieval.

Usage:
    # Direct retrieval with reranking
    python batch_retrieval.py --vector_store vector_store.pkl --output results.pkl
    
    # Rewriter-based retrieval (3 queries per step, 2 steps)
    python batch_retrieval.py --vector_store vector_store.pkl \\
        --num_rewriter_steps 2 --num_rewriter_queries 3 \\
        --rewriter_endpoint http://localhost:8000/v1 \\
        --rewriter_model gpt-4 \\
        --output results.pkl
"""

import argparse
import json
from openai import OpenAI

# Import from our new modules
from vectordb import VectorDB
from retrieval import (
    PromptLoader,
    QueryRewriter,
    RetrievalEngine,
    ResultsCollector,
    ResultsExporter,
    BatchProcessor
)


def main():
    parser = argparse.ArgumentParser(
        description="Batch retrieval and reranking on frames-benchmark dataset",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Vector store / passages arguments
    parser.add_argument(
        "--passages",
        type=str,
        default=None,
        help="Path to the JSON array file with passages\n"
             "'passage' will be the passage text\n"
             "all other keys will be metadata\n"
             "Example: [{'index': int, 'article_filename': str, 'passage': str}]\n"
             "Ignored if --vector_store is provided"
    )
    parser.add_argument(
        "--passage_count",
        type=int,
        default=None,
        help="Number of passages to ingest from --passages file, defaults to all"
    )
    parser.add_argument(
        "--vector_store",
        type=str,
        default=None,
        help="Path to the vector store file\n"
             "If provided, --passages will be ignored"
    )
    
    # Dataset arguments
    parser.add_argument(
        "--tsv_path",
        type=str,
        default=None,
        help="Path to frames dataset TSV file. If provided, loads from file instead of HuggingFace."
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="test",
        help="Split of frames-benchmark to use (default: test). Ignored if --tsv_path is provided."
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=None,
        help="Number of prompts to process (default: all)"
    )
    
    # Model arguments
    parser.add_argument(
        "--retriever_model",
        type=str,
        default="intfloat/e5-base-v2",
        help="HuggingFace model for retrieval"
    )
    parser.add_argument(
        "--reranker_model",
        type=str,
        default="colbert-ir/colbertv2.0",
        help="Model to use for reranking"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for inference (cuda/cpu). Auto-detects if not specified."
    )
    
    # Retrieval arguments
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of passages to retrieve per query"
    )
    parser.add_argument(
        "--no_reranker",
        dest="use_reranker",
        action="store_false",
        default=True,
        help="Disable reranker (only use retrieval). Default: reranker is enabled."
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Top-p (nucleus) filtering: keep chunks until cumulative probability >= top_p. "
             "Overrides fixed top_k. Example: 0.9 keeps ~90%% of probability mass. "
             "Requires reranker (cannot be used with --no_reranker)."
    )
    
    # Rewriter retrieval arguments
    parser.add_argument(
        "--num_rewriter_steps",
        type=int,
        default=0,
        help="Number of rewriter steps. Each step generates num_rewriter_queries and retrieves passages. "
             "0 = disabled (direct retrieval), 1+ = enabled with specified number of iterations. "
             "Example: --num_rewriter_steps 2 runs rewrite->retrieve twice."
    )
    parser.add_argument(
        "--rewriter_endpoint",
        type=str,
        default=None,
        help="Rewriter endpoint URL for query generation (e.g., http://localhost:8000/v1). "
             "Required if --num_rewriter_steps > 0. Compatible with SGLang, vLLM, etc."
    )
    parser.add_argument(
        "--rewriter_model",
        type=str,
        default=None,
        help="Model name for rewriter query generation. Required if --num_rewriter_steps > 0."
    )
    parser.add_argument(
        "--num_rewriter_queries",
        type=int,
        default=3,
        help="Number of queries to generate with rewriter per step (default: 3)"
    )
    parser.add_argument(
        "--rewriter_temperature",
        type=float,
        default=0.7,
        help="Temperature for rewriter query generation (default: 0.7)"
    )
    parser.add_argument(
        "--rewriter_max_tokens",
        type=int,
        default=500,
        help="Max tokens for rewriter query generation (default: 500)"
    )
    parser.add_argument(
        "--rewriter_api_key",
        type=str,
        default="EMPTY",
        help="API key for rewriter endpoint (default: 'EMPTY' for local endpoints)"
    )
    
    # Output arguments
    parser.add_argument(
        "--output",
        type=str,
        default="retrieval_results.pkl",
        help="Output pickle file path for DataFrame"
    )
    parser.add_argument(
        "--save_json",
        action="store_true",
        help="Also save results as JSON"
    )
    parser.add_argument(
        "--save_csv",
        action="store_true",
        help="Also save results as CSV"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed results for each query"
    )
    parser.add_argument(
        "--save_rewriter_io",
        type=str,
        default=None,
        help="Path to save rewriter input/output pairs for debugging (e.g., rewriter_io.pkl). "
             "If not provided, rewriter I/O will not be saved."
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    assert (args.vector_store is None) != (args.passages is None), \
        "Exactly one of --vector_store or --passages must be provided"
    
    if args.top_p is not None:
        if not args.use_reranker:
            raise ValueError("--top_p requires reranker. Cannot use with --no_reranker.")
        if not (0 < args.top_p <= 1):
            raise ValueError("--top_p must be between 0 (exclusive) and 1 (inclusive)")
    
    # Validate rewriter arguments
    if args.num_rewriter_steps > 0:
        if args.rewriter_endpoint is None:
            raise ValueError("--rewriter_endpoint is required when --num_rewriter_steps > 0")
        if args.rewriter_model is None:
            raise ValueError("--rewriter_model is required when --num_rewriter_steps > 0")
        if args.num_rewriter_queries < 1:
            raise ValueError("--num_rewriter_queries must be at least 1")
    
    # Initialize rewriter client if enabled
    rewriter = None
    if args.num_rewriter_steps > 0:
        print(f"Initializing rewriter client...")
        print(f"  Endpoint: {args.rewriter_endpoint}")
        print(f"  Model: {args.rewriter_model}")
        print(f"  Queries per step: {args.num_rewriter_queries}")
        print(f"  Number of steps: {args.num_rewriter_steps}")
        
        rewriter_client = OpenAI(
            base_url=args.rewriter_endpoint,
            api_key=args.rewriter_api_key
        )
        
        rewriter = QueryRewriter(
            client=rewriter_client,
            model=args.rewriter_model,
            temperature=args.rewriter_temperature,
            max_tokens=args.rewriter_max_tokens
        )
    
    # Initialize vector store
    print("Initializing vector store...")
    vector_store = VectorDB(
        retriever_model=args.retriever_model,
        reranker_model=args.reranker_model,
        device=args.device
    )
    
    # Load vector store or ingest passages
    if args.vector_store:
        print(f"Loading vector store from {args.vector_store}...")
        vector_store.from_serialized(args.vector_store)
        print("Vector store loaded successfully")
    else:
        print(f"Loading passages from {args.passages}...")
        with open(args.passages) as f:
            passage_data = json.load(f)
        
        passage_list = [p.pop('passage') for p in passage_data]
        passage_metadata = [p for p in passage_data]
        
        if args.passage_count is not None:
            passage_list = passage_list[:args.passage_count]
            passage_metadata = passage_metadata[:args.passage_count]
        
        print(f"Ingesting {len(passage_list)} passages...")
        import time
        tic = time.time()
        vector_store.ingest(passage_list, passage_metadata)
        toc = time.time()
        print(f"Ingestion completed in {toc - tic:.2f} seconds")
    
    # Load prompts
    prompts = PromptLoader.load(split=args.dataset_split, tsv_path=args.tsv_path)
    
    if args.num_prompts is not None:
        prompts = prompts[:args.num_prompts]
        print(f"Processing first {len(prompts)} prompts")
    
    # Create retrieval engine
    engine = RetrievalEngine(
        vector_store=vector_store,
        use_reranker=args.use_reranker,
        verbose=args.verbose
    )
    
    # Create batch processor
    processor = BatchProcessor(
        engine=engine,
        verbose=args.verbose
    )
    
    # Build configuration dictionary
    config = {
        'retriever_model': args.retriever_model,
        'reranker_model': args.reranker_model,
        'use_reranker': args.use_reranker,
        'top_k': args.top_k,
        'top_p': args.top_p,
        'num_rewriter_steps': args.num_rewriter_steps,
        'num_rewriter_queries': args.num_rewriter_queries,
        'rewriter_model': args.rewriter_model,
        'rewriter': rewriter,
        'save_rewriter_io': args.save_rewriter_io is not None
    }
    
    # Print header
    print(f"\n{'='*80}")
    if args.num_rewriter_steps > 0:
        mode_str = f"rewriter retrieval ({args.num_rewriter_steps} steps × {args.num_rewriter_queries} queries)"
        if args.use_reranker:
            mode_str += " + reranking"
    else:
        mode_str = "retrieval + reranking" if args.use_reranker else "retrieval only"
    filtering_str = f" with top-p={args.top_p}" if args.top_p else f" (top_k={args.top_k})"
    print(f"Starting batch {mode_str} on {len(prompts)} prompts{filtering_str}")
    print(f"{'='*80}\n")
    
    # Process all prompts
    df, rewriter_io_data, stats = processor.process_prompts(prompts, config)
    
    # Save results
    print(f"\n{'='*80}")
    print("Creating results DataFrame...")
    
    formats = ['pickle']
    if args.save_csv:
        formats.append('csv')
    if args.save_json:
        formats.append('json')
    
    ResultsExporter.save(df, args.output, formats=formats, verbose=True)
    
    # Save rewriter I/O if requested
    if args.save_rewriter_io and rewriter_io_data:
        ResultsExporter.save_rewriter_io(
            rewriter_io_data,
            args.save_rewriter_io,
            verbose=True
        )
    
    # Print summary
    processor.print_summary(df, stats, config)


if __name__ == "__main__":
    main()
