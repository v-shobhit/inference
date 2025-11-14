#!/usr/bin/env python3
"""
FRAMES Retrieval Pipeline
Performs iterative query generation, retrieval, and reranking on the FRAMES benchmark dataset.

Flow (per step):
1. Query Generation: Generate multiple search queries (using context from previous steps)
2. Retrieval: Retrieve passages for each generated query
3. Reranking: Rerank and filter retrieved passages
4. Context Update: Use retrieved passages to inform next step

Finally, deduplicate and filter all results across all steps.
"""

CONFIG_EXAMPLE = """
Minimal configuration file (config.yml):
```yaml
data:
  vector_store: "data/wiki_articles_frames/vector_store_passages.pkl"
  tsv_path: "data/frames-benchmark/test.tsv"
  num_prompts: 100

device: "cuda"

retrieval:
  model: "intfloat/e5-base-v2"
  top_k: 20

reranker:
  enabled: true
  model: "colbert-ir/colbertv2.0"
  top_p: 0.4

rewriter:
  enabled: true
  endpoint: "http://localhost:30001/v1"
  model: "meta-llama/Meta-Llama-3.1-8B-Instruct"
  steps: 5
  queries_per_step: 5
  temperature: 0.7
  max_tokens: 1024

parallel:
  max_workers: 4  # Optional: Number of parallel workers (default: 1 = sequential)

output:
  results: "data/retrieval_results.pkl"
  rewriter_io: "data/rewriter_io.pkl"      # Optional: save query generation I/O
  retriever_io: "data/retriever_io.pkl"    # Optional: save retrieval I/O
  reranker_io: "data/reranker_io.pkl"      # Optional: save reranking I/O
```
"""



import argparse
import json
import yaml
import time
import torch
from pathlib import Path
from typing import Dict, Any
from openai import OpenAI

# Import from our new modules
from vectordb import VectorDB
from retrieval import (
    PromptLoader,
    QueryRewriter,
    RetrievalEngine,
    ResultsExporter,
    BatchProcessor
)
from retrieval.config import PipelineConfig


def load_config(config_path: str) -> PipelineConfig:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
    
    Returns:
        PipelineConfig instance with type-safe configuration
    """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create PipelineConfig from dictionary
    config = PipelineConfig.from_dict(config_dict)
    
    # Validate configuration (check files exist in production)
    config.validate(check_files=True)
    
    return config


def print_config_summary(config: PipelineConfig) -> None:
    """Print a summary of the loaded configuration."""
    print(f"\n{'='*80}")
    print("CONFIGURATION SUMMARY")
    print(f"{'='*80}")
    
    # Data section
    print("\n[Data]")
    if config.data.vector_store:
        print(f"  Vector store: {config.data.vector_store}")
    else:
        print(f"  Passages: {config.data.passages}")
        if config.data.passage_count:
            print(f"  Passage count: {config.data.passage_count}")
    
    if config.data.tsv_path:
        print(f"  Dataset TSV: {config.data.tsv_path}")
    else:
        print(f"  Dataset split: {config.data.dataset_split}")
    
    if config.data.num_prompts:
        print(f"  Num prompts: {config.data.num_prompts}")
    
    # Device section (shared by retrieval and reranking)
    device = config.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Retrieval section
    print("\n[Retrieval]")
    print(f"  Model: {config.retrieval.model}")
    print(f"  Top-k: {config.retrieval.top_k}")
    
    # Reranker section
    print("\n[Reranker]")
    print(f"  Enabled: {config.reranker.enabled}")
    if config.reranker.enabled:
        print(f"  Model: {config.reranker.model}")
        if config.reranker.top_p is not None:
            print(f"  Top-p: {config.reranker.top_p}")
    
    # Rewriter section
    print("\n[Rewriter]")
    print(f"  Enabled: {config.rewriter.enabled}")
    if config.rewriter.enabled:
        print(f"  Endpoint: {config.rewriter.endpoint}")
        print(f"  Model: {config.rewriter.model}")
        print(f"  Steps: {config.rewriter.steps}")
        print(f"  Queries per step: {config.rewriter.queries_per_step}")
        print(f"  Temperature: {config.rewriter.temperature}")
        print(f"  Max tokens: {config.rewriter.max_tokens}")
    
    # Parallel section
    print("\n[Parallel Processing]")
    if config.parallel and config.parallel > 1:
        print(f"  Enabled: True")
        print(f"  Max workers: {config.parallel}")
    else:
        print(f"  Enabled: False (sequential processing)")
    
    # Output section
    print("\n[Output]")
    print(f"  Results: {config.output.results}")
    if config.output.rewriter_io:
        print(f"  Rewriter I/O: {config.output.rewriter_io}")
    if config.output.retriever_io:
        print(f"  Retriever I/O: {config.output.retriever_io}")
    if config.output.reranker_io:
        print(f"  Reranker I/O: {config.output.reranker_io}")
    print(f"  Save JSON: {config.output.save_json}")
    print(f"  Save CSV: {config.output.save_csv}")
    print(f"  Verbose: {config.output.verbose}")
    
    print(f"{'='*80}\n")


def main():
    
    parser = argparse.ArgumentParser(
        description="FRAMES Retrieval Pipeline with YAML Configuration",
        epilog=CONFIG_EXAMPLE,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    
    # Optional overrides
    parser.add_argument(
        "--num-prompts",
        type=int,
        help="Override: Number of prompts to process"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Override: Enable verbose output"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Override: Output file path for results"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        help="Override: Number of parallel workers (1 = sequential, >1 = parallel)"
    )
    
    args = parser.parse_args()
    
    # Load and validate configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Apply command-line overrides
    config.apply_overrides(args)
    
    # Print configuration summary
    print_config_summary(config)
    
    # Get verbose flag
    verbose = config.output.verbose
    
    # Initialize rewriter client if enabled
    rewriter = None
    if config.rewriter.enabled:
        print(f"Initializing rewriter client...")
        print(f"  Endpoint: {config.rewriter.endpoint}")
        print(f"  Model: {config.rewriter.model}")
        print(f"  Queries per step: {config.rewriter.queries_per_step}")
        print(f"  Number of steps: {config.rewriter.steps}")
        
        rewriter_client = OpenAI(
            base_url=config.rewriter.endpoint,
            api_key=config.rewriter.api_key
        )
        
        rewriter = QueryRewriter(
            client=rewriter_client,
            model=config.rewriter.model,
            temperature=config.rewriter.temperature,
            max_tokens=config.rewriter.max_tokens
        )
    
    # Initialize vector store
    print("Initializing vector store...")
    device = config.device
    vector_store = VectorDB(
        retriever_model=config.retrieval.model,
        reranker_model=config.reranker.model,
        device=device
    )
    
    # Load vector store or ingest passages
    if config.data.vector_store:
        print(f"Loading vector store from {config.data.vector_store}...")
        vector_store.from_serialized(config.data.vector_store)
        print("Vector store loaded successfully")
    else:
        print(f"Loading passages from {config.data.passages}...")
        with open(config.data.passages) as f:
            passage_data = json.load(f)
        
        passage_list = [p.pop('passage') for p in passage_data]
        passage_metadata = [p for p in passage_data]
        
        if config.data.passage_count:
            passage_list = passage_list[:config.data.passage_count]
            passage_metadata = passage_metadata[:config.data.passage_count]
        
        print(f"Ingesting {len(passage_list)} passages...")
        tic = time.time()
        vector_store.ingest(passage_list, passage_metadata)
        toc = time.time()
        print(f"Ingestion completed in {toc - tic:.2f} seconds")
    
    # Load prompts
    prompts = PromptLoader.load(
        split=config.data.dataset_split,
        tsv_path=config.data.tsv_path
    )
    
    if config.data.num_prompts:
        prompts = prompts[:config.data.num_prompts]
        print(f"Processing first {len(prompts)} prompts")
    
    # Create retrieval engine
    engine = RetrievalEngine(
        vector_store=vector_store,
        use_reranker=config.reranker.enabled,
        verbose=verbose
    )
    
    # Create batch processor
    processor = BatchProcessor(
        engine=engine,
        verbose=verbose,
        max_workers=config.parallel
    )
    
    # Build processing configuration (for backward compatibility with processor)
    proc_config = {
        'retriever_model': config.retrieval.model,
        'reranker_model': config.reranker.model,
        'use_reranker': config.reranker.enabled,
        'top_k': config.retrieval.top_k,
        'top_p': config.reranker.top_p,
        'num_rewriter_steps': config.rewriter.steps if config.rewriter.enabled else 0,
        'num_rewriter_queries': config.rewriter.queries_per_step,
        'rewriter_model': config.rewriter.model,
        'rewriter': rewriter,
        'rewriter_io': config.output.rewriter_io is not None,
        'retriever_io': config.output.retriever_io is not None,
        'reranker_io': config.output.reranker_io is not None
    }
    
    # Print processing header
    print(f"\n{'='*80}")
    num_steps = proc_config['num_rewriter_steps']
    if num_steps > 0:
        mode_str = f"rewriter retrieval ({num_steps} steps × {proc_config['num_rewriter_queries']} queries)"
        if proc_config['use_reranker']:
            mode_str += " + reranking"
    else:
        mode_str = "retrieval + reranking" if proc_config['use_reranker'] else "retrieval only"
    
    filtering_str = f" with top-p={proc_config['top_p']}" if proc_config['top_p'] else f" (top_k={proc_config['top_k']})"
    print(f"Starting batch {mode_str} on {len(prompts)} prompts{filtering_str}")
    print(f"{'='*80}\n")
    
    # Process all prompts
    df, io_data, stats = processor.process_prompts(prompts, proc_config)
    
    # Save results
    print(f"\n{'='*80}")
    print("Creating results DataFrame...")
    
    output_path = config.output.results
    formats = ['pickle']
    if config.output.save_json:
        formats.append('json')
    if config.output.save_csv:
        formats.append('csv')
    
    ResultsExporter.save(df, output_path, formats=formats, verbose=True)
    
    # Save rewriter I/O if requested
    if config.output.rewriter_io and io_data.get('rewriter'):
        ResultsExporter.save_rewriter_io(
            io_data['rewriter'],
            config.output.rewriter_io,
            verbose=True
        )
    
    # Save retriever I/O if requested
    if config.output.retriever_io and io_data.get('retriever'):
        ResultsExporter.save_retriever_io(
            io_data['retriever'],
            config.output.retriever_io,
            verbose=True
        )
    
    # Save reranker I/O if requested
    if config.output.reranker_io and io_data.get('reranker'):
        ResultsExporter.save_reranker_io(
            io_data['reranker'],
            config.output.reranker_io,
            verbose=True
        )
    
    # Print summary
    processor.print_summary(df, stats, proc_config)


if __name__ == "__main__":
    main()
