#!/usr/bin/env python3
"""
FRAMES Retrieval Pipeline
Performs retrieval, reranking and rewriting on the FRAMES benchmark dataset.

Usage:
    # Run with config file
    python run_frames_retrieval.py --config config.yml

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

output:
  results: "data/retrieval_results.pkl"
  rewriter_io: "data/rewriter_io.pkl"
```

Set rewriter.enabled: false for direct retrieval without query rewriting.
See config.example.yml for detailed documentation of all options.
    
    # Override specific settings
    python run_frames_retrieval.py --config config.yml --num-prompts 10 --verbose
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


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
    
    Returns:
        Configuration dictionary with sections: data, retrieval, reranker, rewriter, output
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set defaults for optional sections
    if 'rewriter' not in config:
        config['rewriter'] = {'enabled': False}
    if 'output' not in config:
        config['output'] = {}
    
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration and raise errors for missing required fields.
    
    Args:
        config: Configuration dictionary
    """
    # Validate data section
    if 'data' not in config:
        raise ValueError("Config must have 'data' section")
    
    data = config['data']
    if 'vector_store' not in data and 'passages' not in data:
        raise ValueError("Config data section must have either 'vector_store' or 'passages'")
    
    # Validate retrieval section
    if 'retrieval' not in config:
        raise ValueError("Config must have 'retrieval' section")
    
    # Validate top_p in reranker section
    reranker = config.get('reranker', {})
    use_reranker = reranker.get('enabled', True)
    top_p = reranker.get('top_p')
    
    if top_p is not None:
        if not use_reranker:
            raise ValueError("top_p requires reranker to be enabled")
        if not (0 < top_p <= 1):
            raise ValueError("top_p must be between 0 (exclusive) and 1 (inclusive)")
    
    # Validate rewriter section if enabled
    rewriter = config.get('rewriter', {})
    if rewriter.get('enabled', False):
        if 'steps' not in rewriter or rewriter['steps'] < 1:
            raise ValueError("rewriter.steps must be >= 1 when rewriter is enabled")
        if 'endpoint' not in rewriter:
            raise ValueError("rewriter.endpoint is required when rewriter is enabled")
        if 'model' not in rewriter:
            raise ValueError("rewriter.model is required when rewriter is enabled")


def print_config_summary(config: Dict[str, Any]) -> None:
    """Print a summary of the loaded configuration."""
    print(f"\n{'='*80}")
    print("CONFIGURATION SUMMARY")
    print(f"{'='*80}")
    
    # Data section
    data = config['data']
    print("\n[Data]")
    if 'vector_store' in data:
        print(f"  Vector store: {data['vector_store']}")
    else:
        print(f"  Passages: {data['passages']}")
        if 'passage_count' in data:
            print(f"  Passage count: {data['passage_count']}")
    
    if 'tsv_path' in data:
        print(f"  Dataset TSV: {data['tsv_path']}")
    else:
        print(f"  Dataset split: {data.get('dataset_split', 'test')}")
    
    if 'num_prompts' in data:
        print(f"  Num prompts: {data['num_prompts']}")
    
    # Device section (shared by retrieval and reranking)
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Retrieval section
    retrieval = config['retrieval']
    print("\n[Retrieval]")
    print(f"  Model: {retrieval.get('model', 'intfloat/e5-base-v2')}")
    print(f"  Top-k: {retrieval.get('top_k', 10)}")
    
    # Reranker section
    reranker = config.get('reranker', {})
    print("\n[Reranker]")
    print(f"  Enabled: {reranker.get('enabled', True)}")
    if reranker.get('enabled', True):
        print(f"  Model: {reranker.get('model', 'colbert-ir/colbertv2.0')}")
        if 'top_p' in reranker:
            print(f"  Top-p: {reranker['top_p']}")
    
    # Rewriter section
    rewriter = config.get('rewriter', {})
    print("\n[Rewriter]")
    print(f"  Enabled: {rewriter.get('enabled', False)}")
    if rewriter.get('enabled', False):
        print(f"  Endpoint: {rewriter['endpoint']}")
        print(f"  Model: {rewriter['model']}")
        print(f"  Steps: {rewriter['steps']}")
        print(f"  Queries per step: {rewriter.get('queries_per_step', 3)}")
        print(f"  Temperature: {rewriter.get('temperature', 0.7)}")
        print(f"  Max tokens: {rewriter.get('max_tokens', 500)}")
    
    # Output section
    output = config.get('output', {})
    print("\n[Output]")
    print(f"  Results: {output.get('results', 'retrieval_results.pkl')}")
    if output.get('rewriter_io'):
        print(f"  Rewriter I/O: {output['rewriter_io']}")
    print(f"  Save JSON: {output.get('save_json', False)}")
    print(f"  Save CSV: {output.get('save_csv', False)}")
    print(f"  Verbose: {output.get('verbose', False)}")
    
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="FRAMES Retrieval Pipeline with YAML Configuration",
        formatter_class=argparse.RawTextHelpFormatter
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
    
    args = parser.parse_args()
    
    # Load and validate configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    validate_config(config)
    
    # Apply command-line overrides
    if args.num_prompts is not None:
        config['data']['num_prompts'] = args.num_prompts
    if args.verbose:
        config['output']['verbose'] = True
    if args.output:
        config['output']['results'] = args.output
    
    # Print configuration summary
    print_config_summary(config)
    
    # Extract configuration sections
    data_config = config['data']
    retrieval_config = config['retrieval']
    reranker_config = config.get('reranker', {'enabled': True})
    rewriter_config = config.get('rewriter', {'enabled': False})
    output_config = config.get('output', {})
    
    verbose = output_config.get('verbose', False)
    
    # Initialize rewriter client if enabled
    rewriter = None
    if rewriter_config.get('enabled', False):
        print(f"Initializing rewriter client...")
        print(f"  Endpoint: {rewriter_config['endpoint']}")
        print(f"  Model: {rewriter_config['model']}")
        print(f"  Queries per step: {rewriter_config.get('queries_per_step', 3)}")
        print(f"  Number of steps: {rewriter_config['steps']}")
        
        rewriter_client = OpenAI(
            base_url=rewriter_config['endpoint'],
            api_key=rewriter_config.get('api_key', 'EMPTY')
        )
        
        rewriter = QueryRewriter(
            client=rewriter_client,
            model=rewriter_config['model'],
            temperature=rewriter_config.get('temperature', 0.7),
            max_tokens=rewriter_config.get('max_tokens', 500)
        )
    
    # Initialize vector store
    print("Initializing vector store...")
    device = config.get('device')  # Read from top-level device field
    vector_store = VectorDB(
        retriever_model=retrieval_config.get('model', 'intfloat/e5-base-v2'),
        reranker_model=reranker_config.get('model', 'colbert-ir/colbertv2.0'),
        device=device
    )
    
    # Load vector store or ingest passages
    if 'vector_store' in data_config:
        print(f"Loading vector store from {data_config['vector_store']}...")
        vector_store.from_serialized(data_config['vector_store'])
        print("Vector store loaded successfully")
    else:
        print(f"Loading passages from {data_config['passages']}...")
        with open(data_config['passages']) as f:
            passage_data = json.load(f)
        
        passage_list = [p.pop('passage') for p in passage_data]
        passage_metadata = [p for p in passage_data]
        
        if 'passage_count' in data_config:
            passage_list = passage_list[:data_config['passage_count']]
            passage_metadata = passage_metadata[:data_config['passage_count']]
        
        print(f"Ingesting {len(passage_list)} passages...")
        tic = time.time()
        vector_store.ingest(passage_list, passage_metadata)
        toc = time.time()
        print(f"Ingestion completed in {toc - tic:.2f} seconds")
    
    # Load prompts
    prompts = PromptLoader.load(
        split=data_config.get('dataset_split', 'test'),
        tsv_path=data_config.get('tsv_path')
    )
    
    if 'num_prompts' in data_config:
        prompts = prompts[:data_config['num_prompts']]
        print(f"Processing first {len(prompts)} prompts")
    
    # Create retrieval engine
    engine = RetrievalEngine(
        vector_store=vector_store,
        use_reranker=reranker_config.get('enabled', True),
        verbose=verbose
    )
    
    # Create batch processor
    processor = BatchProcessor(
        engine=engine,
        verbose=verbose
    )
    
    # Build processing configuration
    proc_config = {
        'retriever_model': retrieval_config.get('model', 'intfloat/e5-base-v2'),
        'reranker_model': reranker_config.get('model', 'colbert-ir/colbertv2.0'),
        'use_reranker': reranker_config.get('enabled', True),
        'top_k': retrieval_config.get('top_k', 10),
        'top_p': reranker_config.get('top_p'),  # Read from reranker section
        'num_rewriter_steps': rewriter_config.get('steps', 0) if rewriter_config.get('enabled') else 0,
        'num_rewriter_queries': rewriter_config.get('queries_per_step', 3),
        'rewriter_model': rewriter_config.get('model'),
        'rewriter': rewriter,
        'rewriter_io': 'rewriter_io' in output_config
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
    df, rewriter_io_data, stats = processor.process_prompts(prompts, proc_config)
    
    # Save results
    print(f"\n{'='*80}")
    print("Creating results DataFrame...")
    
    output_path = output_config.get('results', 'retrieval_results.pkl')
    formats = ['pickle']
    if output_config.get('save_json', False):
        formats.append('json')
    if output_config.get('save_csv', False):
        formats.append('csv')
    
    ResultsExporter.save(df, output_path, formats=formats, verbose=True)
    
    # Save rewriter I/O if requested
    if 'rewriter_io' in output_config and rewriter_io_data:
        ResultsExporter.save_rewriter_io(
            rewriter_io_data,
            output_config['rewriter_io'],
            verbose=True
        )
    
    # Print summary
    processor.print_summary(df, stats, proc_config)


if __name__ == "__main__":
    main()
