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

import argparse
import yaml
import torch

from retrieval.config import PipelineConfig
from retrieval.pipeline import Pipeline


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

    # Device section
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

    # Build and run pipeline
    print("Building pipeline...")
    pipeline = Pipeline.from_config(config, verbose=True)

    print("\nRunning pipeline...")
    result = pipeline.run(verbose=True)

    # Save results
    print(f"\n{'='*80}")
    print("Saving results...")
    result.save(verbose=True)

    # Print summary
    result.print_summary(pipeline.processor)


if __name__ == "__main__":
    main()
