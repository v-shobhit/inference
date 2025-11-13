"""
Batch processing of retrieval tasks.
"""

import pandas as pd
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from .engine import RetrievalEngine, RetrievalResult
from .query_rewriter import QueryRewriter
from .results import ResultsCollector
from .metrics import MetricsCalculator


class BatchProcessor:
    """Orchestrates batch retrieval over multiple prompts."""
    
    def __init__(
        self,
        engine: RetrievalEngine,
        verbose: bool = False
    ):
        """
        Initialize the BatchProcessor.
        
        Args:
            engine: RetrievalEngine instance
            verbose: Whether to print verbose output
        """
        self.engine = engine
        self.verbose = verbose
    
    def process_prompts(
        self,
        prompts: List[Dict[str, Any]],
        config: Dict[str, Any]
    ) -> tuple:
        """
        Process a batch of prompts.
        
        Args:
            prompts: List of prompt dictionaries
            config: Configuration dictionary with retrieval parameters
        
        Returns:
            Tuple of (results_df, rewriter_io_data, stats_dict)
        """
        collector = ResultsCollector()
        all_rewriter_io = [] if config.get('save_rewriter_io') else None
        
        # Statistics tracking
        total_lookup_time = 0.0
        total_rerank_time = 0.0
        total_rewriter_time = 0.0
        total_chunks_kept = 0
        
        # Process each prompt
        for i, prompt_data in tqdm(enumerate(prompts), total=len(prompts), desc="Processing prompts", unit="prompt"):
            prompt = prompt_data['prompt']
            
            if self.verbose:
                tqdm.write(f"\n[{i+1}/{len(prompts)}] Processing: {prompt[:80]}...")
            
            # Perform retrieval
            retrieval_result = self._process_single_prompt(prompt, config)
            
            # Collect rewriter I/O if enabled
            if config.get('save_rewriter_io') and retrieval_result.rewriter_io:
                all_rewriter_io.append({
                    'prompt_index': prompt_data['index'],
                    'prompt': prompt,
                    'rewriter_steps': retrieval_result.rewriter_io
                })
            
            # Add to results collector
            collector.add_result(
                prompt_data=prompt_data,
                retrieval_result=retrieval_result,
                verbose=self.verbose
            )
            
            # Update statistics
            total_lookup_time += retrieval_result.lookup_time
            total_rerank_time += retrieval_result.rerank_time
            total_rewriter_time += retrieval_result.rewriter_time
            total_chunks_kept += retrieval_result.num_chunks_kept
            
            if self.verbose:
                if retrieval_result.rewriter_time > 0:
                    tqdm.write(f"  Rewriter: {retrieval_result.rewriter_time:.3f}s | "
                              f"Lookup: {retrieval_result.lookup_time:.3f}s | "
                              f"Rerank: {retrieval_result.rerank_time:.3f}s | "
                              f"Chunks kept: {retrieval_result.num_chunks_kept}")
                else:
                    tqdm.write(f"  Lookup: {retrieval_result.lookup_time:.3f}s | "
                              f"Rerank: {retrieval_result.rerank_time:.3f}s | "
                              f"Chunks kept: {retrieval_result.num_chunks_kept}")
        
        # Create DataFrame
        df = collector.to_dataframe()
        
        # Compile statistics
        stats = {
            'num_prompts': len(prompts),
            'total_lookup_time': total_lookup_time,
            'total_rerank_time': total_rerank_time,
            'total_rewriter_time': total_rewriter_time,
            'total_chunks_kept': total_chunks_kept,
            'avg_lookup_time': total_lookup_time / len(prompts),
            'avg_rerank_time': total_rerank_time / len(prompts),
            'avg_rewriter_time': total_rewriter_time / len(prompts) if total_rewriter_time > 0 else 0,
            'avg_chunks_kept': total_chunks_kept / len(prompts) if prompts else 0
        }
        
        return df, all_rewriter_io, stats
    
    def _process_single_prompt(
        self,
        prompt: str,
        config: Dict[str, Any]
    ) -> RetrievalResult:
        """
        Process a single prompt with the configured retrieval strategy.
        
        Args:
            prompt: The prompt/question to process
            config: Configuration dictionary
        
        Returns:
            RetrievalResult
        """
        # Check if using rewriter
        if config.get('num_rewriter_steps', 0) > 0:
            # Rewriter-based retrieval
            return self.engine.retrieve_with_rewriter(
                user_question=prompt,
                rewriter=config['rewriter'],
                num_rewriter_queries=config.get('num_rewriter_queries', 3),
                num_rewriter_steps=config.get('num_rewriter_steps', 1),
                top_k=config.get('top_k', 10),
                top_p=config.get('top_p'),
                save_io=config.get('save_rewriter_io', False)
            )
        else:
            # Direct retrieval
            return self.engine.retrieve(
                query=prompt,
                top_k=config.get('top_k', 10),
                top_p=config.get('top_p')
            )
    
    def print_summary(
        self,
        df: pd.DataFrame,
        stats: Dict[str, Any],
        config: Dict[str, Any]
    ) -> None:
        """
        Print a summary of the batch processing results.
        
        Args:
            df: Results DataFrame
            stats: Statistics dictionary
            config: Configuration dictionary
        """
        print(f"\n{'='*80}")
        print("SUMMARY STATISTICS")
        print(f"{'='*80}")
        print(f"Total prompts processed: {stats['num_prompts']}")
        print(f"Total rows in DataFrame: {len(df)}")
        
        print(f"\nRetrieval Configuration:")
        print(f"  Retriever: {config.get('retriever_model', 'N/A')}")
        print(f"  Reranker: {config.get('reranker_model', 'Disabled') if config.get('use_reranker', True) else 'Disabled'}")
        
        if config.get('num_rewriter_steps', 0) > 0:
            print(f"  Rewriter: {config.get('rewriter_model', 'N/A')}")
            print(f"  Rewriter steps: {config.get('num_rewriter_steps')}")
            print(f"  Queries per step: {config.get('num_rewriter_queries')}")
            print(f"  Total queries per prompt: {config.get('num_rewriter_steps') * config.get('num_rewriter_queries')}")
        
        print(f"  Initial top_k: {config.get('top_k', 10)}")
        
        if config.get('top_p'):
            print(f"  Top-p filtering: {config.get('top_p')}")
            print(f"  Avg chunks kept per prompt: {stats['avg_chunks_kept']:.1f} (dynamic)")
        else:
            if config.get('num_rewriter_steps', 0) > 0:
                expected = config.get('top_k', 10) * config.get('num_rewriter_steps', 1) * config.get('num_rewriter_queries', 3)
                print(f"  Chunks per prompt: ~{expected} before deduplication (dynamic)")
            else:
                print(f"  Chunks per prompt: {config.get('top_k', 10)} (fixed)")
        
        # Average unique articles per prompt
        avg_unique_articles = df['num_unique_articles'].mean()
        print(f"\nAvg unique articles per prompt: {avg_unique_articles:.2f}")
        
        # Timing
        print(f"\nTiming:")
        if stats['total_rewriter_time'] > 0:
            print(f"  Total rewriter time: {stats['total_rewriter_time']:.2f}s")
            print(f"  Avg rewriter per prompt: {stats['avg_rewriter_time']:.3f}s")
        print(f"  Total lookup time: {stats['total_lookup_time']:.2f}s")
        print(f"  Total rerank time: {stats['total_rerank_time']:.2f}s")
        print(f"  Avg lookup per prompt: {stats['avg_lookup_time']:.3f}s")
        print(f"  Avg rerank per prompt: {stats['avg_rerank_time']:.3f}s")
        
        total_time = stats['total_lookup_time'] + stats['total_rerank_time'] + stats['total_rewriter_time']
        print(f"  Total time: {total_time:.2f}s")
        
        # Aggregate retrieval metrics
        print(f"\n{'='*80}")
        print("RETRIEVAL PERFORMANCE METRICS")
        print(f"{'='*80}")
        
        metrics = MetricsCalculator.aggregate_metrics(df)
        
        # Highlight key metrics
        print(f"\n{'*' * 50}")
        print(f"  Average Recall:    {metrics['avg_recall']:.4f} ({metrics['avg_recall']*100:.2f}%)")
        print(f"  Average Precision: {metrics['avg_precision']:.4f} ({metrics['avg_precision']*100:.2f}%)")
        print(f"  F1 Score:          {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
        print(f"{'*' * 50}")
        
        # Perfect/zero counts
        print(f"\nPerfect recall (100%):     {metrics['perfect_recall_count']}/{len(df)} prompts ({metrics['perfect_recall_pct']:.1f}%)")
        print(f"Perfect precision (100%):  {metrics['perfect_precision_count']}/{len(df)} prompts ({metrics['perfect_precision_pct']:.1f}%)")
        print(f"Zero recall (0%):          {metrics['zero_recall_count']}/{len(df)} prompts ({metrics['zero_recall_pct']:.1f}%)")
        
        # Distribution analysis
        recall_dist = MetricsCalculator.distribution_analysis(df['retrieve_recall'].tolist())
        precision_dist = MetricsCalculator.distribution_analysis(df['retrieve_precision'].tolist())
        
        print(f"\nRecall Distribution:")
        for bin_range, count in recall_dist.items():
            print(f"  {bin_range}: {count} prompts")
        
        print(f"\nPrecision Distribution:")
        for bin_range, count in precision_dist.items():
            print(f"  {bin_range}: {count} prompts")

