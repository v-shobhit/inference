"""
Batch processing of retrieval tasks.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from .engine import RetrievalEngine, RetrievalResult
from .query_rewriter import QueryRewriter
from .results import ResultsCollector
from .metrics import MetricsCalculator
from .stats import RetrievalStats


class BatchProcessor:
    """Orchestrates batch retrieval over multiple prompts."""
    
    def __init__(
        self,
        engine: RetrievalEngine,
        verbose: bool = False,
        max_workers: Optional[int] = None
    ):
        """
        Initialize the BatchProcessor.
        
        Args:
            engine: RetrievalEngine instance
            verbose: Whether to print verbose output
            max_workers: Maximum number of worker threads for parallel processing.
                        If None or 1, processes sequentially. If > 1, uses parallel processing.
        """
        self.engine = engine
        self.verbose = verbose
        self.max_workers = max_workers
        self.lock = threading.Lock()  # For thread-safe operations
    
    def process_prompts(
        self,
        prompts: List[Dict[str, Any]],
        config: Dict[str, Any]
    ) -> tuple:
        """
        Process a batch of prompts (sequentially or in parallel).
        
        Args:
            prompts: List of prompt dictionaries
            config: Configuration dictionary with retrieval parameters
        
        Returns:
            Tuple of (results_df, io_data_dict, stats_dict)
            - results_df: DataFrame with retrieval results
            - io_data_dict: Dict with keys 'rewriter', 'retriever', 'reranker' containing I/O data
            - stats_dict: Dictionary with timing and count statistics
        """
        # Determine whether to use parallel processing
        use_parallel = self.max_workers is not None and self.max_workers > 1

        if use_parallel:
            return self._process_prompts_parallel(prompts, config)
        else:
            return self._process_prompts_sequential(prompts, config)

    def _initialize_collectors(self, config: Dict[str, Any]) -> tuple:
        """
        Initialize result collectors and statistics tracking.
        
        Returns:
            Tuple of (collector, all_rewriter_io, all_retriever_io, all_reranker_io, stats)
        """
        collector = ResultsCollector()
        all_rewriter_io = [] if config.get('rewriter_io') else None
        all_retriever_io = [] if config.get('retriever_io') else None
        all_reranker_io = [] if config.get('reranker_io') else None
        stats = RetrievalStats()
        
        return collector, all_rewriter_io, all_retriever_io, all_reranker_io, stats
    
    def _collect_result(
        self,
        prompt_data: Dict[str, Any],
        retrieval_result: 'RetrievalResult',
        collector: ResultsCollector,
        all_rewriter_io: Optional[List],
        all_retriever_io: Optional[List],
        all_reranker_io: Optional[List],
        stats: RetrievalStats,
        config: Dict[str, Any]
    ) -> None:
        """
        Collect results from a single prompt processing.
        
        This method updates the collector, I/O lists, and statistics in place.
        """
        prompt = prompt_data['prompt']
        
        # Collect rewriter I/O if enabled
        if config.get('rewriter_io') and retrieval_result.rewriter_io:
            for step_data in retrieval_result.rewriter_io:
                all_rewriter_io.append({
                    'prompt_index': prompt_data['index'],
                    'prompt': prompt,
                    'step': step_data.get('step', None),
                    'rewriter_input': step_data.get('rewriter_input', ''),
                    'rewriter_output': step_data.get('rewriter_output', ''),
                    'generated_queries': step_data.get('generated_queries', [])
                })
        
        # Collect retriever I/O if enabled
        if config.get('retriever_io') and retrieval_result.retriever_io:
            for retriever_data in retrieval_result.retriever_io:
                all_retriever_io.append({
                    'prompt_index': prompt_data['index'],
                    'prompt': prompt,
                    'query': retriever_data.get('query', ''),
                    'top_k': retriever_data.get('top_k'),
                    'num_results': retriever_data.get('num_results'),
                    'retrieved_passages': retriever_data.get('retrieved_passages', [])
                })
        
        # Collect reranker I/O if enabled
        if config.get('reranker_io') and retrieval_result.reranker_io:
            for reranker_data in retrieval_result.reranker_io:
                all_reranker_io.append({
                    'prompt_index': prompt_data['index'],
                    'prompt': prompt,
                    'query': reranker_data.get('query', ''),
                    'input_passages': reranker_data.get('input_passages', []),
                    'output_before_normalization': reranker_data.get('output_before_normalization', []),
                    'output_after_normalization': reranker_data.get('output_after_normalization', []),
                    'top_p': reranker_data.get('top_p'),
                    'num_kept_after_top_p': reranker_data.get('num_kept_after_top_p'),
                    'num_filtered_by_top_p': reranker_data.get('num_filtered_by_top_p')
                })
        
        # Add to results collector
        collector.add_result(
            prompt_data=prompt_data,
            retrieval_result=retrieval_result,
            verbose=self.verbose
        )
        
        # Update statistics (now handled by RetrievalStats)
        stats.add_result(retrieval_result)
    
    def _log_retrieval_result(self, retrieval_result: 'RetrievalResult') -> None:
        """Log verbose output for a retrieval result."""
        if not self.verbose:
            return
        
        if retrieval_result.rewriter_time > 0:
            tqdm.write(f"  Rewriter: {retrieval_result.rewriter_time:.3f}s | "
                      f"Lookup: {retrieval_result.lookup_time:.3f}s | "
                      f"Rerank: {retrieval_result.rerank_time:.3f}s | "
                      f"Chunks kept: {retrieval_result.num_chunks_kept}")
        else:
            tqdm.write(f"  Lookup: {retrieval_result.lookup_time:.3f}s | "
                      f"Rerank: {retrieval_result.rerank_time:.3f}s | "
                      f"Chunks kept: {retrieval_result.num_chunks_kept}")
    
    def _finalize_results(
        self,
        collector: ResultsCollector,
        all_rewriter_io: Optional[List],
        all_retriever_io: Optional[List],
        all_reranker_io: Optional[List],
        stats: RetrievalStats,
        num_prompts: int
    ) -> tuple:
        """
        Finalize results by creating DataFrame and computing final statistics.
        
        Returns:
            Tuple of (results_df, io_data_dict, final_stats_dict)
        """
        # Create DataFrame
        df = collector.to_dataframe()
        
        # Compile final statistics (stats class handles all calculations)
        final_stats = stats.to_dict()
        
        # Create I/O data dictionary
        io_data = {
            'rewriter': all_rewriter_io,
            'retriever': all_retriever_io,
            'reranker': all_reranker_io
        }
        
        return df, io_data, final_stats
    
    def _process_prompts_sequential(
        self,
        prompts: List[Dict[str, Any]],
        config: Dict[str, Any]
    ) -> tuple:
        """
        Process prompts sequentially (original implementation).
        
        Args:
            prompts: List of prompt dictionaries
            config: Configuration dictionary with retrieval parameters
        
        Returns:
            Tuple of (results_df, io_data_dict, stats_dict)
        """
        # Initialize collectors and statistics
        collector, all_rewriter_io, all_retriever_io, all_reranker_io, stats = \
            self._initialize_collectors(config)
        
        # Process each prompt
        for i, prompt_data in tqdm(enumerate(prompts), total=len(prompts), desc="Processing prompts", unit="prompt"):
            prompt = prompt_data['prompt']
            
            if self.verbose:
                tqdm.write(f"\n[{i+1}/{len(prompts)}] Processing: {prompt[:80]}...")
            
            # Perform retrieval
            retrieval_result = self._process_single_prompt(prompt, config)
            
            # Collect results
            self._collect_result(
                prompt_data, retrieval_result, collector,
                all_rewriter_io, all_retriever_io, all_reranker_io,
                stats, config
            )
            
            # Log verbose output
            self._log_retrieval_result(retrieval_result)
        
        # Finalize and return results
        return self._finalize_results(
            collector, all_rewriter_io, all_retriever_io, all_reranker_io,
            stats, len(prompts)
        )
    
    def _process_prompts_parallel(
        self,
        prompts: List[Dict[str, Any]],
        config: Dict[str, Any]
    ) -> tuple:
        """
        Process prompts in parallel using ThreadPoolExecutor.
        
        Args:
            prompts: List of prompt dictionaries
            config: Configuration dictionary with retrieval parameters
        
        Returns:
            Tuple of (results_df, io_data_dict, stats_dict)
        """
        # Initialize collectors and statistics
        collector, all_rewriter_io, all_retriever_io, all_reranker_io, stats = \
            self._initialize_collectors(config)
        
        # Helper function to process a single prompt and return all data
        def process_prompt_wrapper(prompt_data):
            """Wrapper to process a single prompt and return all necessary data."""
            prompt = prompt_data['prompt']
            retrieval_result = self._process_single_prompt(prompt, config)
            return (prompt_data, retrieval_result)
        
        # Process prompts in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(process_prompt_wrapper, prompt_data): i 
                for i, prompt_data in enumerate(prompts)
            }
            
            # Process results as they complete
            completed_count = 0
            with tqdm(total=len(prompts), desc="Processing prompts", unit="prompt") as pbar:
                for future in as_completed(futures):
                    prompt_idx = futures[future]
                    try:
                        prompt_data, retrieval_result = future.result()
                        prompt = prompt_data['prompt']
                        
                        if self.verbose:
                            tqdm.write(f"\n[{completed_count+1}/{len(prompts)}] Completed: {prompt[:80]}...")
                        
                        # Thread-safe collection of results
                        with self.lock:
                            self._collect_result(
                                prompt_data, retrieval_result, collector,
                                all_rewriter_io, all_retriever_io, all_reranker_io,
                                stats, config
                            )
                        
                        # Log verbose output (outside lock)
                        self._log_retrieval_result(retrieval_result)
                        
                        completed_count += 1
                        pbar.update(1)
                        
                    except Exception as e:
                        tqdm.write(f"Error processing prompt {prompt_idx}: {str(e)}")
                        raise
        
        # Finalize and return results
        return self._finalize_results(
            collector, all_rewriter_io, all_retriever_io, all_reranker_io,
            stats, len(prompts)
        )
    
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
                save_rewriter_io=config.get('rewriter_io', False),
                save_retriever_io=config.get('retriever_io', False),
                save_reranker_io=config.get('reranker_io', False)
            )
        else:
            # Direct retrieval
            return self.engine.retrieve(
                query=prompt,
                top_k=config.get('top_k', 10),
                top_p=config.get('top_p'),
                save_retriever_io=config.get('retriever_io', False),
                save_reranker_io=config.get('reranker_io', False)
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
        
        # Passage count statistics
        print(f"\n{'='*80}")
        print("PASSAGE COUNT STATISTICS (per prompt)")
        print(f"{'='*80}")
        
        def print_stat_line(label, stat_dict):
            """Helper to print statistics in a formatted line."""
            if not stat_dict:
                print(f"{label}: N/A")
                return
            print(f"{label}:")
            print(f"  Mean: {stat_dict['mean']:.1f}  |  "
                  f"Median: {stat_dict['median']:.1f}  |  "
                  f"Q1: {stat_dict['q1']:.1f}  |  "
                  f"Q3: {stat_dict['q3']:.1f}")
            print(f"  Min: {stat_dict['min']:.0f}  |  Max: {stat_dict['max']:.0f}")
        
        print_stat_line("After Retrieval (top_k)", stats['passages_before_rerank'])
        if config.get('use_reranker', True):
            print()
            print_stat_line("After Reranking + Top-p", stats['passages_after_rerank'])
        
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

