"""
High-level pipeline for FRAMES retrieval.

Provides a simple interface to run the complete retrieval pipeline,
including loading data, processing prompts, and saving results.
"""

import pandas as pd
from typing import Dict, Any
from dataclasses import dataclass

from .config import PipelineConfig
from .data_loader import PromptLoader
from .builder import PipelineBuilder
from .processor import BatchProcessor
from .query_rewriter import QueryRewriter
from .results import ResultsExporter


@dataclass
class PipelineResult:
    """Results from pipeline execution."""
    dataframe: pd.DataFrame
    io_data: Dict
    stats: Dict
    config: PipelineConfig

    def save(self, verbose: bool = True) -> None:
        """
        Save all results according to configuration.

        Args:
            verbose: Whether to print save messages
        """
        # Determine output formats
        formats = ['pickle']
        if self.config.output.save_json:
            formats.append('json')
        if self.config.output.save_csv:
            formats.append('csv')

        # Save main results
        ResultsExporter.save(
            self.dataframe,
            self.config.output.results,
            formats=formats,
            verbose=verbose
        )

        # Save rewriter I/O if requested
        if self.config.output.rewriter_io and self.io_data.get('rewriter'):
            ResultsExporter.save_rewriter_io(
                self.io_data['rewriter'],
                self.config.output.rewriter_io,
                verbose=verbose
            )

        # Save retriever I/O if requested
        if self.config.output.retriever_io and self.io_data.get('retriever'):
            ResultsExporter.save_retriever_io(
                self.io_data['retriever'],
                self.config.output.retriever_io,
                verbose=verbose
            )

        # Save reranker I/O if requested
        if self.config.output.reranker_io and self.io_data.get('reranker'):
            ResultsExporter.save_reranker_io(
                self.io_data['reranker'],
                self.config.output.reranker_io,
                verbose=verbose
            )

    def print_summary(self, processor: BatchProcessor) -> None:
        """
        Print summary of results.

        Args:
            processor: BatchProcessor instance (for print_summary method)
        """
        # Build processing config for backward compatibility with print_summary
        proc_config = {
            'retriever_model': self.config.retrieval.model,
            'reranker_model': self.config.reranker.model,
            'use_reranker': self.config.reranker.enabled,
            'top_k': self.config.retrieval.top_k,
            'top_p': self.config.reranker.top_p,
            'num_rewriter_steps': self.config.rewriter.steps if self.config.rewriter.enabled else 0,
            'num_rewriter_queries': self.config.rewriter.queries_per_step,
            'rewriter_model': self.config.rewriter.model
        }

        processor.print_summary(self.dataframe, self.stats, proc_config)


class Pipeline:
    """Main retrieval pipeline."""

    def __init__(
        self,
        config: PipelineConfig,
        processor: BatchProcessor,
        rewriter: QueryRewriter = None
    ):
        """
        Initialize pipeline.

        Args:
            config: PipelineConfig instance
            processor: BatchProcessor instance
            rewriter: Optional QueryRewriter instance
        """
        self.config = config
        self.processor = processor
        self.rewriter = rewriter

    @classmethod
    def from_config(cls, config: PipelineConfig,
                    verbose: bool = True) -> 'Pipeline':
        """
        Create pipeline from configuration.

        Args:
            config: PipelineConfig instance
            verbose: Whether to print build messages

        Returns:
            Pipeline instance
        """
        builder = PipelineBuilder(config)
        rewriter, vector_store, engine, processor = builder.build_all(
            verbose=verbose)

        return cls(config=config, processor=processor, rewriter=rewriter)

    def run(self, prompts=None, verbose: bool = True) -> PipelineResult:
        """
        Run the complete pipeline.

        Args:
            prompts: Optional list of prompts. If None, loads from config.
            verbose: Whether to print progress messages

        Returns:
            PipelineResult with dataframe, I/O data, and statistics
        """
        # Load prompts if not provided
        if prompts is None:
            prompts = PromptLoader.load(
                split=self.config.data.dataset_split,
                tsv_path=self.config.data.tsv_path
            )

            if self.config.data.num_prompts:
                prompts = prompts[:self.config.data.num_prompts]
                if verbose:
                    print(f"Processing first {len(prompts)} prompts")

        # Build processing config
        proc_config = self._build_proc_config()

        # Print processing header
        if verbose:
            self._print_processing_header(len(prompts), proc_config)

        # Process prompts
        df, io_data, stats = self.processor.process_prompts(
            prompts, proc_config)

        return PipelineResult(
            dataframe=df,
            io_data=io_data,
            stats=stats,
            config=self.config
        )

    def _build_proc_config(self) -> Dict[str, Any]:
        """
        Build processing configuration for backward compatibility.

        Returns:
            Dictionary with processing parameters
        """
        return {
            'retriever_model': self.config.retrieval.model,
            'reranker_model': self.config.reranker.model,
            'use_reranker': self.config.reranker.enabled,
            'top_k': self.config.retrieval.top_k,
            'top_p': self.config.reranker.top_p,
            'num_rewriter_steps': self.config.rewriter.steps if self.config.rewriter.enabled else 0,
            'num_rewriter_queries': self.config.rewriter.queries_per_step,
            'rewriter_model': self.config.rewriter.model,
            'rewriter': self.rewriter,
            'rewriter_io': self.config.output.rewriter_io is not None,
            'retriever_io': self.config.output.retriever_io is not None,
            'reranker_io': self.config.output.reranker_io is not None
        }

    def _print_processing_header(
            self, num_prompts: int, proc_config: Dict[str, Any]) -> None:
        """
        Print processing header.

        Args:
            num_prompts: Number of prompts to process
            proc_config: Processing configuration dictionary
        """
        print(f"\n{'='*80}")
        num_steps = proc_config['num_rewriter_steps']
        if num_steps > 0:
            mode_str = f"rewriter retrieval ({num_steps} steps × {proc_config['num_rewriter_queries']} queries)"
            if proc_config['use_reranker']:
                mode_str += " + reranking"
        else:
            mode_str = "retrieval + reranking" if proc_config['use_reranker'] else "retrieval only"

        filtering_str = f" with top-p={proc_config['top_p']}" if proc_config[
            'top_p'] else f" (top_k={proc_config['top_k']})"
        print(
            f"Starting batch {mode_str} on {num_prompts} prompts{filtering_str}")
        print(f"{'='*80}\n")
