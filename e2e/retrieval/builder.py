"""
Pipeline builder for constructing retrieval pipeline components.

Centralizes component initialization logic, making it easy to create
a complete retrieval pipeline from configuration.
"""

import json
import time
from typing import Optional
from openai import OpenAI

from vectordb import VectorDB
from .config import PipelineConfig
from .query_rewriter import QueryRewriter
from .engine import RetrievalEngine
from .processor import BatchProcessor


class PipelineBuilder:
    """Builds pipeline components from configuration."""

    def __init__(self, config: PipelineConfig):
        """
        Initialize builder with configuration.

        Args:
            config: PipelineConfig instance
        """
        self.config = config

    def build_rewriter(self) -> Optional[QueryRewriter]:
        """
        Build query rewriter if enabled.

        Returns:
            QueryRewriter instance if enabled, None otherwise
        """
        if not self.config.rewriter.enabled:
            return None

        client = OpenAI(
            base_url=self.config.rewriter.endpoint,
            api_key=self.config.rewriter.api_key
        )

        return QueryRewriter(
            client=client,
            model=self.config.rewriter.model,
            temperature=self.config.rewriter.temperature,
            max_tokens=self.config.rewriter.max_tokens
        )

    def build_vector_store(self, verbose: bool = False) -> VectorDB:
        """
        Build and load vector store.

        Args:
            verbose: Whether to print loading messages

        Returns:
            Loaded VectorDB instance
        """
        if verbose:
            print("Initializing vector store...")

        vector_store = VectorDB(
            retriever_model=self.config.retrieval.model,
            reranker_model=self.config.reranker.model,
            device=self.config.device
        )

        # Load vector store or ingest passages
        if self.config.data.vector_store:
            if verbose:
                print(
                    f"Loading vector store from {self.config.data.vector_store}...")
            vector_store.from_serialized(self.config.data.vector_store)
            if verbose:
                print("Vector store loaded successfully")
        elif self.config.data.passages:
            if verbose:
                print(f"Loading passages from {self.config.data.passages}...")

            with open(self.config.data.passages) as f:
                passage_data = json.load(f)

            passage_list = [p.pop('passage') for p in passage_data]
            passage_metadata = [p for p in passage_data]

            if self.config.data.passage_count:
                passage_list = passage_list[:self.config.data.passage_count]
                passage_metadata = passage_metadata[:
                                                    self.config.data.passage_count]

            if verbose:
                print(f"Ingesting {len(passage_list)} passages...")

            tic = time.time()
            vector_store.ingest(passage_list, passage_metadata)
            toc = time.time()

            if verbose:
                print(f"Ingestion completed in {toc - tic:.2f} seconds")

        return vector_store

    def build_engine(self, vector_store: VectorDB) -> RetrievalEngine:
        """
        Build retrieval engine.

        Args:
            vector_store: VectorDB instance

        Returns:
            RetrievalEngine instance
        """
        return RetrievalEngine(
            vector_store=vector_store,
            use_reranker=self.config.reranker.enabled,
            verbose=self.config.output.verbose
        )

    def build_processor(self, engine: RetrievalEngine) -> BatchProcessor:
        """
        Build batch processor.

        Args:
            engine: RetrievalEngine instance

        Returns:
            BatchProcessor instance
        """
        return BatchProcessor(
            engine=engine,
            verbose=self.config.output.verbose,
            max_workers=self.config.parallel
        )

    def build_all(self, verbose: bool = True) -> tuple:
        """
        Build all pipeline components.

        Args:
            verbose: Whether to print build messages

        Returns:
            Tuple of (rewriter, vector_store, engine, processor)
        """
        # Build rewriter
        rewriter = None
        if self.config.rewriter.enabled and verbose:
            print(f"Initializing rewriter client...")
            print(f"  Endpoint: {self.config.rewriter.endpoint}")
            print(f"  Model: {self.config.rewriter.model}")
            print(
                f"  Queries per step: {self.config.rewriter.queries_per_step}")
            print(f"  Number of steps: {self.config.rewriter.steps}")

        rewriter = self.build_rewriter()

        # Build vector store
        vector_store = self.build_vector_store(verbose=verbose)

        # Build engine
        engine = self.build_engine(vector_store)

        # Build processor
        processor = self.build_processor(engine)

        return rewriter, vector_store, engine, processor
