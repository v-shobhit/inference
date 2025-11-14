"""
Retrieval engine for vector search and reranking.
"""

import time
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from vectordb import VectorDB
from .query_rewriter import QueryRewriter


@dataclass
class RetrievalResult:
    """Container for retrieval results and metadata."""
    raw_results: List[Any]  # Raw results from vector store
    reranked_results: List[Tuple[str, float]]  # (passage, score) tuples
    lookup_time: float
    rerank_time: float
    rewriter_time: float = 0.0
    generated_queries: List[str] = field(default_factory=list)
    rewriter_io: List[Dict] = field(default_factory=list)  # For debugging query generation
    retriever_io: List[Dict] = field(default_factory=list)  # For debugging retrieval
    reranker_io: List[Dict] = field(default_factory=list)  # For debugging reranking
    num_chunks_kept: int = field(init=False)
    
    def __post_init__(self):
        self.num_chunks_kept = len(self.reranked_results)


class RetrievalEngine:
    """Handles vector retrieval, reranking, and multi-query retrieval strategies."""
    
    def __init__(
        self,
        vector_store: VectorDB,
        use_reranker: bool = True,
        verbose: bool = False
    ):
        """
        Initialize the RetrievalEngine.
        
        Args:
            vector_store: Initialized VectorDB instance
            use_reranker: Whether to use reranker (default: True)
            verbose: Whether to print verbose output
        """
        self.vector_store = vector_store
        self.use_reranker = use_reranker
        self.verbose = verbose
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        top_p: Optional[float] = None,
        save_retriever_io: bool = False,
        save_reranker_io: bool = False
    ) -> RetrievalResult:
        """
        Perform retrieval and optional reranking for a single query.
        
        Args:
            query: Query string
            top_k: Number of results to retrieve
            top_p: If specified, use top-p filtering instead of top_k
            save_retriever_io: Whether to save retriever input/output
            save_reranker_io: Whether to save reranker input/output
        
        Returns:
            RetrievalResult object with results and timing
        """
        retriever_io = []
        reranker_io = []
        
        # Retrieval
        tic = time.time()
        results = self.vector_store.lookup(query, k=top_k)
        lookup_time = time.time() - tic
        
        # Save retriever I/O if requested
        if save_retriever_io:
            retriever_io.append({
                'query': query,
                'top_k': top_k,
                'num_results': len(results),
                'retrieved_passages': [
                    {
                        'rank': i + 1,
                        'passage': result.page_content,
                        'metadata': result.metadata
                    }
                    for i, result in enumerate(results)
                ]
            })
        
        # Reranking (optional)
        rerank_time = 0.0
        if self.use_reranker:
            top_k_passages = [result.page_content for result in results]
            tic = time.time()
            reranked_results = self.vector_store.rerank(query, top_k_passages)
            rerank_time = time.time() - tic
            
            # Save reranker I/O if requested (before normalization)
            if save_reranker_io:
                reranker_io.append({
                    'query': query,
                    'input_passages': top_k_passages,
                    'output_before_normalization': [
                        {'passage': passage, 'score': score}
                        for passage, score in reranked_results
                    ]
                })
            
            # Normalize scores to probabilities (softmax)
            reranked_results = self._normalize_scores(reranked_results)
            
            # Save normalized results in reranker I/O
            if save_reranker_io and reranker_io:
                reranker_io[-1]['output_after_normalization'] = [
                    {'passage': passage, 'probability': prob}
                    for passage, prob in reranked_results
                ]
            
            # Apply top-p filtering if specified
            if top_p is not None:
                before_top_p = len(reranked_results)
                reranked_results = self._apply_top_p_filtering(reranked_results, top_p)
                if save_reranker_io and reranker_io:
                    reranker_io[-1]['top_p'] = top_p
                    reranker_io[-1]['num_kept_after_top_p'] = len(reranked_results)
                    reranker_io[-1]['num_filtered_by_top_p'] = before_top_p - len(reranked_results)
        else:
            # No reranking - return in original retrieval order with uniform scores
            passages = [result.page_content for result in results]
            uniform_prob = 1.0 / len(passages) if passages else 1.0
            reranked_results = [(p, uniform_prob) for p in passages]
            
            # Apply top-p filtering if specified (less meaningful without reranking)
            if top_p is not None:
                num_to_keep = max(1, int(top_p * len(passages)))
                reranked_results = reranked_results[:num_to_keep]
        
        return RetrievalResult(
            raw_results=results,
            reranked_results=reranked_results,
            lookup_time=lookup_time,
            rerank_time=rerank_time,
            retriever_io=retriever_io,
            reranker_io=reranker_io
        )
    
    def retrieve_with_rewriter(
        self,
        user_question: str,
        rewriter: QueryRewriter,
        num_rewriter_queries: int = 3,
        num_rewriter_steps: int = 1,
        top_k: int = 10,
        top_p: Optional[float] = None,
        save_rewriter_io: bool = False,
        save_retriever_io: bool = False,
        save_reranker_io: bool = False
    ) -> RetrievalResult:
        """
        Perform rewriter-based retrieval with iterative refinement.
        
        For each step:
        1. Generate queries (using context from previous steps if available)
        2. Retrieve and rerank passages for each query
        3. Update context with retrieved passages for next step
        
        Finally, deduplicate and filter all results.
        
        Args:
            user_question: Original user question
            rewriter: QueryRewriter instance for query generation
            num_rewriter_queries: Number of queries to generate per step
            num_rewriter_steps: Number of query generation calls (multiplies total queries)
            top_k: Number of results to retrieve per query
            top_p: If specified, use top-p filtering
            save_rewriter_io: Whether to save rewriter input/output for debugging
            save_retriever_io: Whether to save retriever input/output for debugging
            save_reranker_io: Whether to save reranker input/output for debugging
        
        Returns:
            RetrievalResult with aggregated results from all generated queries
        """
        # Initialize tracking variables
        all_results = []
        all_reranked_results = []
        total_lookup_time = 0.0
        total_rerank_time = 0.0
        total_rewriter_time = 0.0
        all_generated_queries = []
        rewriter_io_list = [] if save_rewriter_io else []
        retriever_io_list = [] if save_retriever_io else []
        reranker_io_list = [] if save_reranker_io else []
        accumulated_context = "None yet - this is the first retrieval step."
        
        # Iterative: For each step, generate queries -> retrieve -> update context
        for step in range(num_rewriter_steps):
            if self.verbose:
                print(f"  === Step {step+1}/{num_rewriter_steps} ===")
                print(f"  Generating {num_rewriter_queries} queries...")
            
            # PHASE 1: Generate queries for this step (using accumulated context)
            tic = time.time()
            if save_rewriter_io:
                step_queries, rewriter_input, rewriter_output = rewriter.generate_queries(
                    user_question=user_question,
                    k=num_rewriter_queries,
                    summarized_context=accumulated_context,
                    return_io=True
                )
                # Store the I/O for this step
                rewriter_io_list.append({
                    'step': step + 1,
                    'rewriter_input': rewriter_input,
                    'rewriter_output': rewriter_output,
                    'parsed_queries': step_queries
                })
            else:
                step_queries = rewriter.generate_queries(
                    user_question=user_question,
                    k=num_rewriter_queries,
                    summarized_context=accumulated_context,
                    return_io=False
                )
            rewriter_time = time.time() - tic
            total_rewriter_time += rewriter_time
            all_generated_queries.extend(step_queries)
            
            if self.verbose:
                print(f"  Generated {len(step_queries)} queries in {rewriter_time:.3f}s")
                for i, q in enumerate(step_queries, 1):
                    print(f"    {i}. {q}")
            
            # PHASE 2: Retrieve and rerank for each generated query in this step
            if self.verbose:
                print(f"  Retrieving for {len(step_queries)} queries...")
            
            step_results = []
            step_raw_results = []
            for i, query in enumerate(step_queries, 1):
                if self.verbose:
                    print(f"    [{i}/{len(step_queries)}] Retrieving: {query[:60]}...")
                
                # Retrieve and rerank for this query
                result = self.retrieve(
                    query=query,
                    top_k=top_k,
                    top_p=top_p,
                    save_retriever_io=save_retriever_io,
                    save_reranker_io=save_reranker_io
                )
                all_results.extend(result.raw_results)
                all_reranked_results.extend(result.reranked_results)
                step_results.extend(result.reranked_results)
                step_raw_results.extend(result.raw_results)
                total_lookup_time += result.lookup_time
                total_rerank_time += result.rerank_time
                
                # Collect retriever I/O
                if save_retriever_io and result.retriever_io:
                    retriever_io_list.extend(result.retriever_io)
                
                # Collect reranker I/O
                if save_reranker_io and result.reranker_io:
                    reranker_io_list.extend(result.reranker_io)
                
                if self.verbose:
                    print(f"      → {len(result.reranked_results)} passages")
            
            # PHASE 3: Update accumulated context for next step
            if step < num_rewriter_steps - 1:  # Don't need to update on last step
                # Deduplicate passages from this step
                deduplicated_step_results = self._deduplicate_results(step_results, step_raw_results)
                
                # Use deduplicated, top-p filtered results as context for next step
                accumulated_context = self._summarize_passages_for_context(deduplicated_step_results)
                if self.verbose:
                    print(f"  Step {step+1} retrieved {len(step_results)} passages → {len(deduplicated_step_results)} unique")
                    print(f"  Updated context for next step with {len(deduplicated_step_results)} passages")
        
        # PHASE 3: Deduplicate and filter all retrieved passages
        if self.verbose:
            print(f"\n  === Deduplication & Filtering Phase ===")
        
        deduplicated_results = self._deduplicate_results(all_reranked_results, all_results)
        
        # Apply top-p filtering to deduplicated results if specified
        if top_p is not None and self.use_reranker:
            # Re-normalize scores to sum to 1
            deduplicated_results = self._normalize_scores(deduplicated_results)
            # Apply top-p filtering
            deduplicated_results = self._apply_top_p_filtering(deduplicated_results, top_p)
        elif top_p is not None and not self.use_reranker:
            # Without reranker, just take top-p fraction
            num_to_keep = max(1, int(top_p * len(deduplicated_results)))
            deduplicated_results = deduplicated_results[:num_to_keep]
        
        if self.verbose:
            print(f"  Total passages retrieved: {len(all_reranked_results)}")
            print(f"  Unique passages after deduplication: {len(deduplicated_results)}")
            if all_reranked_results:
                dedup_ratio = (len(all_reranked_results) - len(deduplicated_results)) / len(all_reranked_results) * 100
                print(f"  Deduplication removed {len(all_reranked_results) - len(deduplicated_results)} duplicates ({dedup_ratio:.1f}%)")
            if top_p:
                print(f"  Final passages after top-p={top_p} filtering: {len(deduplicated_results)}")
        
        return RetrievalResult(
            raw_results=all_results,
            reranked_results=deduplicated_results,
            lookup_time=total_lookup_time,
            rerank_time=total_rerank_time,
            rewriter_time=total_rewriter_time,
            generated_queries=all_generated_queries,
            rewriter_io=rewriter_io_list,
            retriever_io=retriever_io_list,
            reranker_io=reranker_io_list
        )
    
    def _summarize_passages_for_context(
        self,
        passages: List[Tuple[str, float]],
        max_passages: Optional[int] = None
    ) -> str:
        """
        Summarize retrieved passages into a concise context string for the next rewriter step.
        
        Args:
            passages: List of (passage_text, score) tuples (already top-p filtered and deduplicated)
            max_passages: Maximum number of passages to include. If None, uses all passages.
        
        Returns:
            Formatted string with passage summaries
        """
        if not passages:
            return "No relevant documents found in previous step."
        
        # Use all passages by default, or limit if specified
        passages_to_use = passages[:max_passages] if max_passages is not None else passages
        
        # Format as numbered list of passages
        context_lines = []
        for i, (passage_text, score) in enumerate(passages_to_use, 1):
            # Truncate very long passages to keep prompt manageable
            truncated = passage_text[:500] + "..." if len(passage_text) > 500 else passage_text
            context_lines.append(f"[Document {i}] {truncated}")
        
        return "\n\n".join(context_lines)
    
    def _normalize_scores(self, results: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        Normalize scores to probabilities using softmax.
        
        Args:
            results: List of (passage, score) tuples
        
        Returns:
            List of (passage, probability) tuples with normalized scores
        """
        if not results:
            return results
        
        scores = [score for _, score in results]
        scores_array = np.array(scores, dtype=np.float64)
        
        # Softmax for numerical stability
        exp_scores = np.exp(scores_array - np.max(scores_array))
        normalized_scores = exp_scores / exp_scores.sum()
        
        passages = [passage for passage, _ in results]
        return list(zip(passages, normalized_scores.tolist()))
    
    def _apply_top_p_filtering(
        self,
        results: List[Tuple[str, float]],
        top_p: float
    ) -> List[Tuple[str, float]]:
        """
        Apply top-p (nucleus) filtering to keep chunks until cumulative probability >= top_p.
        
        Args:
            results: List of (passage, probability) tuples (should be sorted by score)
            top_p: Cumulative probability threshold (0 < top_p <= 1)
        
        Returns:
            Filtered list of (passage, probability) tuples
        """
        cumulative_prob = 0.0
        filtered_results = []
        
        for passage, prob in results:
            filtered_results.append((passage, prob))
            cumulative_prob += prob
            if cumulative_prob >= top_p:
                break
        
        return filtered_results
    
    def _deduplicate_results(
        self,
        reranked_results: List[Tuple[str, float]],
        raw_results: List[Any]
    ) -> List[Tuple[str, float]]:
        """
        Deduplicate results using metadata (chunk index), keeping highest score for each unique chunk.
        
        Args:
            reranked_results: List of (passage, score) tuples
            raw_results: List of raw result objects with metadata
        
        Returns:
            Deduplicated and sorted list of (passage, score) tuples
        """
        seen_chunks = {}  # key: unique_id (chunk index), value: (passage, score, result_obj)
        
        for passage, score in reranked_results:
            # Find the corresponding result object with metadata
            result_obj = None
            for r in raw_results:
                if r.page_content == passage:
                    result_obj = r
                    break
            
            if result_obj and result_obj.metadata:
                # Use chunk index as unique identifier
                chunk_index = result_obj.metadata.get('index')
                if chunk_index is not None:
                    unique_id = chunk_index
                else:
                    # Fallback: use combination of article_url and passage hash
                    article_url = result_obj.metadata.get('article_url', '')
                    unique_id = f"{article_url}:{hash(passage)}"
            else:
                # No metadata available, fall back to passage hash
                unique_id = hash(passage)
            
            # Keep highest score for each unique chunk
            if unique_id not in seen_chunks or score > seen_chunks[unique_id][1]:
                seen_chunks[unique_id] = (passage, score, result_obj)
        
        # Sort by score (descending)
        deduplicated_results = sorted(
            [(passage, score) for passage, score, _ in seen_chunks.values()],
            key=lambda x: x[1],
            reverse=True
        )
        
        return deduplicated_results

