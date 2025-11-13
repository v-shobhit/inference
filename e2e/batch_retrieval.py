import argparse
import json
import time
import pandas as pd
import numpy as np
from datasets import load_dataset
from retrieve import VectorDB
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import re
from openai import OpenAI
from query_generation_prompts import format_query_generation_prompt


def load_prompts_from_frames(
        split: str = "test", tsv_path: str = None) -> List[Dict[str, Any]]:
    """
    Load prompts from the frames-benchmark dataset.

    Args:
        split: Dataset split to load ('test', 'train', etc.) - used if tsv_path is None
        tsv_path: Path to TSV file. If provided, loads from file instead of HuggingFace

    Returns:
        List of dictionaries containing prompt information
    """
    if tsv_path:
        print(f"Loading frames dataset from TSV: {tsv_path}...")
        df = pd.read_csv(tsv_path, sep='\t')

        prompts = []
        for idx, row in df.iterrows():
            # Handle wiki_links
            wiki_links = row.get('wiki_links', [])
            if pd.isna(wiki_links):
                wiki_links = []
            elif isinstance(wiki_links, str):
                # Parse string representation of list
                if wiki_links.startswith('[') and wiki_links.endswith(']'):
                    import ast
                    try:
                        wiki_links = ast.literal_eval(wiki_links)
                    except BaseException:
                        wiki_links = [wiki_links]
                else:
                    wiki_links = [wiki_links]
            elif not isinstance(wiki_links, list):
                wiki_links = [str(wiki_links)]

            # Handle answers
            answers = row.get('Answer', [])
            if pd.isna(answers):
                answers = []
            elif isinstance(answers, str):
                answers = [answers]
            elif not isinstance(answers, list):
                answers = [str(answers)]

            prompts.append({
                'index': idx,
                'prompt': row['Prompt'],
                'answers': answers,
                'wiki_links': wiki_links
            })

        print(f"Loaded {len(prompts)} prompts from TSV file")
    else:
        print(
            f"Loading frames-benchmark dataset from HuggingFace (split: {split})...")
        ds = load_dataset("google/frames-benchmark", split=split)

        prompts = []
        for idx, example in enumerate(ds):
            # Handle wiki_links - ensure it's a list
            wiki_links = example.get('wiki_links', [])
            if wiki_links is None:
                wiki_links = []
            elif isinstance(wiki_links, str):
                # Try to parse if it's a string representation of a list
                if wiki_links.startswith('[') and wiki_links.endswith(']'):
                    import ast
                    try:
                        wiki_links = ast.literal_eval(wiki_links)
                    except BaseException:
                        wiki_links = [wiki_links]
                else:
                    wiki_links = [wiki_links]
            elif not isinstance(wiki_links, list):
                # If it's some other type, convert to list
                wiki_links = list(wiki_links) if hasattr(
                    wiki_links, '__iter__') else [str(wiki_links)]

            # Handle answers - ensure it's a list
            answers = example.get('Answer', [])
            if answers is None:
                answers = []
            elif isinstance(answers, str):
                answers = [answers]
            elif not isinstance(answers, list):
                answers = list(answers) if hasattr(
                    answers, '__iter__') else [str(answers)]

            prompts.append({
                'index': idx,
                'prompt': example['Prompt'],
                'answers': answers,
                'wiki_links': wiki_links
            })

        print(f"Loaded {len(prompts)} prompts from frames-benchmark")

    return prompts


def generate_rewriter_queries(
    user_question: str,
    k: int,
    rewriter_client: OpenAI,
    model: str,
    summarized_context: str = "None yet - this is the first retrieval step.",
    temperature: float = 0.7,
    max_tokens: int = 500,
    return_io: bool = False
) -> tuple:
    """
    Generate k search queries using a rewriter LLM endpoint.

    Args:
        user_question: The user's original question
        k: Number of queries to generate
        rewriter_client: OpenAI client instance (works with SGLang, vLLM, etc.)
        model: Model name to use
        summarized_context: Summary of previously retrieved documents
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        return_io: If True, return (queries, prompt, raw_output) tuple for debugging

    Returns:
        If return_io is False: List of generated query strings
        If return_io is True: Tuple of (queries, prompt_sent, raw_output)
    """
    # Format the prompt
    prompt = format_query_generation_prompt(
        user_question=user_question,
        k=k,
        summarized_partial_context=summarized_context
    )

    # Call the rewriter LLM
    response = rewriter_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )

    # Extract the generated text
    generated_text = response.choices[0].message.content

    # Store the raw output for debugging (will be returned alongside queries)
    raw_rewriter_output = generated_text

    # Parse the queries from the response
    # Look for numbered lines (1. query, 2. query, etc.)
    queries = []
    lines = generated_text.strip().split('\n')

    for line in lines:
        line = line.strip()
        # Match patterns like "1. query", "1) query", or just numbered lines
        match = re.match(r'^\d+[\.\)]\s*(.+)$', line)
        if match:
            query = match.group(1).strip()
            # Remove any quotes around the query
            query = query.strip('"').strip("'")
            if query:
                queries.append(query)

    # If parsing failed, try to split by lines and take non-empty ones
    if len(queries) < k:
        queries = [line.strip().strip('"').strip("'")
                   for line in lines
                   if line.strip() and not line.strip().startswith('[')]
        # Filter out lines that look like instructions or metadata
        queries = [q for q in queries if len(q) > 10 and '?' not in q[:20]]

    # Return up to k queries
    final_queries = queries[:k] if queries else [
        user_question]  # Fallback to original question

    if return_io:
        return (final_queries, prompt, raw_rewriter_output)
    else:
        return final_queries


def retrieve_and_rerank(
    vector_store: VectorDB,
    query: str,
    top_k: int = 10,
    use_reranker: bool = True,
    top_p: float = None
) -> tuple:
    """
    Perform retrieval and optionally reranking for a single query.

    Args:
        vector_store: Initialized VectorDB instance
        query: Query string
        top_k: Number of results to retrieve
        use_reranker: Whether to use reranker (default: True)
        top_p: If specified, use top-p filtering instead of top_k. Takes chunks until cumulative probability >= top_p

    Returns:
        Tuple of (retrieval_results, reranked_results, lookup_time, rerank_time, num_chunks_kept)
    """
    # Retrieval
    tic = time.time()
    results = vector_store.lookup(query, k=top_k)
    lookup_time = time.time() - tic

    # Reranking (optional)
    rerank_time = 0.0
    if use_reranker:
        top_k_passages = [result.page_content for result in results]
        tic = time.time()
        reranked_results = vector_store.rerank(query, top_k_passages)
        rerank_time = time.time() - tic

        # Normalize scores to sum to 1 (convert to probabilities)
        scores = [score for _, score in reranked_results]
        # Use softmax for better numerical stability
        scores_array = np.array(scores, dtype=np.float64)
        # Subtract max for numerical stability
        exp_scores = np.exp(scores_array - np.max(scores_array))
        normalized_scores = exp_scores / exp_scores.sum()

        # Update results with normalized scores
        passages = [passage for passage, _ in reranked_results]
        reranked_results = list(zip(passages, normalized_scores.tolist()))

        # Apply top-p (nucleus) filtering if specified
        # Keeps highest-ranked chunks until cumulative probability >= top_p
        # Note: top_p > 0 ensures at least 1 chunk is always kept
        if top_p is not None:
            cumulative_prob = 0.0
            filtered_results = []
            for passage, prob in reranked_results:
                filtered_results.append((passage, prob))
                cumulative_prob += prob
                if cumulative_prob >= top_p:
                    break
            reranked_results = filtered_results
    else:
        # No reranking - just return in original retrieval order with uniform
        # scores
        passages = [result.page_content for result in results]
        # Uniform probabilities
        uniform_prob = 1.0 / len(passages)
        reranked_results = [(p, uniform_prob) for p in passages]

        # Apply top-p filtering if specified (though less meaningful without
        # reranking)
        if top_p is not None:
            num_to_keep = max(1, int(top_p * len(passages)))
            reranked_results = reranked_results[:num_to_keep]

    num_chunks_kept = len(reranked_results)

    return results, reranked_results, lookup_time, rerank_time, num_chunks_kept


def rewriter_retrieve_and_rerank(
    vector_store: VectorDB,
    user_question: str,
    rewriter_client: OpenAI,
    rewriter_model: str,
    num_rewriter_queries: int = 3,
    num_rewriter_steps: int = 1,
    top_k: int = 10,
    use_reranker: bool = True,
    top_p: float = None,
    rewriter_temperature: float = 0.7,
    rewriter_max_tokens: int = 500,
    verbose: bool = False,
    save_io: bool = False
) -> tuple:
    """
    Perform rewriter-based retrieval: generate multiple queries with rewriter LLM, then retrieve and rerank.
    Can repeat this process multiple times (num_rewriter_steps).

    Args:
        vector_store: Initialized VectorDB instance
        user_question: Original user question
        rewriter_client: OpenAI client for query generation
        rewriter_model: Model name for rewriter
        num_rewriter_queries: Number of queries to generate with rewriter per step
        num_rewriter_steps: Number of times to repeat the rewrite->retrieve sequence
        top_k: Number of results to retrieve per query
        use_reranker: Whether to use reranker
        top_p: If specified, use top-p filtering
        rewriter_temperature: Temperature for rewriter query generation
        rewriter_max_tokens: Max tokens for rewriter generation
        verbose: Whether to print verbose output
        save_io: Whether to save rewriter input/output for debugging

    Returns:
        If save_io is False:
            Tuple of (all_results, all_reranked_results, total_lookup_time, total_rerank_time,
                     total_rewriter_time, all_generated_queries, num_chunks_kept)
        If save_io is True:
            Tuple of (all_results, all_reranked_results, total_lookup_time, total_rerank_time,
                     total_rewriter_time, all_generated_queries, num_chunks_kept, rewriter_io_list)
    """
    # Initialize tracking variables
    all_results = []
    all_reranked_results = []
    total_lookup_time = 0
    total_rerank_time = 0
    total_rewriter_time = 0
    all_generated_queries = []
    rewriter_io_list = [] if save_io else None

    # Repeat rewrite->retrieve sequence for num_rewriter_steps
    for step in range(num_rewriter_steps):
        if verbose:
            print(f"  === Rewriter Step {step+1}/{num_rewriter_steps} ===")
            print(
                f"  Generating {num_rewriter_queries} queries with rewriter...")

        # Step 1: Generate queries using rewriter
        tic = time.time()
        if save_io:
            step_queries, rewriter_input, rewriter_output = generate_rewriter_queries(
                user_question=user_question,
                k=num_rewriter_queries,
                rewriter_client=rewriter_client,
                model=rewriter_model,
                temperature=rewriter_temperature,
                max_tokens=rewriter_max_tokens,
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
            step_queries = generate_rewriter_queries(
                user_question=user_question,
                k=num_rewriter_queries,
                rewriter_client=rewriter_client,
                model=rewriter_model,
                temperature=rewriter_temperature,
                max_tokens=rewriter_max_tokens,
                return_io=False
            )
        rewriter_time = time.time() - tic
        total_rewriter_time += rewriter_time
        all_generated_queries.extend(step_queries)

        if verbose:
            print(f"  Generated queries in {rewriter_time:.3f}s:")
            for i, q in enumerate(step_queries, 1):
                print(f"    {i}. {q}")

        # Step 2: Retrieve for each generated query in this step
        for query in step_queries:
            results, reranked_results, lookup_time, rerank_time, _ = retrieve_and_rerank(
                vector_store=vector_store,
                query=query,
                top_k=top_k,
                use_reranker=use_reranker,
                top_p=top_p
            )
            all_results.extend(results)
            all_reranked_results.extend(reranked_results)
            total_lookup_time += lookup_time
            total_rerank_time += rerank_time

        if verbose:
            print(
                f"  Step {step+1} retrieved {len(step_queries) * top_k} passages")

    # Step 3: Deduplicate and re-rank all retrieved passages across all steps
    # Use metadata (chunk index) for deduplication, not passage text
    # key: unique_id (chunk index), value: (passage, score, result_obj)
    seen_chunks = {}

    for passage, score in all_reranked_results:
        # Find the corresponding result object with metadata
        result_obj = None
        for r in all_results:
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

    # Apply top-p filtering to deduplicated results if specified
    if top_p is not None and use_reranker:
        # Re-normalize scores to sum to 1
        scores = [score for _, score in deduplicated_results]
        scores_array = np.array(scores, dtype=np.float64)
        exp_scores = np.exp(scores_array - np.max(scores_array))
        normalized_scores = exp_scores / exp_scores.sum()

        # Update with normalized scores
        passages = [passage for passage, _ in deduplicated_results]
        deduplicated_results = list(zip(passages, normalized_scores.tolist()))

        # Apply top-p filtering
        cumulative_prob = 0.0
        filtered_results = []
        for passage, prob in deduplicated_results:
            filtered_results.append((passage, prob))
            cumulative_prob += prob
            if cumulative_prob >= top_p:
                break
        deduplicated_results = filtered_results
    elif top_p is not None and not use_reranker:
        # Without reranker, just take top-p fraction
        num_to_keep = max(1, int(top_p * len(deduplicated_results)))
        deduplicated_results = deduplicated_results[:num_to_keep]

    num_chunks_kept = len(deduplicated_results)

    if verbose:
        print(
            f"  Retrieved {len(all_reranked_results)} total passages across {num_rewriter_steps} steps")
        print(f"  Unique chunks after deduplication: {len(seen_chunks)}")
        dedup_ratio = (len(all_reranked_results) - len(seen_chunks)) / \
            len(all_reranked_results) * 100 if all_reranked_results else 0
        print(
            f"  Deduplication: removed {len(all_reranked_results) - len(seen_chunks)} duplicates ({dedup_ratio:.1f}%)")
        print(f"  Final passages after top-p filtering: {num_chunks_kept}")

    if save_io:
        return (all_results, deduplicated_results, total_lookup_time, total_rerank_time,
                total_rewriter_time, all_generated_queries, num_chunks_kept, rewriter_io_list)
    else:
        return (all_results, deduplicated_results, total_lookup_time, total_rerank_time,
                total_rewriter_time, all_generated_queries, num_chunks_kept)


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
            raise ValueError(
                "--top_p requires reranker. Cannot use with --no_reranker.")
        if not (0 < args.top_p <= 1):
            raise ValueError(
                "--top_p must be between 0 (exclusive) and 1 (inclusive)")

    # Validate rewriter arguments
    if args.num_rewriter_steps > 0:
        if args.rewriter_endpoint is None:
            raise ValueError(
                "--rewriter_endpoint is required when --num_rewriter_steps > 0")
        if args.rewriter_model is None:
            raise ValueError(
                "--rewriter_model is required when --num_rewriter_steps > 0")
        if args.num_rewriter_queries < 1:
            raise ValueError("--num_rewriter_queries must be at least 1")

    # Initialize rewriter client if enabled
    rewriter_client = None
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
        tic = time.time()
        vector_store.ingest(passage_list, passage_metadata)
        toc = time.time()
        print(f"Ingestion completed in {toc - tic:.2f} seconds")

    # Load prompts from frames-benchmark
    prompts = load_prompts_from_frames(
        split=args.dataset_split,
        tsv_path=args.tsv_path)

    if args.num_prompts is not None:
        prompts = prompts[:args.num_prompts]
        print(f"Processing first {len(prompts)} prompts")

    # Process all prompts
    print(f"\n{'='*80}")
    if args.num_rewriter_steps > 0:
        mode_str = f"rewriter retrieval ({args.num_rewriter_steps} steps × {args.num_rewriter_queries} queries)"
        if args.use_reranker:
            mode_str += " + reranking"
    else:
        mode_str = "retrieval + reranking" if args.use_reranker else "retrieval only"
    filtering_str = f" with top-p={args.top_p}" if args.top_p else f" (top_k={args.top_k})"
    print(
        f"Starting batch {mode_str} on {len(prompts)} prompts{filtering_str}")
    print(f"{'='*80}\n")

    all_results = []
    # Collect rewriter I/O for all prompts
    all_rewriter_io = [] if args.save_rewriter_io else None
    total_lookup_time = 0
    total_rerank_time = 0
    total_rewriter_time = 0
    total_chunks_kept = 0

    # Use tqdm for progress tracking
    for i, prompt_data in tqdm(enumerate(prompts), total=len(
            prompts), desc="Processing prompts", unit="prompt"):
        prompt = prompt_data['prompt']

        if args.verbose:
            tqdm.write(
                f"\n[{i+1}/{len(prompts)}] Processing: {prompt[:80]}...")

        # Perform retrieval (direct or with rewriter)
        if args.num_rewriter_steps > 0:
            # Rewriter-based retrieval with query generation
            if args.save_rewriter_io:
                results, reranked_results, lookup_time, rerank_time, rewriter_time, generated_queries, num_chunks_kept, rewriter_io_list = \
                    rewriter_retrieve_and_rerank(
                        vector_store=vector_store,
                        user_question=prompt,
                        rewriter_client=rewriter_client,
                        rewriter_model=args.rewriter_model,
                        num_rewriter_queries=args.num_rewriter_queries,
                        num_rewriter_steps=args.num_rewriter_steps,
                        top_k=args.top_k,
                        use_reranker=args.use_reranker,
                        top_p=args.top_p,
                        rewriter_temperature=args.rewriter_temperature,
                        rewriter_max_tokens=args.rewriter_max_tokens,
                        verbose=args.verbose,
                        save_io=True
                    )
                # Store I/O for this prompt
                all_rewriter_io.append({
                    'prompt_index': prompt_data['index'],
                    'prompt': prompt,
                    'rewriter_steps': rewriter_io_list
                })
            else:
                results, reranked_results, lookup_time, rerank_time, rewriter_time, generated_queries, num_chunks_kept = \
                    rewriter_retrieve_and_rerank(
                        vector_store=vector_store,
                        user_question=prompt,
                        rewriter_client=rewriter_client,
                        rewriter_model=args.rewriter_model,
                        num_rewriter_queries=args.num_rewriter_queries,
                        num_rewriter_steps=args.num_rewriter_steps,
                        top_k=args.top_k,
                        use_reranker=args.use_reranker,
                        top_p=args.top_p,
                        rewriter_temperature=args.rewriter_temperature,
                        rewriter_max_tokens=args.rewriter_max_tokens,
                        verbose=args.verbose,
                        save_io=False
                    )
            total_rewriter_time += rewriter_time

            if args.verbose:
                tqdm.write(f"  Rewriter: {rewriter_time:.3f}s | Lookup: {lookup_time:.3f}s | "
                           f"Rerank: {rerank_time:.3f}s | Chunks kept: {num_chunks_kept}")
        else:
            # Direct retrieval (original behavior)
            results, reranked_results, lookup_time, rerank_time, num_chunks_kept = retrieve_and_rerank(
                vector_store, prompt, args.top_k,
                use_reranker=args.use_reranker,
                top_p=args.top_p
            )
            rewriter_time = 0.0
            generated_queries = []

            if args.verbose:
                tqdm.write(
                    f"  Lookup: {lookup_time:.3f}s | Rerank: {rerank_time:.3f}s | Chunks kept: {num_chunks_kept}")

        total_lookup_time += lookup_time
        total_rerank_time += rerank_time
        total_chunks_kept += num_chunks_kept

        # Properly format ground truth data
        gt_wiki_links = prompt_data.get('wiki_links', [])
        if isinstance(gt_wiki_links, list):
            gt_wiki_links_str = '|'.join(str(link) for link in gt_wiki_links)
        else:
            gt_wiki_links_str = str(gt_wiki_links)

        gt_answers = prompt_data.get('answers', [])
        if isinstance(gt_answers, list):
            gt_answers_str = '|'.join(str(ans) for ans in gt_answers)
        else:
            gt_answers_str = str(gt_answers)

        # Collect arrays of retrieved chunk data AND extract unique URLs
        # IMPORTANT: We use reranked_results (after filtering), not raw results
        ranks = []
        rerank_scores = []
        chunk_indices = []
        article_filenames = []
        article_titles = []
        article_urls = []
        source_urls = []
        passage_lengths = []
        passages = []
        retrieved_wiki_urls = set()  # Track unique URLs from FILTERED results

        for rank, (passage, score) in enumerate(reranked_results, start=1):
            # Find the metadata for this passage
            metadata = None
            for r in results:
                if r.page_content == passage:
                    metadata = r.metadata
                    break

            ranks.append(rank)
            rerank_scores.append(score)
            passages.append(passage)
            chunk_indices.append(
                metadata.get(
                    'index',
                    None) if metadata else None)
            article_filenames.append(
                metadata.get(
                    'article_filename',
                    None) if metadata else None)
            article_titles.append(
                metadata.get(
                    'article_title',
                    None) if metadata else None)

            # Extract URLs
            article_url = metadata.get(
                'article_url', None) if metadata else None
            source_url = metadata.get('source_url', None) if metadata else None
            article_urls.append(article_url)
            source_urls.append(source_url)

            # Add to unique URL set for recall/precision calculation
            if article_url:
                retrieved_wiki_urls.add(article_url)

            passage_lengths.append(
                metadata.get(
                    'passage_length',
                    None) if metadata else None)

        # Convert to pipe-separated string for DataFrame
        retrieved_wiki_urls_str = '|'.join(sorted(retrieved_wiki_urls))

        # Calculate retrieval recall and precision
        # Ground truth: set of wiki links
        gt_wiki_set = set(prompt_data.get('wiki_links', []))
        # Retrieved: set of article URLs
        retrieved_url_set = retrieved_wiki_urls

        # Calculate metrics
        if len(gt_wiki_set) > 0:
            # Number of ground truth articles that were retrieved
            num_correct = len(gt_wiki_set.intersection(retrieved_url_set))
            retrieve_recall = num_correct / len(gt_wiki_set)
        else:
            retrieve_recall = 0.0

        if len(retrieved_url_set) > 0:
            num_correct = len(gt_wiki_set.intersection(retrieved_url_set))
            retrieve_precision = num_correct / len(retrieved_url_set)
        else:
            retrieve_precision = 0.0

        # Create a single entry for this prompt with arrays of all chunks
        result_entry = {
            'prompt_index': prompt_data['index'],
            'prompt': prompt,
            'ground_truth_wiki_links': gt_wiki_links_str,
            'ground_truth_answers': gt_answers_str,
            'retrieved_wiki_urls': retrieved_wiki_urls_str,
            'num_unique_articles': len(retrieved_wiki_urls),
            # Retrieval metrics
            'retrieve_recall': retrieve_recall,
            'retrieve_precision': retrieve_precision,
            # Arrays of retrieved chunk data
            'ranks': ranks,
            'rerank_scores': rerank_scores,
            'chunk_indices': chunk_indices,
            'article_filenames': article_filenames,
            'article_titles': article_titles,
            'article_urls': article_urls,
            'source_urls': source_urls,
            'passage_lengths': passage_lengths,
            'passages': passages,
            # Timing
            'lookup_time': lookup_time,
            'rerank_time': rerank_time,
            'rewriter_time': rewriter_time
        }

        # Add generated queries if rewriter is enabled
        if args.num_rewriter_steps > 0 and generated_queries:
            result_entry['rewriter_queries'] = '|'.join(generated_queries)

        all_results.append(result_entry)

        if args.verbose:
            tqdm.write(f"  Top result: {article_titles[0]}")

    # Create DataFrame
    print(f"\n{'='*80}")
    print("Creating results DataFrame...")
    df = pd.DataFrame(all_results)

    # Save to pickle (main output format)
    print(f"Saving results to {args.output}...")
    df.to_pickle(args.output)
    print(f"✅ Saved {len(df)} rows to {args.output} (pickle format)")

    # Optionally save as CSV
    if args.save_csv:
        csv_output = args.output.replace(
            '.pkl', '.csv').replace(
            '.pickle', '.csv')
        df.to_csv(csv_output, index=False)
        print(f"✅ Also saved to {csv_output} (CSV format)")

    # Optionally save as JSON
    if args.save_json:
        json_output = args.output.replace(
            '.pkl', '.json').replace(
            '.pickle', '.json')
        with open(json_output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"✅ Also saved to {json_output} (JSON format)")

    # Optionally save rewriter I/O
    if args.save_rewriter_io and all_rewriter_io:
        print(f"\nSaving rewriter I/O to {args.save_rewriter_io}...")
        io_df = pd.DataFrame(all_rewriter_io)
        io_df.to_pickle(args.save_rewriter_io)
        print(
            f"✅ Saved {len(io_df)} rewriter I/O records to {args.save_rewriter_io}")

        # Print some diagnostics
        total_steps = sum(len(record['rewriter_steps'])
                          for record in all_rewriter_io)
        print(f"   Total rewriter steps across all prompts: {total_steps}")

        # Check for reasoning tokens
        reasoning_count = 0
        for record in all_rewriter_io:
            for step in record['rewriter_steps']:
                output = step['rewriter_output']
                if any(token in output.lower() for token in [
                       '<channel>', '>analysis<', '<message>', 'we need to']):
                    reasoning_count += 1
                    break
        if reasoning_count > 0:
            print(
                f"   ⚠️  {reasoning_count}/{len(all_rewriter_io)} prompts have reasoning tokens in output")

    # Print summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"Total prompts processed: {len(prompts)}")
    print(f"Total rows in DataFrame: {len(df)}")
    print(f"\nRetrieval Configuration:")
    print(f"  Retriever: {args.retriever_model}")
    print(
        f"  Reranker: {args.reranker_model if args.use_reranker else 'Disabled'}")
    if args.num_rewriter_steps > 0:
        print(f"  Rewriter: {args.rewriter_model}")
        print(f"  Rewriter steps: {args.num_rewriter_steps}")
        print(f"  Queries per step: {args.num_rewriter_queries}")
        print(
            f"  Total queries per prompt: {args.num_rewriter_steps * args.num_rewriter_queries}")
    print(f"  Initial top_k: {args.top_k}")
    if args.top_p:
        print(f"  Top-p filtering: {args.top_p}")
        avg_chunks = total_chunks_kept / len(prompts)
        print(f"  Avg chunks kept per prompt: {avg_chunks:.1f} (dynamic)")
    else:
        if args.num_rewriter_steps > 0:
            print(
                f"  Chunks per prompt: ~{args.top_k * args.num_rewriter_steps * args.num_rewriter_queries} before deduplication (dynamic)")
        else:
            print(f"  Chunks per prompt: {args.top_k} (fixed)")

    # Calculate average unique articles per prompt
    avg_unique_articles = df['num_unique_articles'].mean()
    print(f"\nAvg unique articles per prompt: {avg_unique_articles:.2f}")

    print(f"\nTiming:")
    if args.num_rewriter_steps > 0:
        print(f"  Total rewriter time: {total_rewriter_time:.2f}s")
        print(
            f"  Avg rewriter per prompt: {total_rewriter_time/len(prompts):.3f}s")
    print(f"  Total lookup time: {total_lookup_time:.2f}s")
    print(f"  Total rerank time: {total_rerank_time:.2f}s")
    print(f"  Avg lookup per prompt: {total_lookup_time/len(prompts):.3f}s")
    print(f"  Avg rerank per prompt: {total_rerank_time/len(prompts):.3f}s")
    total_time = total_lookup_time + total_rerank_time
    if args.num_rewriter_steps > 0:
        total_time += total_rewriter_time
    print(f"  Total time: {total_time:.2f}s")

    # Aggregate retrieval metrics
    print(f"\n{'='*80}")
    print("RETRIEVAL PERFORMANCE METRICS")
    print(f"{'='*80}")

    avg_recall = df['retrieve_recall'].mean()
    avg_precision = df['retrieve_precision'].mean()

    # Calculate F1 score (harmonic mean of precision and recall)
    if avg_precision + avg_recall > 0:
        f1_score = 2 * (avg_precision * avg_recall) / \
            (avg_precision + avg_recall)
    else:
        f1_score = 0.0

    # Highlight key metrics
    print(f"\n{'*' * 50}")
    print(f"  Average Recall:    {avg_recall:.4f} ({avg_recall*100:.2f}%)")
    print(
        f"  Average Precision: {avg_precision:.4f} ({avg_precision*100:.2f}%)")
    print(f"  F1 Score:          {f1_score:.4f} ({f1_score*100:.2f}%)")
    print(f"{'*' * 50}")

    # Perfect recall/precision counts
    perfect_recall = (df['retrieve_recall'] == 1.0).sum()
    perfect_precision = (df['retrieve_precision'] == 1.0).sum()
    zero_recall = (df['retrieve_recall'] == 0.0).sum()

    print(
        f"\nPerfect recall (100%):     {perfect_recall}/{len(df)} prompts ({perfect_recall/len(df)*100:.1f}%)")
    print(
        f"Perfect precision (100%):  {perfect_precision}/{len(df)} prompts ({perfect_precision/len(df)*100:.1f}%)")
    print(
        f"Zero recall (0%):          {zero_recall}/{len(df)} prompts ({zero_recall/len(df)*100:.1f}%)")

    # Recall and precision distribution
    print(f"\nRecall Distribution:")
    print(
        f"  0.00 - 0.25: {((df['retrieve_recall'] >= 0.0) & (df['retrieve_recall'] < 0.25)).sum()} prompts")
    print(
        f"  0.25 - 0.50: {((df['retrieve_recall'] >= 0.25) & (df['retrieve_recall'] < 0.50)).sum()} prompts")
    print(
        f"  0.50 - 0.75: {((df['retrieve_recall'] >= 0.50) & (df['retrieve_recall'] < 0.75)).sum()} prompts")
    print(
        f"  0.75 - 1.00: {((df['retrieve_recall'] >= 0.75) & (df['retrieve_recall'] <= 1.00)).sum()} prompts")

    print(f"\nPrecision Distribution:")
    print(
        f"  0.00 - 0.25: {((df['retrieve_precision'] >= 0.0) & (df['retrieve_precision'] < 0.25)).sum()} prompts")
    print(
        f"  0.25 - 0.50: {((df['retrieve_precision'] >= 0.25) & (df['retrieve_precision'] < 0.50)).sum()} prompts")
    print(
        f"  0.50 - 0.75: {((df['retrieve_precision'] >= 0.50) & (df['retrieve_precision'] < 0.75)).sum()} prompts")
    print(
        f"  0.75 - 1.00: {((df['retrieve_precision'] >= 0.75) & (df['retrieve_precision'] <= 1.00)).sum()} prompts")


if __name__ == "__main__":
    main()
