import argparse
import json
import time
import pandas as pd
from retrieve import VectorDB

# Taken below from frames: https://huggingface.co/datasets/google/frames-benchmark
DEFAULT_QUERY = "Who won the French Open Mens Singles tournament the year that New York City FC won their first MLS Cup title?"

def load_queries_from_frames_tsv(tsv_path, num_queries):
    """Load the first num_queries from the frames dataset TSV file."""
    try:
        df = pd.read_csv(tsv_path, sep='\t')
        if 'Prompt' not in df.columns:
            raise ValueError(f"'Prompt' column not found in {tsv_path}. Available columns: {list(df.columns)}")
        
        # Get the first num_queries rows
        queries = df['Prompt'].head(num_queries).tolist()
        print(f"Loaded {len(queries)} queries from {tsv_path}")
        return queries
    except Exception as e:
        print(f"Error loading queries from {tsv_path}: {e}")
        return None


if __name__ == "__main__":
    args = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    args.add_argument("--passages", type=str, default=None, help="Path to the JSON array file with passages\n"
                        "'passage' will be the passage text\n"
                        "all other keys will be metadata\n"
                        "Example: [{'index': int, 'pdf_filename': str, 'passage': str}]\n"
                        "Ignored if --vector_store is provided")

    args.add_argument("--passage_count", type=int, default=None, help="Number of passages to ingest from --passages file, defaults to all")
    args.add_argument("--vector_store", type=str, default=None, help="Path to the vector store file\n"
                        "If provided, --passages will be ignored")

    args.add_argument("--query", type=str, default=DEFAULT_QUERY, help="Query to search for (ignored if --num_queries is provided)")
    args.add_argument("--num_queries", type=int, default=None, help="Number of queries to process from frames dataset TSV file")
    args.add_argument("--frames_tsv", type=str, default="data/frames_dataset.tsv", help="Path to frames dataset TSV file")

    args.add_argument("--retriever_model", type=str, default="intfloat/e5-base-v2")
    args.add_argument("--reranker_model", type=str, default="colbert-ir/colbertv2.0", help="Model to use for reranking")

    args.add_argument("--top_k", type=int, default=10)
    args = args.parse_args()

    assert (args.vector_store is None) != (args.passages is None), "Exactly one of --vector_store or --passages must be provided"
    vector_store = VectorDB(retriever_model=args.retriever_model, reranker_model=args.reranker_model)

    # Load vector store or ingest passages
    if args.vector_store:
        vector_store.from_serialized(args.vector_store)
    else:
        passage_data = json.load(open(args.passages))
        passage_list = [p.pop('passage') for p in passage_data]
        passage_metadata = [p for p in passage_data] # All keys except 'passage' are metadata

        if args.passage_count is not None:
            passage_list = passage_list[:args.passage_count]
            passage_metadata = passage_metadata[:args.passage_count]

        print(f"Ingesting {len(passage_list)} passages from {args.passages}")
        tic = time.time()
        vector_store.ingest(passage_list, passage_metadata)
        toc = time.time()
        print(f"Ingestion of {len(passage_list)} passages took {toc - tic} seconds")

    # Determine queries to process
    if args.num_queries is not None:
        queries = load_queries_from_frames_tsv(args.frames_tsv, args.num_queries)
        if queries is None:
            print("Failed to load queries from TSV file. Exiting.")
            exit(1)
    else:
        queries = [args.query]

    # Process each query
    all_results = []
    for i, query in enumerate(queries):
        print(f"\n{'='*80}")
        print(f"PROCESSING QUERY {i+1}/{len(queries)}")
        print(f"{'='*80}")
        print(f"Query: {query}\n")

        # Lookup
        print(f"Looking up top-{args.top_k} passages...")
        tic = time.time()
        results = vector_store.lookup(query, k=args.top_k)
        toc = time.time()
        lookup_time = toc - tic
        print(f"Lookup took {lookup_time:.3f} seconds")

        # Display retrieval results
        print(f"\nTop-{args.top_k} retrieval results:")
        for j, result in enumerate(results):
            print(f"{j+1}. {result.metadata}")
        
        # Rerank
        top_k_passages = [result.page_content for result in results]
        print(f"\nReranking {len(results)} passages...")
        tic = time.time()
        reranked_results = vector_store.rerank(query, top_k_passages)
        toc = time.time()
        rerank_time = toc - tic
        print(f"Reranking took {rerank_time:.3f} seconds")

        # Display reranking results
        print(f"\nReranked results:")
        reranked_with_metadata = []
        for j, (passage, score) in enumerate(reranked_results):
            for r in results:
                if r.page_content == passage:
                    metadata = r.metadata
                    break
            print(f"{j+1}. Score: {score:.4f} | {metadata}")
            reranked_with_metadata.append((metadata, score, passage))

        # Store results for this query
        query_result = {
            'query_index': i,
            'query': query,
            'lookup_time': lookup_time,
            'rerank_time': rerank_time,
            'retrieval_results': [r.metadata for r in results],
            'reranked_results': reranked_with_metadata
        }
        all_results.append(query_result)

    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY - Processed {len(queries)} queries")
    print(f"{'='*80}")
    total_lookup_time = sum(r['lookup_time'] for r in all_results)
    total_rerank_time = sum(r['rerank_time'] for r in all_results)
    print(f"Total lookup time: {total_lookup_time:.3f} seconds")
    print(f"Total rerank time: {total_rerank_time:.3f} seconds")
    print(f"Average lookup time per query: {total_lookup_time/len(queries):.3f} seconds")
    print(f"Average rerank time per query: {total_rerank_time/len(queries):.3f} seconds")