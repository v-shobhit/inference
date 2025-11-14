"""
Results collection and export utilities.
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
from .engine import RetrievalResult
from .metrics import MetricsCalculator


class ResultsCollector:
    """Collects and organizes retrieval results."""
    
    def __init__(self):
        """Initialize an empty results collector."""
        self.results = []
    
    def add_result(
        self,
        prompt_data: Dict[str, Any],
        retrieval_result: RetrievalResult,
        verbose: bool = False
    ) -> None:
        """
        Add a retrieval result for a single prompt.
        
        Args:
            prompt_data: Dictionary with 'index', 'prompt', 'answers', 'wiki_links'
            retrieval_result: RetrievalResult from the retrieval engine
            verbose: Whether to print verbose output
        """
        prompt = prompt_data['prompt']
        
        # Format ground truth data
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
        
        # Collect arrays of retrieved chunk data
        ranks = []
        rerank_scores = []
        chunk_indices = []
        article_filenames = []
        article_titles = []
        article_urls = []
        source_urls = []
        passage_lengths = []
        passages = []
        retrieved_wiki_urls = set()  # Track unique URLs from filtered results
        
        for rank, (passage, score) in enumerate(retrieval_result.reranked_results, start=1):
            # Find the metadata for this passage
            metadata = None
            for r in retrieval_result.raw_results:
                if r.page_content == passage:
                    metadata = r.metadata
                    break
            
            ranks.append(rank)
            rerank_scores.append(score)
            passages.append(passage)
            chunk_indices.append(metadata.get('index', None) if metadata else None)
            article_filenames.append(metadata.get('article_filename', None) if metadata else None)
            article_titles.append(metadata.get('article_title', None) if metadata else None)
            
            # Extract URLs
            article_url = metadata.get('article_url', None) if metadata else None
            source_url = metadata.get('source_url', None) if metadata else None
            article_urls.append(article_url)
            source_urls.append(source_url)
            
            # Add to unique URL set for recall/precision calculation
            if article_url:
                retrieved_wiki_urls.add(article_url)
            
            passage_lengths.append(metadata.get('passage_length', None) if metadata else None)
        
        # Convert to pipe-separated string for DataFrame
        retrieved_wiki_urls_str = '|'.join(sorted(retrieved_wiki_urls))
        
        # Calculate retrieval metrics
        gt_wiki_set = set(prompt_data.get('wiki_links', []))
        retrieved_url_set = retrieved_wiki_urls
        
        metrics = MetricsCalculator.calculate_metrics(gt_wiki_set, retrieved_url_set)
        
        # Create entry for this prompt
        result_entry = {
            'prompt_index': prompt_data['index'],
            'prompt': prompt,
            'ground_truth_wiki_links': gt_wiki_links_str,
            'ground_truth_answers': gt_answers_str,
            'retrieved_wiki_urls': retrieved_wiki_urls_str,
            'num_unique_articles': len(retrieved_wiki_urls),
            # Retrieval metrics
            'retrieve_recall': metrics['recall'],
            'retrieve_precision': metrics['precision'],
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
            'lookup_time': retrieval_result.lookup_time,
            'rerank_time': retrieval_result.rerank_time,
            'rewriter_time': retrieval_result.rewriter_time
        }
        
        # Add generated queries if available
        if retrieval_result.generated_queries:
            result_entry['rewriter_queries'] = '|'.join(retrieval_result.generated_queries)
        
        self.results.append(result_entry)
        
        if verbose and article_titles:
            print(f"  Top result: {article_titles[0]}")
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert collected results to a pandas DataFrame.
        
        Returns:
            DataFrame with all results
        """
        return pd.DataFrame(self.results)
    
    def get_rewriter_io(self) -> List[Dict]:
        """
        Extract rewriter I/O data from results.
        
        Returns:
            List of rewriter I/O records
        """
        # This would need to be populated during result collection
        # For now, return empty list
        return []


class ResultsExporter:
    """Exports results to various formats."""
    
    @staticmethod
    def save(
        df: pd.DataFrame,
        output_path: str,
        formats: List[str] = ['pickle'],
        verbose: bool = True
    ) -> None:
        """
        Save results DataFrame to file(s).
        
        Args:
            df: DataFrame to save
            output_path: Base output path (e.g., 'results.pkl')
            formats: List of formats to save ('pickle', 'csv', 'json')
            verbose: Whether to print confirmation messages
        """
        output_path = Path(output_path)
        
        # Save pickle (main format)
        if 'pickle' in formats:
            df.to_pickle(output_path)
            if verbose:
                print(f"✅ Saved {len(df)} rows to {output_path} (pickle format)")
        
        # Save CSV
        if 'csv' in formats:
            csv_path = output_path.with_suffix('.csv')
            df.to_csv(csv_path, index=False)
            if verbose:
                print(f"✅ Also saved to {csv_path} (CSV format)")
        
        # Save JSON
        if 'json' in formats:
            json_path = output_path.with_suffix('.json')
            results_list = df.to_dict('records')
            with open(json_path, 'w') as f:
                json.dump(results_list, f, indent=2)
            if verbose:
                print(f"✅ Also saved to {json_path} (JSON format)")
    
    @staticmethod
    def save_rewriter_io(
        io_data: List[Dict],
        output_path: str,
        verbose: bool = True
    ) -> None:
        """
        Save rewriter I/O data to a pickle file.
        
        Args:
            io_data: List of rewriter I/O records
            output_path: Output file path
            verbose: Whether to print confirmation messages
        """
        if not io_data:
            return
        
        output_path = Path(output_path)
        io_df = pd.DataFrame(io_data)
        io_df.to_pickle(output_path)
        
        if verbose:
            print(f"\n✅ Saved {len(io_df)} rewriter I/O records to {output_path}")
            
            # Print diagnostics (with new flat structure: one row per step)
            num_unique_prompts = io_df['prompt_index'].nunique() if 'prompt_index' in io_df.columns else 0
            print(f"   Total rewriter steps: {len(io_df)}")
            print(f"   Unique prompts: {num_unique_prompts}")
            if num_unique_prompts > 0:
                print(f"   Average steps per prompt: {len(io_df) / num_unique_prompts:.1f}")
            
            # Check for reasoning tokens in outputs
            if 'rewriter_output' in io_df.columns:
                reasoning_count = io_df['rewriter_output'].apply(
                    lambda x: any(token in str(x).lower() for token in ['<channel>', '>analysis<', '<message>', 'we need to'])
                ).sum()
                
                if reasoning_count > 0:
                    print(f"   ⚠️  {reasoning_count}/{len(io_df)} steps have reasoning tokens in output")
    
    @staticmethod
    def save_retriever_io(
        io_data: List[Dict],
        output_path: str,
        verbose: bool = True
    ) -> None:
        """
        Save retriever I/O data to a pickle file.
        
        Args:
            io_data: List of retriever I/O records
            output_path: Output file path
            verbose: Whether to print confirmation messages
        """
        if not io_data:
            return
        
        output_path = Path(output_path)
        io_df = pd.DataFrame(io_data)
        io_df.to_pickle(output_path)
        
        if verbose:
            print(f"\n✅ Saved {len(io_df)} retriever I/O records to {output_path}")
            
            # Print diagnostics
            num_unique_prompts = io_df['prompt_index'].nunique() if 'prompt_index' in io_df.columns else 0
            print(f"   Total retrieval calls: {len(io_df)}")
            print(f"   Unique prompts: {num_unique_prompts}")
            if num_unique_prompts > 0:
                print(f"   Average retrievals per prompt: {len(io_df) / num_unique_prompts:.1f}")
            
            # Calculate average number of results
            if 'num_results' in io_df.columns:
                avg_results = io_df['num_results'].mean()
                print(f"   Average results per retrieval: {avg_results:.1f}")
    
    @staticmethod
    def save_reranker_io(
        io_data: List[Dict],
        output_path: str,
        verbose: bool = True
    ) -> None:
        """
        Save reranker I/O data to a pickle file.
        
        Args:
            io_data: List of reranker I/O records
            output_path: Output file path
            verbose: Whether to print confirmation messages
        """
        if not io_data:
            return
        
        output_path = Path(output_path)
        io_df = pd.DataFrame(io_data)
        io_df.to_pickle(output_path)
        
        if verbose:
            print(f"\n✅ Saved {len(io_df)} reranker I/O records to {output_path}")
            
            # Print diagnostics
            num_unique_prompts = io_df['prompt_index'].nunique() if 'prompt_index' in io_df.columns else 0
            print(f"   Total reranking calls: {len(io_df)}")
            print(f"   Unique prompts: {num_unique_prompts}")
            if num_unique_prompts > 0:
                print(f"   Average rerankings per prompt: {len(io_df) / num_unique_prompts:.1f}")
            
            # Calculate top-p filtering statistics
            if 'top_p' in io_df.columns and 'num_filtered_by_top_p' in io_df.columns:
                has_top_p = io_df['top_p'].notna().sum()
                if has_top_p > 0:
                    avg_filtered = io_df[io_df['top_p'].notna()]['num_filtered_by_top_p'].mean()
                    print(f"   Top-p filtering: {has_top_p} calls, average {avg_filtered:.1f} passages filtered")

