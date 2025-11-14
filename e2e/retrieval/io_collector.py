"""
I/O data collection for debugging and analysis.

Collects input/output data from retrieval, reranking, and rewriting operations
for debugging and analysis purposes.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class IOCollector:
    """Collects I/O data from retrieval, reranking, and rewriting."""
    
    # Enable/disable collection for each type
    collect_rewriter: bool = False
    collect_retriever: bool = False
    collect_reranker: bool = False
    
    # Data storage
    rewriter_data: List[Dict] = field(default_factory=list, init=False)
    retriever_data: List[Dict] = field(default_factory=list, init=False)
    reranker_data: List[Dict] = field(default_factory=list, init=False)
    
    def add_rewriter(self, prompt_index: int, prompt: str, step_data: Dict) -> None:
        """
        Add rewriter I/O data.
        
        Args:
            prompt_index: Index of the prompt
            prompt: The prompt text
            step_data: Dictionary with step info (step, rewriter_input, rewriter_output, generated_queries)
        """
        if not self.collect_rewriter:
            return
        
        self.rewriter_data.append({
            'prompt_index': prompt_index,
            'prompt': prompt,
            'step': step_data.get('step'),
            'rewriter_input': step_data.get('rewriter_input', ''),
            'rewriter_output': step_data.get('rewriter_output', ''),
            'generated_queries': step_data.get('generated_queries', [])
        })
    
    def add_retriever(self, prompt_index: int, prompt: str, retriever_data: Dict) -> None:
        """
        Add retriever I/O data.
        
        Args:
            prompt_index: Index of the prompt
            prompt: The prompt text
            retriever_data: Dictionary with retriever info (query, top_k, num_results, retrieved_passages)
        """
        if not self.collect_retriever:
            return
        
        self.retriever_data.append({
            'prompt_index': prompt_index,
            'prompt': prompt,
            'query': retriever_data.get('query', ''),
            'top_k': retriever_data.get('top_k'),
            'num_results': retriever_data.get('num_results'),
            'retrieved_passages': retriever_data.get('retrieved_passages', [])
        })
    
    def add_reranker(self, prompt_index: int, prompt: str, reranker_data: Dict) -> None:
        """
        Add reranker I/O data.
        
        Args:
            prompt_index: Index of the prompt
            prompt: The prompt text
            reranker_data: Dictionary with reranker info (query, input_passages, output_before/after_normalization, top_p, etc.)
        """
        if not self.collect_reranker:
            return
        
        self.reranker_data.append({
            'prompt_index': prompt_index,
            'prompt': prompt,
            'query': reranker_data.get('query', ''),
            'input_passages': reranker_data.get('input_passages', []),
            'output_before_normalization': reranker_data.get('output_before_normalization', []),
            'output_after_normalization': reranker_data.get('output_after_normalization', []),
            'top_p': reranker_data.get('top_p'),
            'num_kept_after_top_p': reranker_data.get('num_kept_after_top_p'),
            'num_filtered_by_top_p': reranker_data.get('num_filtered_by_top_p')
        })
    
    def get_data(self) -> Dict[str, Optional[List]]:
        """
        Get all collected data.
        
        Returns:
            Dictionary with keys 'rewriter', 'retriever', 'reranker'.
            Values are lists of data if collection was enabled, None otherwise.
        """
        return {
            'rewriter': self.rewriter_data if self.collect_rewriter else None,
            'retriever': self.retriever_data if self.collect_retriever else None,
            'reranker': self.reranker_data if self.collect_reranker else None
        }
    
    def is_empty(self) -> bool:
        """
        Check if any data was collected.
        
        Returns:
            True if no data was collected
        """
        return (not self.rewriter_data and 
                not self.retriever_data and 
                not self.reranker_data)
    
    def __len__(self) -> int:
        """Return total number of collected items across all types."""
        return len(self.rewriter_data) + len(self.retriever_data) + len(self.reranker_data)

