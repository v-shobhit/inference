"""
Statistics tracking for retrieval operations.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .engine import RetrievalResult


@dataclass
class RetrievalStats:
    """Tracks retrieval statistics with automatic calculation of derived metrics."""
    
    num_prompts: int = 0
    total_lookup_time: float = 0.0
    total_rerank_time: float = 0.0
    total_rewriter_time: float = 0.0
    total_chunks_kept: int = 0
    
    passages_before_rerank: List[int] = field(default_factory=list)
    passages_after_rerank: List[int] = field(default_factory=list)
    
    def add_result(self, result: 'RetrievalResult') -> None:
        """
        Add a retrieval result to statistics.
        
        Args:
            result: RetrievalResult object to add to stats
        """
        self.num_prompts += 1
        self.total_lookup_time += result.lookup_time
        self.total_rerank_time += result.rerank_time
        self.total_rewriter_time += result.rewriter_time
        self.total_chunks_kept += result.num_chunks_kept
        
        self.passages_before_rerank.append(len(result.raw_results))
        self.passages_after_rerank.append(len(result.reranked_results))
    
    @property
    def avg_lookup_time(self) -> float:
        """Average lookup time per prompt."""
        return self.total_lookup_time / self.num_prompts if self.num_prompts > 0 else 0.0
    
    @property
    def avg_rerank_time(self) -> float:
        """Average rerank time per prompt."""
        return self.total_rerank_time / self.num_prompts if self.num_prompts > 0 else 0.0
    
    @property
    def avg_rewriter_time(self) -> float:
        """Average rewriter time per prompt."""
        return self.total_rewriter_time / self.num_prompts if self.num_prompts > 0 else 0.0
    
    @property
    def avg_chunks_kept(self) -> float:
        """Average chunks kept per prompt."""
        return self.total_chunks_kept / self.num_prompts if self.num_prompts > 0 else 0.0
    
    def get_passage_stats(self, passage_list: List[int]) -> Dict[str, float]:
        """
        Calculate statistics for passage counts.
        
        Args:
            passage_list: List of passage counts
            
        Returns:
            Dictionary with mean, min, quartiles, and max
        """
        if not passage_list:
            return {
                'mean': 0.0,
                'min': 0,
                'q1': 0.0,
                'median': 0.0,
                'q3': 0.0,
                'max': 0
            }
        
        arr = np.array(passage_list)
        return {
            'mean': float(np.mean(arr)),
            'min': int(np.min(arr)),
            'q1': float(np.percentile(arr, 25)),
            'median': float(np.percentile(arr, 50)),
            'q3': float(np.percentile(arr, 75)),
            'max': int(np.max(arr))
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for backward compatibility.
        
        Returns:
            Dictionary with all statistics
        """
        return {
            'num_prompts': self.num_prompts,
            'total_lookup_time': self.total_lookup_time,
            'total_rerank_time': self.total_rerank_time,
            'total_rewriter_time': self.total_rewriter_time,
            'total_chunks_kept': self.total_chunks_kept,
            'avg_lookup_time': self.avg_lookup_time,
            'avg_rerank_time': self.avg_rerank_time,
            'avg_rewriter_time': self.avg_rewriter_time,
            'avg_chunks_kept': self.avg_chunks_kept,
            'passages_before_rerank': self.get_passage_stats(self.passages_before_rerank),
            'passages_after_rerank': self.get_passage_stats(self.passages_after_rerank)
        }

