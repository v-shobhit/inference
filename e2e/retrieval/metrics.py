"""
Metrics calculation for retrieval evaluation.
"""

from typing import Set, Dict, List
import pandas as pd


class MetricsCalculator:
    """Calculates retrieval metrics like recall, precision, and F1."""
    
    @staticmethod
    def calculate_recall(ground_truth: Set, retrieved: Set) -> float:
        """
        Calculate recall: fraction of ground truth items that were retrieved.
        
        Args:
            ground_truth: Set of ground truth items (e.g., article URLs)
            retrieved: Set of retrieved items
        
        Returns:
            Recall score between 0 and 1
        """
        if len(ground_truth) == 0:
            return 0.0
        
        num_correct = len(ground_truth.intersection(retrieved))
        return num_correct / len(ground_truth)
    
    @staticmethod
    def calculate_precision(ground_truth: Set, retrieved: Set) -> float:
        """
        Calculate precision: fraction of retrieved items that are correct.
        
        Args:
            ground_truth: Set of ground truth items (e.g., article URLs)
            retrieved: Set of retrieved items
        
        Returns:
            Precision score between 0 and 1
        """
        if len(retrieved) == 0:
            return 0.0
        
        num_correct = len(ground_truth.intersection(retrieved))
        return num_correct / len(retrieved)
    
    @staticmethod
    def calculate_f1(precision: float, recall: float) -> float:
        """
        Calculate F1 score: harmonic mean of precision and recall.
        
        Args:
            precision: Precision score
            recall: Recall score
        
        Returns:
            F1 score between 0 and 1
        """
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def calculate_metrics(ground_truth: Set, retrieved: Set) -> Dict[str, float]:
        """
        Calculate all metrics (recall, precision, F1) at once.
        
        Args:
            ground_truth: Set of ground truth items
            retrieved: Set of retrieved items
        
        Returns:
            Dictionary with 'recall', 'precision', and 'f1' keys
        """
        recall = MetricsCalculator.calculate_recall(ground_truth, retrieved)
        precision = MetricsCalculator.calculate_precision(ground_truth, retrieved)
        f1 = MetricsCalculator.calculate_f1(precision, recall)
        
        return {
            'recall': recall,
            'precision': precision,
            'f1': f1
        }
    
    @staticmethod
    def aggregate_metrics(df: pd.DataFrame) -> Dict[str, float]:
        """
        Aggregate metrics across a DataFrame of results.
        
        Args:
            df: DataFrame with 'retrieve_recall' and 'retrieve_precision' columns
        
        Returns:
            Dictionary with aggregated statistics
        """
        avg_recall = df['retrieve_recall'].mean()
        avg_precision = df['retrieve_precision'].mean()
        f1_score = MetricsCalculator.calculate_f1(avg_precision, avg_recall)
        
        perfect_recall = (df['retrieve_recall'] == 1.0).sum()
        perfect_precision = (df['retrieve_precision'] == 1.0).sum()
        zero_recall = (df['retrieve_recall'] == 0.0).sum()
        
        return {
            'avg_recall': avg_recall,
            'avg_precision': avg_precision,
            'f1_score': f1_score,
            'perfect_recall_count': perfect_recall,
            'perfect_recall_pct': perfect_recall / len(df) * 100,
            'perfect_precision_count': perfect_precision,
            'perfect_precision_pct': perfect_precision / len(df) * 100,
            'zero_recall_count': zero_recall,
            'zero_recall_pct': zero_recall / len(df) * 100
        }
    
    @staticmethod
    def distribution_analysis(scores: List[float], bins: List[Tuple[float, float]] = None) -> Dict:
        """
        Analyze the distribution of scores across bins.
        
        Args:
            scores: List of scores (e.g., recall or precision values)
            bins: List of (min, max) tuples defining bins. Default: [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
        
        Returns:
            Dictionary mapping bin ranges to counts
        """
        if bins is None:
            bins = [(0.0, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.00)]
        
        distribution = {}
        for bin_min, bin_max in bins:
            if bin_max == 1.0:
                # Include upper bound for last bin
                count = sum(1 for s in scores if bin_min <= s <= bin_max)
            else:
                count = sum(1 for s in scores if bin_min <= s < bin_max)
            
            bin_label = f"{bin_min:.2f} - {bin_max:.2f}"
            distribution[bin_label] = count
        
        return distribution

