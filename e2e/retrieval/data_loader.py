"""
Data loading utilities for prompt datasets.
"""

import pandas as pd
from datasets import load_dataset
from typing import List, Dict, Any, Optional


class PromptLoader:
    """Handles loading prompts from various sources (HuggingFace, TSV files)."""
    
    @staticmethod
    def load_from_huggingface(split: str = "test") -> List[Dict[str, Any]]:
        """
        Load prompts from the frames-benchmark dataset on HuggingFace.
        
        Args:
            split: Dataset split to load ('test', 'train', etc.)
        
        Returns:
            List of dictionaries containing prompt information
        """
        print(f"Loading frames-benchmark dataset from HuggingFace (split: {split})...")
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
                    except:
                        wiki_links = [wiki_links]
                else:
                    wiki_links = [wiki_links]
            elif not isinstance(wiki_links, list):
                # If it's some other type, convert to list
                wiki_links = list(wiki_links) if hasattr(wiki_links, '__iter__') else [str(wiki_links)]
            
            # Handle answers - ensure it's a list
            answers = example.get('Answer', [])
            if answers is None:
                answers = []
            elif isinstance(answers, str):
                answers = [answers]
            elif not isinstance(answers, list):
                answers = list(answers) if hasattr(answers, '__iter__') else [str(answers)]
            
            prompts.append({
                'index': idx,
                'prompt': example['Prompt'],
                'answers': answers,
                'wiki_links': wiki_links
            })
        
        print(f"Loaded {len(prompts)} prompts from frames-benchmark")
        return prompts
    
    @staticmethod
    def load_from_tsv(tsv_path: str) -> List[Dict[str, Any]]:
        """
        Load prompts from a TSV file.
        
        Args:
            tsv_path: Path to TSV file
        
        Returns:
            List of dictionaries containing prompt information
        """
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
                    except:
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
        return prompts
    
    @staticmethod
    def load(split: str = "test", tsv_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Load prompts from either TSV file or HuggingFace dataset.
        
        Args:
            split: Dataset split to load if using HuggingFace (default: 'test')
            tsv_path: Path to TSV file. If provided, loads from file instead of HuggingFace
        
        Returns:
            List of dictionaries containing prompt information
        """
        if tsv_path:
            return PromptLoader.load_from_tsv(tsv_path)
        else:
            return PromptLoader.load_from_huggingface(split)

