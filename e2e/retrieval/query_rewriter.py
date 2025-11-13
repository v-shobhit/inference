"""
Query rewriter for generating multiple search queries from a single question.
"""

import re
from typing import List, Tuple, Optional
from openai import OpenAI
from query_generation_prompts import format_query_generation_prompt


class QueryRewriter:
    """Generates multiple search queries from a user question using an LLM."""
    
    def __init__(
        self,
        client: OpenAI,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 500
    ):
        """
        Initialize the QueryRewriter.
        
        Args:
            client: OpenAI client instance (works with SGLang, vLLM, etc.)
            model: Model name to use
            temperature: Sampling temperature for generation
            max_tokens: Maximum tokens to generate
        """
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def generate_queries(
        self,
        user_question: str,
        k: int,
        summarized_context: str = "None yet - this is the first retrieval step.",
        return_io: bool = False
    ) -> Tuple[List[str], ...]:
        """
        Generate k search queries from a user question.
        
        Args:
            user_question: The user's original question
            k: Number of queries to generate
            summarized_context: Summary of previously retrieved documents
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
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        # Extract the generated text
        generated_text = response.choices[0].message.content
        
        # Parse the queries from the response
        queries = self._parse_queries(generated_text, k)
        
        if return_io:
            return (queries, prompt, generated_text)
        else:
            return queries
    
    def _parse_queries(self, generated_text: str, k: int) -> List[str]:
        """
        Parse queries from LLM-generated text.
        
        Args:
            generated_text: Raw output from the LLM
            k: Expected number of queries
        
        Returns:
            List of parsed query strings
        """
        queries = []
        lines = generated_text.strip().split('\n')
        
        # Try to match numbered lines (1. query, 2. query, etc.)
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
        
        # Return up to k queries, or fallback to empty list if parsing completely failed
        return queries[:k] if queries else []

