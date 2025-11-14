"""
Prompt templates for query generation in multi-step retrieval.

Provides model-specific formatters that return messages in the format:
[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
"""

from typing import List, Dict
from abc import ABC, abstractmethod


class QueryPromptFormatter(ABC):
    """
    Base class for formatting query generation prompts.
    
    Each model-specific subclass defines its own prompt templates and
    returns a list of message dicts with 'role' and 'content' keys.
    """
    
    @abstractmethod
    def format_messages(
        self,
        k: int,
        user_question: str,
        summarized_partial_context: str = "None yet - this is the first retrieval step."
    ) -> List[Dict[str, str]]:
        """
        Format prompt into a list of message dictionaries.

        Args:
            k: Number of queries to generate
            user_question: The user's original question
            summarized_partial_context: Summary of documents/facts already retrieved
        
        Returns:
            List of message dicts with 'role' and 'content' keys
            Example: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        """
        pass


class LlamaQueryPromptFormatter(QueryPromptFormatter):
    """
    Llama-specific prompt formatting for query generation.
    
    Uses a clear system/user role separation with structured prompts.
    """

    SYSTEM_PROMPT = \
"""You are an expert at generating search queries to help answer complex questions using a collection of Wikipedia articles.

Given the following:
- The user's original question.
- Relevant facts or documents already gathered so far (if any).

Your task:
Generate {k} concise, focused search queries that could be used to find specific information from Wikipedia to help answer the question.
- Make each query target a different aspect of the problem or missing information.
- Avoid duplicating information already in the context.
- Do not reference source filenames, document titles, or include any special characters.
- Think step by step before writing each query.
- List the missing pieces of information, then write {k} queries that could best retrieve them."""
    
    USER_PROMPT = """[User Question:]
{user_question}

[Known Facts / Retrieved Documents:]
{summarized_partial_context}

Please respond with exactly {k} search queries, one per line, numbered 1-{k}."""
    
    def format_messages(
        self,
        k: int,
        user_question: str,
        summarized_partial_context: str = "None yet - this is the first retrieval step."
    ) -> List[Dict[str, str]]:
        """
        Format messages for Llama models.
        
        Returns:
            List with system and user messages
        """
        system_content = self.SYSTEM_PROMPT.format(k=k)
        user_content = self.USER_PROMPT.format(
            k=k,
            user_question=user_question,
            summarized_partial_context=summarized_partial_context
        )
        
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]


# Default formatter (Llama)
DEFAULT_FORMATTER = LlamaQueryPromptFormatter()


def format_query_generation_prompt(
    user_question: str,
    k: int,
    summarized_partial_context: str = "None yet - this is the first retrieval step.",
    formatter: QueryPromptFormatter = None
) -> List[Dict[str, str]]:
    """
    Format the query generation prompt with the given parameters.
    
    This is a convenience function that uses the specified formatter
    (or the default Llama formatter if none provided).

    Args:
        user_question: The user's original question
        k: Number of queries to generate
        summarized_partial_context: Summary of documents/facts already retrieved (optional)
        formatter: QueryPromptFormatter instance (defaults to LlamaQueryPromptFormatter)

    Returns:
        List of message dicts: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
    """
    if formatter is None:
        formatter = DEFAULT_FORMATTER
    
    return formatter.format_messages(k, user_question, summarized_partial_context)
