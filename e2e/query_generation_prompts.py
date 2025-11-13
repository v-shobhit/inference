"""
Prompt templates for query generation in multi-step retrieval.
"""

# Template for generating search queries from a user question
QUERY_GENERATION_PROMPT = """You are an expert at generating search queries to help answer complex questions using a collection of Wikipedia articles.

Given the following:
- The user's original question.
- Relevant facts or documents already gathered so far (if any).

Your task:  
Generate {k} concise, focused search queries that could be used to find specific information from Wikipedia to help answer the question.  
- Make each query target a different aspect of the problem or missing information.  
- Avoid duplicating information already in the context.  
- Do not reference source filenames, document titles, or include any special characters.
- Think step by step before writing each query.
- List the missing pieces of information, then write {k} queries that could best retrieve them.

[User Question:]
{user_question}

[Known Facts / Retrieved Documents:]
{summarized_partial_context}

Please respond with exactly {k} search queries, one per line, numbered 1-{k}."""


def format_query_generation_prompt(
    user_question: str,
    k: int,
    summarized_partial_context: str = "None yet - this is the first retrieval step."
) -> str:
    """
    Format the query generation prompt with the given parameters.
    
    Args:
        user_question: The user's original question
        k: Number of queries to generate
        summarized_partial_context: Summary of documents/facts already retrieved (optional)
    
    Returns:
        Formatted prompt string
    """
    return QUERY_GENERATION_PROMPT.format(
        k=k,
        user_question=user_question,
        summarized_partial_context=summarized_partial_context
    )

