"""
aggregation.py

For aggregating page-level metadata results to represent the document/work as a whole.
"""

from core.llm_interface import get_model_response

DEFAULT_SYSTEM_PROMPT = """
    You are a specialized document metadata processor creating summaries that will be converted to vector embeddings for semantic search. Your task is to produce summaries optimized for embedding performance.
    
    IMPORTANT: Only summarize information explicitly provided in the metadata fields. Do not speculate, infer, or add any information not directly present.

    Guidelines:
    - Create semantically dense sentences
    - Prioritize domain-specific terminology and key concepts
    - If relevant to the document, include specific entities and proper nouns
    - Minimize filler words and focus on content-rich language
    - Ensure conceptual completeness rather than stylistic elegance
    
    Your output must effectively capture the document's semantic essence for both human readers and vector-based retrieval systems.
"""

DEFAULT_USER_PROMPT = """
    Based on the following aggregated page-level document metadata, create a 3-4 sentence summary that captures the most essential elements. This is document-level metadata.
    
    Minimize filler words and only summarize information explicitly provided in the metadata -- do not speculate, infer, or add any information not present.

    Ensure the summary is exactly 3-4 sentences long and suitable for generating vector embeddings.

    --------------------
"""

def generate_single_aggregated_metadata(model_name: str, concatenated_metadata: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT, user_prompt: str = DEFAULT_USER_PROMPT, **kwargs):
    """
    Generate aggregated metadata from concatenated page-level metadata.

    Args:
        model_name (str): The name of the LLM to use.
        concatenated_metadata (str): The concatenated page-level metadata to summarize.
        system_prompt (str): The system prompt to use for generating the summary.
        user_prompt (str): The user prompt to use for generating the summary.
        **kwargs: Additional arguments to pass to the LLM.

    Returns:
        (str, elapsed_time, cost): The generated summary, and the elapsed time, and cost of the operation.
    """

    elapsed_time = None
    cost = None
    complete_user_prompt = f"{user_prompt}\n\n{concatenated_metadata}"

    try:

        response, elapsed_time, cost = get_model_response(
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt=complete_user_prompt,
            **kwargs
        )

        summary_entry = {
            "summary": response.content,
            "elapsed_time": elapsed_time,
            "cost": cost
        }

    except Exception as e:
        summary_entry = {
            "error": str(e)
        }

        if elapsed_time is not None:
            summary_entry["elapsed_time"] = elapsed_time
        if cost is not None:
            summary_entry["cost"] = cost

    return summary_entry

if __name__ == "__main__":
    print(generate_single_aggregated_metadata(
        model_name="gemini-2.0-flash-001",
        concatenated_metadata=".",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        user_prompt=DEFAULT_USER_PROMPT
    ))