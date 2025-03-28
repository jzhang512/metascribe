"""
aggregation.py

For aggregating page-level metadata results to represent the document/work as a whole.
"""

from core.llm_interface import get_model_response

def generate_single_aggregated_metadata(model_name: str, concatenated_metadata: str, system_prompt: str, user_prompt: str, **kwargs):
