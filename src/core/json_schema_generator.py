"""
json_schema_generator.py

Generate JSON schemas from a given (loose) criterion by user.
"""

import json
from typing import Dict, Any, Optional
from llm_interface import get_model_response

def generate_schema(criteria: str, model_name: str = "gpt-4o-2024-11-20", **kwargs) -> Dict[str, Any]:
    """
    Generate a JSON schema from loose user inputted criteria with LLMs.
    
    Args:
        criteria (str): User's loose criteria for metadata extraction
        model_name (str): Name of the LLM to use
        **kwargs: Additional arguments to pass to the model
        
    Returns:
        Dict[str, Any]: Generated JSON schema reflecting the user's criteria.
    """
    system_prompt = """
    You are a JSON schema expert. Your task is to convert the given loose criteria into a structured JSON schema.
    The schema should organize metadata extraction into sections, with each section containing specific questions.
    
    Follow these guidelines:
    1. Create a hierarchical structure with sections and questions
    2. Each section should focus on a coherent theme from the criteria
    3. Each question should have a clear description
    4. Use appropriate data types (string, number, boolean, array)
    5. Include any relevant constraints or formats
    """
    
    user_prompt = f"""
    Please convert the following criteria into a well-structured JSON schema for metadata extraction:
    
    {criteria}
    
    The schema should follow this general structure:
    {{
      "sections": [
        {{
          "title": "Section Title",
          "description": "Section description",
          "questions": [
            {{
              "id": "unique_id",
              "question": "Question text",
              "description": "Detailed description",
              "type": "string|number|boolean|array",
              "required": true|false
            }}
          ]
        }}
      ]
    }}
    
    Ensure each section is focused and not too broad, as each section will be a separate query to the LLM.
    """
    
    schema_output_structure = {
        "type": "object",
        "properties": {
            "sections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                        "questions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "question": {"type": "string"},
                                    "description": {"type": "string"},
                                    "type": {"type": "string", "enum": ["string", "number", "boolean", "array"]},
                                    "required": {"type": "boolean"}
                                },
                                "required": ["id", "question", "type"]
                            }
                        }
                    },
                    "required": ["title", "questions"]
                }
            }
        },
        "required": ["sections"]
    }
    
    response, elapsed_time, cost = get_model_response(
        model_name=model_name,
        sys_prompt=system_prompt,
        user_prompt=user_prompt,
        structured_output_schema=schema_output_structure,
        **kwargs
    )
    
    # Log performance metrics
    print(f"Schema generation completed in {elapsed_time:.2f} seconds. Cost: ${cost:.6f}")
    
    return response

