"""
generate_metadata.py

Generate metadata from a given image and JSON schema with LLMs.
"""

from llm_interface import get_model_response
import json
import os
DEFAULT_SYSTEM_PROMPT = """
You are a metadata generator. Your task is to generate metadata from a given image and JSON schema.
"""
DEFAULT_USER_PROMPT = """
Based on the image and JSON schema, generate metadata for the image.
"""


def generate_single_metadata(model_name: str, image_path: str, json_schema: dict, system_prompt: str = DEFAULT_SYSTEM_PROMPT, user_prompt: str = DEFAULT_USER_PROMPT, **kwargs):
    """
    Generate metadata from a given image and JSON schema with LLMs.

    Args:
        model_name (str): The name of the LLM to use.
        image_path (str): The path to the image to generate metadata for.
        json_schema (dict): The JSON schema to use for generating metadata.
        system_prompt (str): The system prompt to use for generating metadata.
        user_prompt (str): The user prompt to use for generating metadata.
        **kwargs: Additional arguments to pass to the LLM.

    Returns:
        Response (in JSON format) from the LLM.
    """
    
    response, elapsed_time, cost = get_model_response(model_name, system_prompt, user_prompt, image_path, json_schema, **kwargs)

    return response, elapsed_time, cost


def generate_metadata_batch(model_name: str, image_folder_path: str, json_schema: dict, output_file_path: str, **kwargs):
    """
     Generate metadata for all images in a folder. 
     
     ** Assumes images are named by their unique IDs (uses filename as ID).

     Args:
        model_name (str): The name of the LLM to use.
        image_folder_path (str): The path to the folder containing the images to generate metadata for.
        json_schema (dict): The JSON schema to use for generating metadata.
        output_file_path (str): The path to the file to save the metadata to.
        **kwargs: Additional arguments to pass to the LLM.

    Outputs:
        JSONL file with metadata for each image (one image per line).
    """
     
    # Get all image paths in the folder.
    image_paths = [os.path.join(image_folder_path, f) for f in os.listdir(image_folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # Generate metadata for each image.
    metadata = []
    for image_path in image_paths:
    

if __name__ == "__main__":
    
    # Test.
    image_path = "20_nnc1.cu01975331.jpg"
    json_schema = "../../ppa_examples/metadata_criteria.json"

    criteria = json.load(open(json_schema))

    for _ in range(1):
        response, _, _ = generate_single_metadata("gemini-2.0-flash-001", image_path, criteria)
        print(response)
        print(str(response.additional_kwargs["parsing_error"]) + "**********************")