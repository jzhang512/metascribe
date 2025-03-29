"""
generate_metadata.py

Generate metadata from a given image and JSON schema with LLMs.
"""

from core.llm_interface import get_model_response
import json
import os
import datetime

DEFAULT_SYSTEM_PROMPT = """
You are a metadata generator. Your task is to generate metadata from a given image and JSON schema.
"""
DEFAULT_USER_PROMPT = """
Based on the image and JSON schema, generate metadata for the image.
"""


def generate_single_metadata(model_name: str, image_path: str, json_schema: dict, system_prompt: str = DEFAULT_SYSTEM_PROMPT, user_prompt: str = DEFAULT_USER_PROMPT, image_id: str = None, **kwargs):
    """
    Generate metadata from a given image and JSON schema with LLM API call.

    ** Will use image filename as ID if ID not explicitly provided.

    Args:
        model_name (str): The name of the LLM to use.
        image_path (str): The path to the image to generate metadata for.
        json_schema (dict): The JSON schema to use for generating metadata.
        system_prompt (str): The system prompt to use for generating metadata.
        user_prompt (str): The user prompt to use for generating metadata.
        image_id (str): The ID of the image to generate metadata for.
        **kwargs: Additional arguments to pass to the LLM.

    Returns:
        Response (dict in JSON format) from the LLM.
    """

    elapsed_time = None
    cost = None
    image_id = os.path.basename(image_path).rsplit('.', 1)[0] if image_id is None else image_id
    
    try:
        response, elapsed_time, cost = get_model_response(model_name, system_prompt, user_prompt, image_path, json_schema, **kwargs)

        parsing_error = response.additional_kwargs["parsing_error"]
        parsed_data = response.additional_kwargs["parsed_data"]
        if parsing_error is not None:
            raise Exception(f"Parsing error: {parsed_data}")    # parsed_data explains why

        usage = {
            "prompt_tokens": response.usage_metadata["input_tokens"],
            "completion_tokens": response.usage_metadata["output_tokens"],
            "reasoning_tokens": response.usage_metadata.get("output_token_details", {}).get("reasoning", 0)
        }

        metadata_entry = {
            "model_name": model_name,
            "image_id": image_id,
            "metadata": parsed_data,
            "usage": usage,
            "elapsed_time": elapsed_time,
            "cost": cost
        }

        if "logprobs" in kwargs and kwargs["logprobs"] is True:
            if hasattr(response, "response_metadata") and response.response_metadata and response.response_metadata.get("logprobs"):
                metadata_entry["logprobs"] = response.response_metadata.get("logprobs")


        print(f"{_get_printable_time()} - generated metadata for {image_id} with {model_name} in {elapsed_time} s")

    except Exception as e:
        print(f"{_get_printable_time()} - ERROR: generating metadata for image {image_id} with {model_name}: {str(e)}")

        metadata_entry = {
            "model_name": model_name,
            "image_id": image_id,
            "error": str(e)
        }

        if elapsed_time is not None:
            metadata_entry["elapsed_time"] = elapsed_time
        if cost is not None:
            metadata_entry["cost"] = cost

    return metadata_entry


def generate_metadata_batch(model_name: str, image_folder_path: str, json_schema: dict, output_file_path: str, **kwargs):
    """
     Generate metadata for all images in a folder with a specified LLM. 
     
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

    # Generate and output metadata for each image.
    with open(output_file_path, 'a') as f:
        for image_path in image_paths:
            image_id = os.path.basename(image_path).rsplit('.', 1)[0]   # page's filename is its ID
            
            metadata_entry = generate_single_metadata(model_name, image_path, json_schema, image_id=image_id, **kwargs)

            f.write(json.dumps(metadata_entry) + '\n')
            f.flush()

# Helper. 
def _get_printable_time():
    """Returns the current date and time as a formatted string."""
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %I:%M:%S %p")
    return formatted_datetime

if __name__ == "__main__":

    # Batch generate test.
    # image_folder_path = "../../ppa_examples/test_images"
    # json_schema = "../../ppa_examples/metadata_criteria.json"
    # criteria = json.load(open(json_schema))
    # output_file_path = "test_metadata_batch_output.jsonl"
    # generate_metadata_batch("gpt-4o-mini-2024-07-18", image_folder_path, criteria, output_file_path, logprobs=True)
    
    # Single generate test.
    image_path = "20_nnc1.cu01975331.jpg"
    json_schema = "../../ppa_examples/metadata_criteria.json"

    criteria = json.load(open(json_schema))

    for _ in range(1):
        response = generate_single_metadata("gpt-4o-mini-2024-07-18", image_path, criteria, logprobs=True, top_logprobs=1)
        #response, _, _ = generate_single_metadata("gemini-2.0-flash-001", image_path, criteria, responseLogprobs=True)
        #print(response)
        print(response)