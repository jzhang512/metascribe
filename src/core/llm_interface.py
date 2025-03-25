"""
llm_interface.py

Handles interactions with LLM providers.

Currently supports:
- OpenAI
- Anthropic
- Google Vertex
- Together AI
"""

import os
import time
import base64
import json
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_together import ChatTogether
from langchain_core.messages import HumanMessage, SystemMessage

from jsonschema import validate, ValidationError
from dotenv import load_dotenv

# API keys.
# Ensure the names are:
# OPENAI_API_KEY
# ANTHROPIC_API_KEY
# GOOGLE_API_KEY
# TOGETHER_API_KEY
load_dotenv()

# Pre-defined models (and pricing as of 3/1/2025)
# model_name: (provider, input price, output price)
# Pricing is $ per 1 million tokens
SUPPORTED_MODELS = {
    "o1-2024-12-17": ("openai", 15/1000000, 60/1000000),
    "gpt-4.5-preview-2025-02-27": ("openai", 75/1000000, 150/1000000),
    "gpt-4o-2024-11-20": ("openai", 2.5/1000000, 10/1000000),
    "gpt-4o-mini-2024-07-18": ("openai", 0.15/1000000, 0.60/1000000),
    "gpt-4-turbo-2024-04-09": ("openai", 10/1000000, 30/1000000),
    "claude-3-7-sonnet-20250219": ("anthropic", 3/1000000, 15/1000000),
    "claude-3-5-sonnet-20241022": ("anthropic", 3/1000000, 15/1000000),
    "claude-3-5-haiku-20241022": ("anthropic", 0.8/1000000, 4/1000000),
    "claude-3-opus-20240229": ("anthropic", 15/1000000, 75/1000000),
    "claude-3-haiku-20240307": ("anthropic", 0.25/1000000, 1.25/1000000),
    "gemini-2.0-flash-001": ("google", 0.1/1000000, 0.4/1000000),
    "gemini-2.0-flash-lite-001": ("google", 0.075/1000000, 0.3/1000000),
    "gemini-1.5-flash-002": ("google", 0.075/1000000, 0.3/1000000),
    "gemini-1.5-flash-8b-001": ("google", 0.0375/1000000, 0.15/1000000),
    "gemini-1.5-pro-002": ("google", 1.25/1000000, 5/1000000),
    "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo": ("together", 0.18/1000000, 0.18/1000000),
    "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo": ("together", 1.2/1000000, 1.2/1000000),
    "Qwen/Qwen2-VL-72B-Instruct": ("together", 1.2/1000000, 1.2/1000000),
}


def get_model_response(model_name: str, sys_prompt: str, user_prompt: str, image_path: str = None, structured_output_schema: dict = None, **kwargs):
    """
    Master function for routing which LLM provider to get response from based on the model name.

    Args:
        model_name (str): The name of the model to use.
        sys_prompt (str): The system prompt to use.
        user_prompt (str): The user prompt to use.
        image_path (str): The path to the image to use.
        is_structured_output (bool): Whether to use structured output.
        **kwargs: Additional arguments to pass to the model.

    Returns:
        dict: The response from the model.
    """

    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Model {model_name} is currently not supported. We currently support the following models: {SUPPORTED_MODELS.keys()}")
    
    provider, input_price, output_price = SUPPORTED_MODELS[model_name]

    # Special handling for Together AI models with structured output
    if provider == "together" and structured_output_schema is not None:
        # Together AI models may not support structured output directly
        # Instead, we'll modify the prompt to request structured output
        structured_instructions = f"""
        Make sure your output is one proper JSON object. Do not say anything else.

        {json.dumps(structured_output_schema, indent=2)}
        """
        
        # Append the structured instructions to the system prompt
        user_prompt = f"{user_prompt}\n\n{structured_instructions}"
        
        # Call without structured output schema
        llm = ChatTogether(model = model_name, **kwargs)
    else:
        # Route to right LLM provider as before
        if provider == "openai":
            llm = ChatOpenAI(model = model_name, **kwargs)
        elif provider == "anthropic":
            llm = ChatAnthropic(model = model_name, **kwargs)
        elif provider == "google":
            llm = ChatGoogleGenerativeAI(model = model_name, **kwargs)
        elif provider == "together":
            llm = ChatTogether(model = model_name, **kwargs)
        
        if structured_output_schema is not None:
            llm = llm.with_structured_output(structured_output_schema, include_raw = True)

    messages = [
        SystemMessage(content = sys_prompt),
        HumanMessage(
            content = [
                {"type": "text", "text": user_prompt}
            ]
        )
    ]

    if image_path is not None:
        image_base64, image_media_type = _encode_image(image_path)
        messages[1].content.append(
            {
                "type": "image_url", 
                "image_url": {"url":f"data:{image_media_type};base64,{image_base64}"}
            }
        )
    
    start_time = time.time()
    response = llm.invoke(messages)
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Structured output: LangChain returns a tuple of (raw, parsed, parsing_error).
    if structured_output_schema is not None:
        if provider == "together":  # response is already raw. Need manual validation.
            content = _validate_together_output(response.content, structured_output_schema)

            if content is None:
                parsed_data = "LLM response not structured as expected."
                parsing_error = True
            else:
                parsed_data = content
                parsing_error = None
        else: 
            parsed_data = response["parsed"]
            parsing_error = response["parsing_error"]
            response = response["raw"]

            if parsing_error is not None:
                parsed_data = "LLM response not structured as expected."


        if not hasattr(response, "additional_kwargs") or response.additional_kwargs is None:
            response.additional_kwargs = {}
        
        response.additional_kwargs["parsed_data"] = parsed_data
        response.additional_kwargs["parsing_error"] = parsing_error


    cost = response.usage_metadata["input_tokens"] * input_price + response.usage_metadata["output_tokens"] * output_price

    # Google AI doesn't include model used.
    if provider == "google":
        response.response_metadata["model_name"] = model_name
    
    return response, elapsed_time, cost

# Helper: extract and validate output from Together AI.
def _validate_together_output(response: str, structured_output_schema: dict):
    """
    Extract JSON from Together AI response and validate against schema.
    
    Args:
        response (str): Raw text response from Together AI
        structured_output_schema (dict): JSON schema to validate against
        
    Returns:
        str: JSON if valid, None if invalid
    """
    try:
        print (response, "\n\n\n")
        # Find the first { and last }
        start_idx = response.find('{')
        end_idx = response.rfind('}')
        
        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            return None
            
        # Extract JSON string
        json_str = response[start_idx:end_idx+1]
        
        # Parse JSON
        parsed_json = json.loads(json_str)
        
        # Validate against schema
        validate(instance=parsed_json, schema=structured_output_schema)
        
        return json_str
    except (json.JSONDecodeError, ValidationError) as e:
        return None
   

# Helper: encodes local images into base64. Also returns media type.
def _encode_image(image_path):
    
    if ".png" in os.path.basename(image_path):
        image_media_type = "image/png"
    elif ".jpg" in os.path.basename(image_path):
        image_media_type = "image/jpeg"
    else:
        raise Exception("Unsupported image format given (must be PNG or JPEG).")
    
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8"), image_media_type
    

if __name__ == "__main__":

    # Tests.
    image_path = "20_nnc1.cu01975331.jpg"

    # Purely text.
    # print(get_model_response("gpt-4o-mini-2024-07-18", "You are a helpful assistant.", "what's in this page?", None, None), "\n\n----------------")
    # print(get_model_response("claude-3-5-sonnet-20241022", "You are a helpful assistant.", "what's in this page?", None, None), "\n\n----------------")
    # print(get_model_response("gemini-2.0-flash-001", "You are a helpful assistant.", "what's in this page?", None, None), "\n\n----------------")
    # print(get_model_response("meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo", "You are a helpful assistant.", "what's in this page?", None, None), "\n\n----------------")
    print(get_model_response("Qwen/Qwen2-VL-72B-Instruct", "You are a helpful assistant.", "what's in this page?", None, None), "\n\n----------------")

    # Text and image.
    # print(get_model_response("gpt-4o-mini-2024-07-18", "You are a helpful assistant.", "what's in this page?", image_path, None), "\n\n----------------")
    # print(get_model_response("claude-3-5-sonnet-20241022", "You are a helpful assistant.", "what's in this page?", image_path, None), "\n\n----------------")
    # print(get_model_response("gemini-2.0-flash-001", "You are a helpful assistant.", "what's in this page?", image_path, None), "\n\n----------------")
    # print(get_model_response("meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo", "You are a helpful assistant.", "what's in this page?", image_path, None), "\n\n----------------")
    #print(get_model_response("Qwen/Qwen2-VL-72B-Instruct", "You are a helpful assistant.", "what's in this page?", image_path, None), "\n\n----------------")

    # Extra parameters.
    # print(get_model_response("gpt-4o-mini-2024-07-18", "You are a helpful assistant.", "what's in this page?", image_path, None, max_tokens = 2), "\n\n----------------")
    # print(get_model_response("claude-3-5-sonnet-20241022", "You are a helpful assistant.", "what's in this page?", image_path, None, max_tokens = 2), "\n\n----------------")
    # print(get_model_response("gemini-2.0-flash-001", "You are a helpful assistant.", "what's in this page?", image_path, None, max_tokens = 2), "\n\n----------------")
    # print(get_model_response("meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo", "You are a helpful assistant.", "what's in this page?", image_path, None, max_tokens = 2), "\n\n----------------")
    #print(get_model_response("Qwen/Qwen2-VL-72B-Instruct", "You are a helpful assistant.", "what's in this page?", image_path, None, max_tokens = 2), "\n\n----------------")