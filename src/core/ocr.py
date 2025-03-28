"""
ocr.py

Run OCR on given images with LLM APIs.
"""

from core.llm_interface import get_model_response
import os

DEFAULT_SYSTEM_PROMPT = """
You are an OCR assistant. Extract and return only the text from the image. Do NOT add any extra explanations, formatting, or special markdown commands. Output plain text only.
"""

DEFAULT_USER_PROMPT = (
    "Extract and return the text from this image according to these principles:\n\n"
    "- Capture all Unicode-compatible characters, including symbols and special characters.\n"
    "- Preserve the author’s intended meaning and formatting.\n"
    "- Do not correct grammar, spelling, or typographic errors—transcribe them exactly as they appear.\n\n"
    "DO NOT add any extra explanations, formatting, or special markdown commands."
)

def generate_single_ocr(model_name: str, image_path: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT, user_prompt: str = DEFAULT_USER_PROMPT, image_id: str = None, **kwargs):
    """
    Generate OCR of a given (page) image with given LLM's API.

    ** Will use image filename as ID if ID not explicitly provided.

    Args:
        model_name (str): The name of the LLM to use.
        image_path (str): The path to the image to generate OCR for.
        system_prompt (str): The system prompt to use for generating OCR.
        user_prompt (str): The user prompt to use for generating OCR.
        **kwargs: Additional arguments to pass to the LLM.

    Returns:
        Response (dict in JSON format) from the LLM.
        (id, ocr_text, elapsed_time, cost); (id, error) if failed.
    """

    elapsed_time = None
    cost = None
    image_id = os.path.basename(image_path).rsplit('.', 1)[0] if image_id is None else image_id

    try:
        response, elapsed_time, cost = get_model_response(model_name, system_prompt, user_prompt, image_path, **kwargs)

        ocr_text = response.content

        ocr_entry = {
            "id": image_id,
            "ocr_text": ocr_text,
            "elapsed_time": elapsed_time,
            "cost": cost
        }

        print(f"OCRed {image_id}")
    except Exception as e:
        print(f"ERROR: OCRing {image_id}")

        ocr_entry = {
            "id": image_id,
            "error": str(e)
        }

        if elapsed_time is not None:
            ocr_entry["elapsed_time"] = elapsed_time
        if cost is not None:
            ocr_entry["cost"] = cost

    return ocr_entry

if __name__ == "__main__":
    print(generate_single_ocr("gpt-4o-mini-2024-07-18", "20_nnc1.cu01975331.jpg"))