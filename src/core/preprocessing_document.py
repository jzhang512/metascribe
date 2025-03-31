"""
preprocessing_document.py

Functions for preprocessing documents (resizing to meet LLM API requirements and binarization for OCR)
"""

import os
from PIL import Image
import subprocess
import tempfile
Image.MAX_IMAGE_PIXELS = None


def binarize_image(input_image, jar_path, mode=1, size=30, percent=60, lossless="true", debug="false"):
    """
    Binarizes given image with ZigZag algorithm (Bloechler et al. 2024).
    https://doi.org/10.1145/3685650.3685661

    Args:
        input_image (PIL Image): The input image.
        jar_path (str): The path to the ZigZag jar file.
        mode (0-4): (default: 1)
            0: Standard binarization mode.
            1: Upsampled binarization mode (x2).
            2: Antialiased binarization mode.
            3: Background removal with gray-level foreground.
            4: Background removal with color foreground.
        size (int): pixel window size (default: 30)
        percent (int): weight factor (default: 60)
        lossless (str bool): lossless compression (default: "true")
        debug (str bool): debug mode (default: "false")

    Returns:
        PIL Image: The binarized image.
    """
    
    with tempfile.TemporaryDirectory() as temp_dir:

        # Temporarily save to disk for external processing by ZigZag
        input_dir = os.path.join(temp_dir, "input")
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        temp_input_path = os.path.join(input_dir, "temp_input.jpg")
        input_image.save(temp_input_path)

        # Run ZigZag via Java runtime.
        command = [
            "java", "-cp", jar_path, "zig.zag.ZigZag",
            "-mode", str(mode),
            "-size", str(size),
            "-percent", str(percent),
            "-lossless", lossless,
            "-debug", debug,
            "-input", input_dir,
            "-output", output_dir,
            "-exit", "true"
        ]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Load output
        output_file = os.path.join(output_dir, "temp_input.png")
        if os.path.exists(output_file):      
            binarized_image = Image.open(output_file).convert("RGB")
            return binarized_image
        else:
            raise Exception("Binarization subprocess failed. Please try again.")


def binarize_directory(input_dir, output_dir, jar_path, mode=1, size=30, percent=60, lossless="true", debug="false"):
    """
    Binarizes all images in a directory with ZigZag algorithm (Bloechler et al. 2024).
    https://doi.org/10.1145/3685650.3685661

    See binarize_image() for more details.

    Args:
        input_dir (str): The path to the directory containing the images to binarize.
        output_dir (str): The path to the directory to save the binarized images.
        jar_path (str): The path to the ZigZag jar file.

    Result:
        Saves binarized images to output_dir.
    """
    
    command = [
        "java", "-cp", jar_path, "zig.zag.ZigZag",
        "-mode", str(mode),
        "-size", str(size),
        "-percent", str(percent),
        "-lossless", lossless,
        "-debug", debug,
        "-input", input_dir,
        "-output", output_dir,
        "-exit", "true"
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    for filename in os.listdir(output_dir):
        if filename.lower().endswith(".png"):
            binarized_image = Image.open(os.path.join(output_dir, filename)).convert("RGB")
            binarized_image.save(os.path.join(output_dir, os.path.splitext(filename)[0] + ".jpg"))
        
        os.remove(os.path.join(output_dir, filename))   # delete original files


def resize_image(input_image, max_width=2000, max_height=2000):
    """
    Resize an image to a size suitable for LLM APIs.

    Args:
        input_image (PIL Image): The input image.
        max_input_image_pixels (int): The maximum number of pixels allowed in the input image (avoid DOS attacks). ** Default is None.
        max_width (int): The maximum width of output image. Default is 2000 px.
        max_height (int): The maximum height of output image. Default is 2000 px.

    Returns:
        PIL Image: The resized image; the given image if failed to resize.
    """
   
    width, height = input_image.size
    ratio = min(max_width/width, max_height/height)

    try:
        if ratio < 1:
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            resized_image = input_image.resize((new_width, new_height), Image.LANCZOS)
        else:
            resized_image = input_image     # no need

        return resized_image
    except Exception as e:
        return input_image


if __name__ == "__main__":
    image = Image.open("./20_nnc1.cu01975331.jpg")
    #binarized_image = binarize_image(image, jar_path="../resources/ZigZag.jar")
    resized_image = resize_image(image)
    resized_image.save("./20_nnc1.cu01975331_resized.jpg")
   #binarized_image.show()
