"""
controller.py

Main controller for MetaScribe pipeline.
"""

import yaml
import os

from core.generate_metadata import generate_single_metadata
from core.ocr import generate_single_ocr
from core.preprocessing_document import resize_images

class MetaScribeController:
    """
    Main controller for the Metascribe pipeline that orchestrates document processing.
    
    This controller handles the complete workflow from preprocessing documents
    through OCR to metadata generation and aggregation, using configuration from a YAML file.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the controller with configuration.

        Args:
            config_path (str, optional): Path to the YAML configuration file.
                If None, looks in environment and current working directory.
        """

        config_locations = [
            config_path,
            os.environ.get("METASCRIBE_CONFIG"),
            os.path.join(os.getcwd(), "metascribe_config.yaml")
        ]

        for loc in config_locations:
            if loc and os.path.exists(loc):
                try:
                    with open(loc, "r") as f:
                        self.config = yaml.safe_load(f)
                        print(f"Successfully loaded configuration from {loc}")
                        break
                except Exception as e:
                    print(f"Error loading configuration from {loc}: {e}")
                    continue

        if not self.config:
            raise ValueError("""
                No valid configuration found.
                Please provide a configuration file with any of the following options:
                    - Setting the METASCRIBE_CONFIG environment variable
                    - Placing a metascribe_config.yaml file in the current working directory
                    - Providing a path to a custom configuration file as an argument
            """)

    def process_documents(self, input_dir, output_dir=None):
        """
        Process all documents in the input directory and save the results in the output directory.
            - Documents must be in PDF, JPG, or PNG format
            - Output will be saved in a single JSONL per document

        Args:
            input_dir (str): The directory containing user's documents to process.
            output_dir (str): The directory to save the processed documents.
        """

        print(f"Starting MetaScribe pipeline for {input_dir}...")

        # Set up output directory.
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), "generated_llm_metadata")
            print(f"No output directory provided, using default: {output_dir}")

        if os.path.exists(output_dir):
            user_response = input(f"WARNING: Output directory {output_dir} already exists. Continue? (y/n): ")
            if user_response.lower() not in ["y", "yes"]:
                print("MetaScribe pipeline aborted by user.")
                return
            
        os.makedirs(output_dir, exist_ok=True)

        # Get list of files in input directory
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

        # Preprocess files serially.

        # STEP 1: Preprocessing -- resizing & binarization.

        # STEP 2: Metadata generation.

        # STEP 3: OCR.

        # STEP 4: Metadata aggregation.
        
        
        
        
        
