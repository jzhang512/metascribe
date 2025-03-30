#!/usr/bin/env python3
"""
MetaScribe - Document processing pipeline for metadata extraction and OCR

This script serves as the main entry point for the MetaScribe application.
"""

import sys
import argparse
from src.controller import MetaScribeController

__version__ = "1.0.0"

def main():
    """Main entry point for the MetaScribe application."""
    parser = argparse.ArgumentParser(
            description="MetaScribe: An LLM-powered tool for extracting structured metadata from digital collections",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
# Process documents with default settings
python run.py input_directory

# Process documents with custom output directory
python run.py input_directory -o custom_output

# Process documents with custom config
python run.py input_directory -c my_config.yaml
            """
    )
    parser.add_argument(
            "input_dir",
            help="Directory containing documents to process (PDF, JPG, PNG only)"
    )
    parser.add_argument(
            "-o", "--output-dir",
            help="Directory to save processed results (default: ./metascribe_output)",
            default="./metascribe_output"
    )
    parser.add_argument(
            "-c", "--config",
            help="Path to custom config file (default: ./metascribe_config.yaml)",
            default="./metascribe_config.yaml"
    )
    parser.add_argument(
            "-v", "--version",
            action="version",
            version=f"MetaScribe {__version__}"
    )
    
    args = parser.parse_args()
    
    try:
        controller = MetaScribeController(config_path=args.config)
        
        # Process documents
        controller.process_documents(
            input_dir=args.input_dir,
            output_dir=args.output_dir
        )
        
        return 0  # Success 
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1  # Error 

if __name__ == "__main__":
    sys.exit(main())
