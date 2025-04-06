# MetaScribe
![image](https://github.com/user-attachments/assets/8ed98e8c-9656-4e7c-8419-97642897341f)

![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

> An LVLM-powered tool for extracting structured metadata from digital collections

## Overview

MetaScribe automates metadata creation for libraries and archives by extracting (structured) information from digitized documents using modern large language models. Process documents page-by-page with flexible extraction schemas, all defined by your needs.

## Features

- Customizable metadata schemas
- Configurable pipeline
- Page-by-page extraction
- Batch processing
- Structured JSONL output format

## How It Works

1. **Upload Your Documents**: Drop your files into the input directory — organize by collection or batch
2. **Define Your Schema**: Specify what metadata you want using a customizable JSON Schema
3. **Preview the Output**: Run a quick sample to check accuracy and adjust before full processing
4. **Extract Metadata**: Process your full collection with a single command — fast and scalable
5. **Export in JSONL**: Get structured metadata for each page, ready to use or archive

## Installation

```console
# Clone the repository
git clone https://github.com/jzhang512/metascribe.git
cd metascribe

# Install dependencies
pip install -r requirements.txt

# Set up environment variables for API keys (in .env file) or like this:
export OPENAI_API_KEY=your_openai_key
export ANTHROPIC_API_KEY=your_anthropic_key
export TOGETHER_API_KEY=your_together_key
export GOOGLE_API_KEY=your_google_key
```

## Setup & Execution

1. **Prepare Required Files**:
   - Configure extraction settings through a YAML configuration file
   - Define your metadata structure with a custom JSON Schema

2. **Run the Extraction**:
```console
# Run with default settings
python run.py input_directory

# Specify custom output location
python run.py input_directory -o custom_output

# Use custom configuration
python run.py input_directory -c my_config.yaml
```

## Important Notes

- **Output Files**: Manually editing output files (`manifest.json`, metadata files) is not supported and may cause unexpected behavior or pipeline failures. These files contain structured data that the library depends on for proper operation.

- **Resume Processing**: MetaScribe tracks progress in the `manifest.json` file. So, when running MetaScribe multiple times on the same data, the run will resume from where processing stopped (or previously had errors).

- **Error Handling**: MetaScribe notes errors in the `manifest.json` file and continues processing remaining documents when possible. 

- **Aggregation**: Currently, MetaScribe can only aggregate the metadata fields defined under the `'properties'` section of your JSON Schema.

## Requirements

- Python 3.11+
- Access to LLM API Keys (OpenAI, Anthropic, etc.). MetaScribe works with the following LLM models:

```python
SUPPORTED_MODELS = [
    # OpenAI Models
    "o1-2024-12-17",
    "gpt-4.5-preview-2025-02-27",
    "gpt-4o-2024-11-20",
    "gpt-4o-mini-2024-07-18",
    "gpt-4-turbo-2024-04-09",
    
    # Anthropic Models
    "claude-3-7-sonnet-20250219",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "claude-3-opus-20240229",
    "claude-3-haiku-20240307",
    
    # Google Models
    "gemini-2.0-flash-001",
    "gemini-2.0-flash-lite-001",
    "gemini-1.5-flash-002",
    "gemini-1.5-flash-8b-001",
    "gemini-1.5-pro-002",
    
    # Together Models
    "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
    "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
    "Qwen/Qwen2-VL-72B-Instruct"
]
```
