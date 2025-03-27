# MetaScribe
![image](https://github.com/user-attachments/assets/cd60a64c-e60b-4555-97e0-e308c3488b9c)

![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

> An LLM-powered tool for extracting structured metadata from digital collections

## Overview

MetaScribe automates metadata creation for libraries and archives by extracting (structured) information from digitized documents using modern large language models. Process documents page-by-page with flexible extraction schemas, all defined by your needs.

## Features

- Structured customizable metadata schemas
- Page-by-page extraction
- Preview and batch processing for metadata creation
- JSONL output format

## How It Works

1. **Upload Your Documents**: Drop your files into the input directory — organize by collection or batch
2. **Define Your Schema**: Specify what metadata you want using a customizable JSON Schema
3. **Preview the Output**: Run a quick sample to check accuracy and adjust before full processing
4. **Extract Metadata**: Process your full collection with a single command — fast and scalable
5.** Export in JSONL**: Get structured metadata for each page, ready to use or archive

## Basic Usage

```python
from metascribe import MetadataExtractor

# Initialize
extractor = MetadataExtractor(api_key="your_llm_api_key")

# Load schema
extractor.load_schema("schemas/basic_schema.json")

# Preview single document
preview = extractor.generate_once("document.pdf")

# Process collection
results = extractor.run_full("document_collection/")

# Export
extractor.export_results("metadata.jsonl")
```

## Requirements

- Python 3.8+
- Access to LLM API (OpenAI, etc.)
