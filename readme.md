# MetaScribe

![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

> An LLM-powered tool for extracting structured metadata from digital collections

## Overview

MetaScribe automates metadata creation for libraries and archives by extracting (structured) information from digitized documents using language models. Process documents page-by-page with customizable extraction schemas.

## Features

- Customizable metadata schemas
- Page-by-page extraction
- Preview and batch processing for metadata creation
- JSONL output format

## How It Works

1. **Define Schema**: Organize your extraction needs into a criterion of sections with specific questions
2. **Review Prompts**: System generates LLM prompts from your schema
3. **Preview**: Test on sample documents
4. **Process**: Run extraction on full collection
5. **Export**: Save metadata in JSONL format

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

## License

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
