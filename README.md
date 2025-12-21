# Data Extraction from Papers using LLMs

An automated tool to extract structured information from brain imaging research papers using OpenAI's LLMs.

## Overview
This system processes research papers (PDF-extracted text) to extract key parameters like imaging modalities, patient groups, white matter tracts, and subjects.

## Key Features
- **LLM-Powered Extraction**: Uses GPT-4o-mini for high-accuracy metadata extraction.
- **Multiple Modes**: Supports full-text processing, chunked processing (for long papers), and metadata-only extraction.
- **Batch Processing**: Efficiently handles multiple documents with built-in rate limiting.

## Setup
1. **Repository**:
   ```bash
   git clone https://github.com/stephenkiilu/Data-extraction-from-paper-using-LLMs.git
   ```
2. **Environment**:
   - Install dependencies: `pip install -r requirements.txt`
   - Configure `.env` with `OPENAI_API_KEY`.

## Usage
Run the main extraction script:
```python
python main1.py
```

## Structure
- `main.py` / `main1.py`: Core extraction logic.
- `prompts/`: LLM system prompts.
- `evaluation.py`: Tools for verifying extraction accuracy.
