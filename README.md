# LLM Information Extraction from Neuroimaging Papers

An automated pipeline for extracting structured metadata from brain imaging research papers using OpenAI's large language models (LLMs).

## Overview

This system processes research papers (PDF-extracted text) to extract key biomedical parameters such as imaging modalities, patient groups, white matter tracts studied, and subject demographics. It supports both full-text and abstract-only processing, with and without look-up table (LUT) guidance.

## Key Features

- **LLM-Powered Extraction**: Uses GPT-4o-mini and GPT-5-mini for high-accuracy metadata extraction.
- **Multiple Processing Modes**: Full-text, abstract-only, and LUT-guided extraction.
- **Batch Processing**: Efficiently handles multiple documents with built-in rate limiting.
- **Evaluation Suite**: F1 and Jaccard accuracy scoring across all extracted fields, with grouped comparison plots.
- **White Matter Analysis**: Distribution and error analysis tools for WM tract extraction results.

## Repository Structure

```
├── main.py                          # Core extraction pipeline
├── prompts/                         # LLM system prompts (with & without LUT)
├── evaluation_gpt4_vs_gpt5.py      # Compare GPT-4o-mini vs GPT-5-mini F1 scores
├── evaluation_lut.py                # Evaluate LUT vs no-LUT extraction
├── evaluation_full_vs_abstract.py  # Full-text vs abstract comparison
├── whitematter_distributions.py     # WM tract distribution analysis
├── whitematter_error_analysis.py    # WM tract error breakdown
├── data/
│   ├── raw/                         # Raw input data (CSVs, XML article sets)
│   └── processed/                   # Model outputs, F1 results, and plots
```

## Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/stephenkiilu/LLM-Information-extraction.git
   cd LLM-Information-extraction
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**:
   - Create a `.env` file and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_key_here
   ```

## Usage

Run the main extraction script:
```bash
python main.py
```

Run evaluation comparisons:
```bash
python evaluation_gpt4_vs_gpt5.py
python evaluation_lut.py
python evaluation_full_vs_abstract.py
```

## Models Evaluated

| Model | Mode |
|-------|------|
| GPT-4o-mini | Full-text, Abstract, LUT |
| GPT-5-mini | Full-text, Abstract, LUT |
