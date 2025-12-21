#%%
"""White matter tract extraction from research papers using OpenAI API."""

import json
import csv
import time
from typing import List, Dict, Any, Tuple


from openai import OpenAI
from dotenv import load_dotenv

from config import OPENAI_API_KEY
from prompts.brain_extraction import SYSTEM_PROMPT

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Constants
WHITEMATTER_JSON_PATH = "data/processed/whitematter_full_data.json"
DEFAULT_OUTPUT_CSV = "data/processed/whitematter_full_predicted_data_GPT_5_mini.csv"

# Define all extraction fields
EXTRACTION_FIELDS = [
    "subjects",
    "patient_groups",
    "imaging_modalities",
    "whitematter_tracts",
    "analysis_software",
    "study_type",
    "diffusion_measures",
    "template_space",
    "results_method",
    "white_integrity",
    "question_of_study",
    "DTI_study",
    "Human_study",
    "Dementia_study",
    "Disease_study"
]

# CSV fieldnames (paper metadata + extraction fields)
CSV_FIELDNAMES = ["PMID", "title"] + EXTRACTION_FIELDS

# Load white matter data
with open(WHITEMATTER_JSON_PATH, "r", encoding="utf-8") as f:
    whitematter_json = json.load(f)

def _get_paper_field(paper: Dict[str, Any], field: str) -> str:
    """Extract and convert a paper field to string, handling None values."""
    # Check metadata first
    metadata = paper.get("metadata", {})
    if field in metadata:
        value = metadata.get(field)
    else:
        value = paper.get(field)
    return str(value) if value is not None else ""


def _clean_content(content: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively clean content structure of the paper.
    Removes empty sections and subsections.
    """
    cleaned = {}
    
    for section_title, section_data in content.items():
        new_section = {}
        
        # Keep text if present and not empty/None
        if "text" in section_data and section_data["text"]:
            new_section["text"] = section_data["text"]
            
        # Process subsections recursively
        if "subsections" in section_data and isinstance(section_data["subsections"], dict):
            cleaned_subs = _clean_content(section_data["subsections"])
            if cleaned_subs:
                new_section["subsections"] = cleaned_subs
        
        # Only add section if it has content (text or subsections)
        if new_section:
            cleaned[section_title] = new_section
            
    return cleaned


def process_data(paper: Dict[str, Any], processing_mode: int) -> Tuple[str, List[str]]:
    """
    Preprocess paper data according to the selected processing mode.
    
    Args:
        paper: Dictionary containing paper details
        processing_mode: Processing strategy (Only mode 1 used for this version)
    
    Returns:
        Tuple of (combined_data_json_string, body_chunks)
    """
    # 1. Prepare Metadata (excluding authors)
    metadata = paper.get("metadata", {}).copy()
    if "authors" in metadata:
        del metadata["authors"]
    
    # Ensure title/abstract are present if missing in metadata but in root (fallback)
    if "title" not in metadata and "title" in paper:
        metadata["title"] = paper["title"]
    if "abstract" not in metadata and "abstract" in paper:
        metadata["abstract"] = paper["abstract"]

    # 2. Prepare Content (Cleaned Structure)
    content = paper.get("content", {})
    cleaned_content = _clean_content(content)
    
    # 3. Construct Final Structure
    final_structure = {
        "metadata": metadata,
        "body": cleaned_content
    }
    
    # Convert to JSON string
    data = json.dumps(final_structure, ensure_ascii=False)
    
    return data, []


def extract_one(WM_paper: Dict[str, Any], 
                model: str = "gpt-4o-mini", 
                processing_mode: int = 1) -> Dict[str, Any]:
    """
    Extract structured data from a single paper using OpenAI API.
    
    Args:
        WM_paper: Dictionary containing paper details
        model: OpenAI model identifier (default: gpt-4o-mini)
        processing_mode: Data processing mode (ignored, always full text)
    
    Returns:
        Dictionary containing extracted fields with lists of values
    """
    # Force processing mode 1 logic implicitly via new process_data
    data, _ = process_data(WM_paper, processing_mode)

    # Prepare chunks for processing (single chunk)
    chunks = [data]

    # Initialize results dictionary
    all_data = {field: [] for field in EXTRACTION_FIELDS}

    # Process each chunk through OpenAI API
    for chunk in chunks:
        chunk_result = _process_chunk_with_api(chunk, model)
        _merge_chunk_results(all_data, chunk_result)

    # Remove duplicates from aggregated results
    for key in all_data:
        all_data[key] = list(set(all_data[key]))

    return all_data


def _process_chunk_with_api(chunk: str, model: str) -> Dict[str, Any]:
    """Send a text chunk to OpenAI API and parse the response."""
    user_payload = {"body": chunk}
    
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=1,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
            ]
        )
        
        content = resp.choices[0].message.content
        return json.loads(content)
    
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Raw content: {content}")
        return {}
    except Exception as e:
        print(f"API error: {e}")
        return {}


def _merge_chunk_results(all_data: Dict[str, List], chunk_result: Dict[str, Any]) -> None:
    """Merge results from a single chunk into the aggregated results."""
    for key in all_data:
        value = chunk_result.get(key)
        if isinstance(value, list):
            all_data[key].extend(value)
        elif value:
            all_data[key].append(value)


def _build_csv_row(paper: Dict[str, Any], extracted_data: Dict[str, List]) -> Dict[str, str]:
    """Build a CSV row from paper metadata and extracted data."""
    # pmcid is not present in the new structure, defaulting to empty
    # title is in metadata
    title = _get_paper_field(paper, "title")
    pmid = _get_paper_field(paper, "PMID")
    
    row = {
        "PMID": pmid, 
        "title": title
    }
    
    # Add extracted fields with semicolon-separated values
    for field in EXTRACTION_FIELDS:
        row[field] = ";".join(extracted_data.get(field, []))
    
    return row


#gpt-4o-mini
def extract_all(WM_papers: List[Dict[str, Any]],
                out_csv: str = DEFAULT_OUTPUT_CSV,
                model: str = "gpt-4o-mini",
                sleep_sec: float = 0.001,
                processing_mode: int = 1) -> List[Dict[str, str]]:
    """
    Extract data from multiple papers and save results to CSV.
    
    Args:
        WM_papers: List of paper dictionaries to process
        out_csv: Output CSV file path
        model: OpenAI model identifier
        sleep_sec: Delay between API calls to avoid rate limits
        processing_mode: Data processing mode (1=all, 2=chunked, 3=metadata only)
    
    Returns:
        List of CSV row dictionaries
    """
    results = []
    total_papers = len(WM_papers)
    
    for i, paper in enumerate(WM_papers, 1):
        # Extract data from paper
        extracted_data = extract_one(paper, model=model, processing_mode=processing_mode)
        
        # Build CSV row
        row = _build_csv_row(paper, extracted_data)
        print(row)
        results.append(row)
        
        print(f"Processed {i}/{total_papers}: {row.get('PMID', 'Unknown')}")
        time.sleep(sleep_sec)
       
    
    # Write results to CSV
    _write_results_to_csv(results, out_csv)
    print(f"✅ Successfully saved {len(results)} records to {out_csv}")

    return results

def _write_results_to_csv(results: List[Dict[str, str]], output_path: str) -> None:
    """Write extraction results to a CSV file."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(results)
    
def main():
    """Main execution function."""
    print(f"Starting extraction for {len(whitematter_json)} papers...")
    extract_all(whitematter_json, processing_mode=1)
    print("Extraction complete!")


if __name__ == "__main__":
    # main()
    extract_all(whitematter_json)

