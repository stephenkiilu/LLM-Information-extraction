#%%
"""White matter tract extraction from research papers using OpenAI API.

Two processing modes are supported:
  - ProcessingMode.ABSTRACT   : uses only the title + abstract
  - ProcessingMode.FULL_TEXT  : uses title, abstract, and the full cleaned body
"""

import json
import csv
import time
from enum import Enum
from typing import List, Dict, Any, Tuple

from openai import OpenAI
from dotenv import load_dotenv

from config import OPENAI_API_KEY1
from prompts.brain_extraction_no_lut import SYSTEM_PROMPT

print(SYSTEM_PROMPT)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY1)

# ── Constants ──────────────────────────────────────────────────────────────────
WHITEMATTER_JSON_PATH = "data/processed/whitematter_full_data.json"

OUTPUT_CSV = {
    "abstract":  "data/processed/whitematter_no_lut_predicted_data_GPT_5_mini.csv",
    "full_text": "data/processed/whitematter_no_lut_predicted_data_GPT_5_mini.csv",
}


# ── Processing mode ────────────────────────────────────────────────────────────
class ProcessingMode(str, Enum):
    ABSTRACT  = "abstract"   # title + abstract only
    FULL_TEXT = "full_text"  # title + abstract + full body content

# ── Extraction fields ──────────────────────────────────────────────────────────
EXTRACTION_FIELDS = [
    "whitematter_tracts"
]

CSV_FIELDNAMES = ["PMID", "title"] + EXTRACTION_FIELDS

# ── Load data ──────────────────────────────────────────────────────────────────
with open(WHITEMATTER_JSON_PATH, "r", encoding="utf-8") as f:
    whitematter_json = json.load(f)


# ── Helper utilities ───────────────────────────────────────────────────────────

def _get_paper_field(paper: Dict[str, Any], field: str) -> str:
    """Extract and convert a paper field to string, handling None values."""
    metadata = paper.get("metadata", {})
    value = metadata.get(field) if field in metadata else paper.get(field)
    return str(value) if value is not None else ""


def _clean_content(content: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively clean the body content structure.
    Removes empty sections and subsections.
    """
    cleaned = {}
    for section_title, section_data in content.items():
        new_section = {}
        if "text" in section_data and section_data["text"]:
            new_section["text"] = section_data["text"]
        if "subsections" in section_data and isinstance(section_data["subsections"], dict):
            cleaned_subs = _clean_content(section_data["subsections"])
            if cleaned_subs:
                new_section["subsections"] = cleaned_subs
        if new_section:
            cleaned[section_title] = new_section
    return cleaned


# ── Payload builders ───────────────────────────────────────────────────────────

def _build_abstract_payload(paper: Dict[str, Any]) -> str:
    """
    Build a JSON payload containing ONLY the title and abstract.
    The full body/content is intentionally excluded.
    """
    metadata = paper.get("metadata", {})
    title    = metadata.get("title")    or paper.get("title", "")
    abstract = metadata.get("abstract") or paper.get("abstract", "")

    return json.dumps({"title": title, "abstract": abstract}, ensure_ascii=False)


def _build_fulltext_payload(paper: Dict[str, Any]) -> str:
    """
    Build a JSON payload containing the title, abstract (via metadata),
    and the full cleaned body content.
    """
    metadata = paper.get("metadata", {}).copy()
    metadata.pop("authors", None)  # strip author list to save tokens

    # Fallback: ensure title/abstract are in metadata
    if "title" not in metadata and "title" in paper:
        metadata["title"] = paper["title"]
    if "abstract" not in metadata and "abstract" in paper:
        metadata["abstract"] = paper["abstract"]

    content         = paper.get("content", {})
    cleaned_content = _clean_content(content)

    final_structure = {
        "metadata": metadata,
        "body":     cleaned_content,
    }
    return json.dumps(final_structure, ensure_ascii=False)


def prepare_payload(paper: Dict[str, Any], mode: ProcessingMode) -> str:
    """
    Return the appropriate JSON payload string for the given processing mode.

    Args:
        paper: Dictionary containing paper details
        mode:  ProcessingMode.ABSTRACT or ProcessingMode.FULL_TEXT

    Returns:
        JSON string ready to be sent to the OpenAI API
    """
    if mode == ProcessingMode.ABSTRACT:
        return _build_abstract_payload(paper)
    elif mode == ProcessingMode.FULL_TEXT:
        return _build_fulltext_payload(paper)
    else:
        raise ValueError(f"Unknown processing mode: {mode!r}")



# ── API interaction ────────────────────────────────────────────────────────────

def _process_chunk_with_api(chunk: str, model: str) -> Dict[str, Any]:
    """Send a text chunk to the OpenAI API and parse the JSON response."""
    user_payload = {"body": chunk}
    content = ""
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=1,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
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
    """Merge results from a single API response into the aggregated results dict."""
    for key in all_data:
        value = chunk_result.get(key)
        if isinstance(value, list):
            all_data[key].extend(value)
        elif value:
            all_data[key].append(value)


# ── Single-paper extraction ────────────────────────────────────────────────────

def extract_one(WM_paper: Dict[str, Any],
                model: str = "gpt-5-mini",
                mode: ProcessingMode = ProcessingMode.ABSTRACT) -> Dict[str, Any]:
    """
    Extract structured data from a single paper using the OpenAI API.

    Args:
        WM_paper: Dictionary containing paper details
        model:    OpenAI model identifier (default: gpt-4o-mini)
        mode:     ProcessingMode.ABSTRACT  → title + abstract only
                  ProcessingMode.FULL_TEXT → title + abstract + full body

    Returns:
        Dictionary containing extracted fields with lists of values
    """
    data = prepare_payload(WM_paper, mode)

    all_data = {field: [] for field in EXTRACTION_FIELDS}
    chunk_result = _process_chunk_with_api(data, model)
    _merge_chunk_results(all_data, chunk_result)

    # Deduplicate
    for key in all_data:
        all_data[key] = list(set(all_data[key]))

    return all_data


# ── CSV helpers ────────────────────────────────────────────────────────────────

def _build_csv_row(paper: Dict[str, Any], extracted_data: Dict[str, List]) -> Dict[str, str]:
    """Build a CSV row from paper metadata and extracted data."""
    row = {
        "PMID":  _get_paper_field(paper, "PMID"),
        "title": _get_paper_field(paper, "title"),
    }
    for field in EXTRACTION_FIELDS:
        row[field] = ";".join(extracted_data.get(field, []))
    return row


def _write_results_to_csv(results: List[Dict[str, str]], output_path: str) -> None:
    """Write extraction results to a CSV file."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(results)


# ── Batch extraction ───────────────────────────────────────────────────────────

def extract_all(WM_papers: List[Dict[str, Any]],
                out_csv: str | None = None,
                model: str = "gpt-5-mini",
                sleep_sec: float = 0.001,
                mode: ProcessingMode = ProcessingMode.ABSTRACT) -> List[Dict[str, str]]:
    """
    Extract data from multiple papers and save results to CSV.

    Args:
        WM_papers:  List of paper dictionaries to process
        out_csv:    Output CSV file path (auto-selected from mode if None)
        model:      OpenAI model identifier
        sleep_sec:  Delay between API calls to avoid rate limits
        mode:       ProcessingMode.ABSTRACT  → abstract-only extraction
                    ProcessingMode.FULL_TEXT → full-text extraction

    Returns:
        List of CSV row dictionaries
    """
    if out_csv is None:
        out_csv = OUTPUT_CSV[mode.value]

    results      = []
    total_papers = len(WM_papers)
    mode_label   = "ABSTRACT-ONLY" if mode == ProcessingMode.ABSTRACT else "FULL-TEXT"

    print(f"Starting {mode_label} extraction for {total_papers} papers...")
    print(f"Output → {out_csv}\n")

    for i, paper in enumerate(WM_papers, 1):
        extracted_data = extract_one(paper, model=model, mode=mode)
        row = _build_csv_row(paper, extracted_data)
        print(row)
        results.append(row)
        print(f"Processed {i}/{total_papers}: {row.get('PMID', 'Unknown')}")
        time.sleep(sleep_sec)

    _write_results_to_csv(results, out_csv)
    print(f"\n✅ Successfully saved {len(results)} records to {out_csv}")
    return results


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Choose your mode here ──────────────────────────────────────────────────
    #
    #   ProcessingMode.ABSTRACT   → sends only title + abstract to the API
    #   ProcessingMode.FULL_TEXT  → sends title + abstract + full body content
    #
    extract_all(whitematter_json, mode=ProcessingMode.FULL_TEXT)
    # extract_all(whitematter_json, mode=ProcessingMode.FULL_TEXT)
