# %%
import os
import pandas as pd
import json
import glob


# Paths for data
RAW_DATA_PATH = 'data/raw'  # Raw data 
PROCESSED_DATA_PATH = 'data/processed'  # Processed data directory
RAW_JSON_DIR = 'data/raw/WM_full_data_json' # Raw JSON data directory


os.path.dirname(os.path.dirname(__file__))

def get_file_path(file_name: str, folder: str = 'raw') -> str:
    """Generate and return the absolute file path for the data."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', folder, file_name)

def load_raw_data(file_name: str) -> pd.DataFrame:
    """Load raw data from the specified file."""
    file_path = get_file_path(file_name, 'raw')
    return pd.read_csv(file_path)

def save_processed_data(df: pd.DataFrame, file_name: str):
    """Save processed data to the specified processed data directory."""
    file_path = get_file_path(file_name, 'processed')
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")

def process_data():
    """Main data processing function."""
    # Load raw data
    WM_data = load_raw_data('WM_data.csv')  # Read raw data
      
    # Extract PMCID column
    pmcids = WM_data['PMCID']  

    # Strip 'PMC' from the PMCID column
    pmcids = pmcids.str.replace('PMC', '', regex=False)  # Remove 'PMC' prefix

    # Save cleaned pmcids to processed folder --- myfile of all pmcids i.e pmcid.txt
    pmcids.to_csv(get_file_path('pmcid.txt', 'processed'), index=False, header=False)  # Save to processed folder

    # Read pmcid.txt, clean the PMC IDs and save them
    pmcid_ids = pd.read_csv(get_file_path('pmcid.txt', 'processed'), header=None, names=["pmcid"])
    save_processed_data(pmcid_ids, 'pmcids.csv')

    # Read extracted data
    text_csv = pd.read_csv(get_file_path('text.csv', 'raw')) # this is the data from the pubget, but it is reshuffled, I want to put it in the order of my validation set
    ordered_ids = pd.read_csv(get_file_path('pmcids.csv', 'processed'))

    # Merge the data on PMCID - the data is to be ordered in the same way as my validation set
    ordered_text = ordered_ids.merge(
        text_csv[["pmcid", "title", "keywords", "abstract", "body"]],
        on="pmcid",
        how="left",
        validate="1:1"
    )

    # Save ordered text data to processed folder
    save_processed_data(ordered_text, 'Whitematter_data.csv')

def generate_json_file(file_name: str, output="whitematter_data.json"):
    """Generate a JSON file from the processed data."""
    file_name = get_file_path(file_name, 'processed')
    data = pd.read_csv(file_name)
    whitematter_pmcid = data['pmcid'].tolist()
    whitematter_title = data['title'].tolist()
    whitematter_keyword = data['keywords'].tolist()  # Not used but extracted
    whitematter_abstract = data['abstract'].tolist()
    whitematter_body = data['body'].tolist()

    # Prepare the data for JSON
    whitematter_json = []
    for pmcid, abstract, title, keyword, body in zip(whitematter_pmcid, whitematter_abstract, whitematter_title, whitematter_keyword, whitematter_body):
        whitematter_json.append({
            "PMID": pmcid,
            "title": title,
            "keywords": keyword,
            "abstract": abstract,
            "body": body
        })

    # Save JSON to output file
    output_path = get_file_path(output, "processed")  # Save to processed folder
    with open(output_path, "w") as f:
        json.dump(whitematter_json, f, indent=4, ensure_ascii=False)

    print(f"JSON data saved to {output_path}")
    print(f"JSON data saved to {output_path}")

def consolidate_json_files(output_filename='whitematter_full_data.json'):
    """
    Consolidate all JSON files from the RAW_JSON_DIR into a single list
    and save it to the processed folder.
    """
    json_files = glob.glob(os.path.join(get_file_path('', 'raw'), '..', 'WM_full_data_json', '*.json'))
    # Adjust path because get_file_path defaults to data/raw, but we need data/raw/WM_full_data_json
    # Actually, let's fix get_file_path usage or just construct path directly for clarity relative to project root
    # Using the constant RAW_JSON_DIR relative to project root:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_dir = os.path.join(base_dir, RAW_JSON_DIR)
    
    # Sort files numerically by prefix (e.g. 1_x.json before 10_x.json)
    def numerical_sort_key(filepath):
        basename = os.path.basename(filepath)
        prefix = basename.split('_')[0]
        try:
            return int(prefix)
        except ValueError:
            return 999999
            
    json_files = sorted(glob.glob(os.path.join(json_dir, '*.json')), key=numerical_sort_key)
    print(f"Found {len(json_files)} JSON files to consolidate.")
    
    consolidated_data = []
    
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                # Rename pmcid to PMID
                if "pmcid" in data:
                    data["PMID"] = data.pop("pmcid")
                if "metadata" in data and "pmcid" in data["metadata"]:
                    data["metadata"]["PMID"] = data["metadata"].pop("pmcid")
                
                # Reorder to ensure PMID is first in metadata and removed from root
                
                # 1. Ensure PMID in metadata
                pmid_val = data.get("PMID") or (data.get("metadata", {}).get("PMID"))

                # Fallback: Extract from filename
                if not pmid_val:
                    basename = os.path.basename(file_path)
                    try:
                        name_parts = os.path.splitext(basename)[0].split('_')
                        if len(name_parts) >= 2:
                            pmid_val = name_parts[-1]
                    except Exception:
                        pass

                if pmid_val:
                    if "metadata" not in data:
                        data["metadata"] = {}
                    data["metadata"]["PMID"] = str(pmid_val)
                    
                # 2. Remove PMID from root if exists
                if "PMID" in data:
                    del data["PMID"]

                # 3. Helper to reorder a dict (PMID then title then others)
                def reorder_metadata(d):
                    new_d = {}
                    if "PMID" in d:
                        new_d["PMID"] = d["PMID"]
                    if "title" in d:
                        new_d["title"] = d["title"]
                    for k, v in d.items():
                        if k not in ["PMID", "title"]:
                            new_d[k] = v
                    return new_d

                if "metadata" in data and isinstance(data["metadata"], dict):
                    data["metadata"] = reorder_metadata(data["metadata"])
                
                consolidated_data.append(data)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            
    # Save the consolidated list
    output_path = get_file_path(output_filename, 'processed')
    with open(output_path, 'w') as f:
        json.dump(consolidated_data, f, indent=4, ensure_ascii=False)
        
    print(f"Successfully consolidated {len(consolidated_data)} records to {output_path}")

# process_data()
# generate_json_file('Whitematter_data.csv')

