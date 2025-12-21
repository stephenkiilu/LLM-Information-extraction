#%%
import csv
import re
import sys
import os
from collections import defaultdict

class WhiteMatterLocalLookup:
    def __init__(self, obo_path):
        self.obo_path = obo_path
        self.terms = {}          # Map: ID -> Term Data
        self.incoming_map = defaultdict(list) # Map: Target ID -> [(Relation, Source ID)]
        self.id_to_name = {}     # Helper to look up names by ID quickly

        print(f"Loading OBO file from: {self.obo_path}...")
        self.parse_obo()
        print(f"Loaded {len(self.terms)} terms.")

    def parse_obo(self):
        """
        Parses the OBO file line by line to build term dictionary and relationship graph.
        """
        if not os.path.exists(self.obo_path):
            print(f"Error: File not found at {self.obo_path}")
            return

        current_term = {}
        in_term = False

        # Regex patterns for parsing specific lines
        # Matches: relationship: part_of UBERON:123 ! comment
        rel_pattern = re.compile(r'^relationship:\s+(\S+)\s+(\S+)')
        # Matches: is_a: UBERON:123 ! comment
        isa_pattern = re.compile(r'^is_a:\s+(\S+)')

        with open(self.obo_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                if line == '[Term]':
                    # Save previous term if exists
                    if current_term and 'id' in current_term:
                        self._store_term(current_term)

                    # Start new term
                    in_term = True
                    current_term = {
                        'id': None,
                        'name': '',
                        'def': [],
                        'synonyms': [],
                        'xrefs': [],
                        'subset': [],
                        'relationships': [] # List of (relation_type, target_id)
                    }
                    continue

                elif line == '[Typedef]':
                    # We can skip Typedefs for this purpose, or save previous term and stop tracking
                    if current_term and 'id' in current_term:
                        self._store_term(current_term)
                    in_term = False
                    current_term = {}
                    continue

                if not in_term:
                    continue

                # --- Parsing Fields ---

                # ID
                if line.startswith('id:'):
                    current_term['id'] = line.split('id:')[1].strip()

                # Name
                elif line.startswith('name:'):
                    current_term['name'] = line.split('name:')[1].strip()

                # Definition (extracts text inside quotes)
                elif line.startswith('def:'):
                    # simple quote extractor
                    match = re.search(r'"([^"]*)"', line)
                    if match:
                        current_term['def'].append(match.group(1))

                # Synonyms
                elif line.startswith('synonym:'):
                    # format: synonym: "Name" EXACT []
                    match = re.search(r'"([^"]*)"', line)
                    if match:
                        current_term['synonyms'].append(match.group(1))

                # Subsets
                elif line.startswith('subset:'):
                    current_term['subset'].append(line.split(':')[1].strip())

                # Xrefs
                elif line.startswith('xref:'):
                    current_term['xrefs'].append(line.split('xref:')[1].strip().split(' ')[0])

                # Relationships: Is_a (Subclass of)
                elif line.startswith('is_a:'):
                    match = isa_pattern.match(line)
                    if match:
                        target_id = match.group(1)
                        current_term['relationships'].append(('is_a', target_id))

                # Relationships: General (part_of, develops_from, etc)
                elif line.startswith('relationship:'):
                    match = rel_pattern.match(line)
                    if match:
                        rel_type = match.group(1)
                        target_id = match.group(2)
                        current_term['relationships'].append((rel_type, target_id))

            # Store last term
            if current_term and 'id' in current_term:
                self._store_term(current_term)

    def _store_term(self, term):
        """
        Helper to save term to memory and build the 'Related From' graph index.
        """
        term_id = term['id']
        self.terms[term_id] = term
        self.id_to_name[term_id] = term['name']

        # Index relationships for the "Related From" lookup
        for rel_type, target_id in term['relationships']:
            # If Term A is "part_of" Term B...
            # We add to Term B's incoming list: (part_of, Term A)
            self.incoming_map[target_id].append((rel_type, term_id))

    def get_term_name(self, term_id):
        return self.id_to_name.get(term_id, term_id)

    def run(self, input_id):
        """
        Look up a single term by ID and format the data.
        """
        print(f"--- Looking up: {input_id} ---")

        if input_id not in self.terms:
            print(f"ID {input_id} not found in loaded OBO file.")
            return None

        raw_data = self.terms[input_id]

        # Structure matches the API version for consistency
        result = {
            'Id': raw_data['id'],
            'Label': raw_data['name'],
            'Definition': raw_data['def'] or ["N/A"],
            'Synonyms': raw_data['synonyms'],
            'Annotations': {
                'xrefs': raw_data['xrefs'],
                'subset': raw_data['subset']
            },
            'Relationships (Outgoing)': defaultdict(list),
            'Related From (Incoming)': defaultdict(list)
        }

        # Process Outgoing
        for rel_type, target_id in raw_data['relationships']:
            target_name = self.get_term_name(target_id)
            # Convert 'is_a' to human readable if preferred
            if rel_type == 'is_a': rel_type = 'Subclass Of (Is A)'
            result['Relationships (Outgoing)'][rel_type].append(target_name)

        # Process Incoming (The Reverse Graph)
        if input_id in self.incoming_map:
            for rel_type, source_id in self.incoming_map[input_id]:
                source_name = self.get_term_name(source_id)
                # Clean up relation names
                if rel_type == 'is_a': rel_type = 'Subclass Of (Is A)'
                result['Related From (Incoming)'][rel_type].append(source_name)

        return result

    def flatten_entry(self, entry):
        """Flattens dictionary for CSV export"""
        flat = {}
        flat['Id'] = entry.get('Id')
        flat['Label'] = entry.get('Label')

        flat['Synonyms'] = " ; ".join(entry.get('Synonyms', []))
        flat['Definition'] = " | ".join(entry.get('Definition', []))

        for k, v in entry.get('Annotations', {}).items():
            if isinstance(v, list): v = " ; ".join(v)
            flat[f"Annotation: {k}"] = v

        for k, v in entry.get('Relationships (Outgoing)', {}).items():
            flat[f"Outgoing: {k}"] = " ; ".join(v)

        for k, v in entry.get('Related From (Incoming)', {}).items():
            flat[f"Incoming: {k}"] = " ; ".join(v)

        return flat

    def save_to_csv(self, data_list, filename="lookup_results.csv"):
        if not data_list:
            print("No data to save.")
            return

        flattened_data = [self.flatten_entry(entry) for entry in data_list]

        headers = set()
        for item in flattened_data:
            headers.update(item.keys())

        headers = sorted(list(headers))
        if 'Id' in headers: headers.insert(0, headers.pop(headers.index('Id')))
        if 'Label' in headers: headers.insert(1, headers.pop(headers.index('Label')))

        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                writer.writeheader()
                writer.writerows(flattened_data)
            print(f"Successfully saved results to {filename}")
        except IOError as e:
            print(f"Error saving CSV: {e}")

original_path = "data/raw/uberon.obo"
tool = WhiteMatterLocalLookup(original_path)
tool.get_term_name("UBERON:0002037")
tool.run("UBERON:0002037")
tool.parse_obo()

uberon_ids = [
    'UBERON:0003961', 'UBERON:0000935', 'UBERON:0034746', 'UBERON:0035937', 
    'UBERON:0007416', 'UBERON:0002623', 'UBERON:0003961', 'UBERON:0004682', 
    'UBERON:0002336', 'UBERON:0022271', 'UBERON:0002707', 'UBERON:0004545', 
    'UBERON:0014528', 'UBERON:0034678', 'UBERON:0034676', 'UBERON:0000052', 
    'UBERON:0022430', 'UBERON:0034753', 'UBERON:0034743', 'UBERON:0001887', 
    'UBERON:0002794', 'UBERON:0002265', 'UBERON:0022264', 'UBERON:0035931', 
    'UBERON:0022250', 'UBERON:0022246', 'UBERON:0000373', 'UBERON:0034745', 
    'UBERON:0003044'
]

# --- Main Execution ---
if __name__ == "__main__":
    # UPDATE THIS PATH to point to your actual file
    original_path = "data/raw/uberon.obo"

    # Initialize the parser (this handles the loading)
    # Note: This might take a few seconds depending on file size (Uberon is large)
    tool = WhiteMatterLocalLookup(original_path)

    ids_to_query = [
        "UBERON:0015510",  # Body of corpus callosum
        "UBERON:0002336",  # Corpus callosum layer/structure
    ]

    results = []
    for term_id in ids_to_query:
        lookup_result = tool.run(term_id)
        if lookup_result:
            results.append(lookup_result)

    tool.save_to_csv(results, "white_matter_lookup_local.csv")