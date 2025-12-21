#%%
import requests
import json
import urllib.parse
import csv
import os

class WhiteMatterLookup:
    def __init__(self):
        self.base_url = "https://www.ebi.ac.uk/ols4/api"
        self.ontology = "uberon"
        
        # Map common relation IRIs to readable names
        self.RELATION_MAP = {
            "hierarchicalParents": "Subclass Of (Is A)",
            "http://purl.obolibrary.org/obo/BFO_0000050": "part of",
            "http://purl.obolibrary.org/obo/RO_0002202": "develops from",
            "http://purl.obolibrary.org/obo/BSPO_0000096": "anterior to",
            "http://purl.obolibrary.org/obo/BSPO_0000097": "posterior to",
            "http://purl.obolibrary.org/obo/BSPO_0000098": "superior to",
            "http://purl.obolibrary.org/obo/BSPO_0000099": "inferior to",
            "http://purl.obolibrary.org/obo/RO_0002131": "overlaps",
            "http://purl.obolibrary.org/obo/RO_0002160": "only in taxon",
            "http://purl.obolibrary.org/obo/RO_0001025": "located in",
            "http://purl.obolibrary.org/obo/RO_0002220": "adjacent to",
            "http://purl.obolibrary.org/obo/RO_0002298": "results in morphogenesis of",
            "http://purl.obolibrary.org/obo/RO_0002299": "results in development of"
        }

    def get_iri_from_id(self, term_id):
        search_url = f"{self.base_url}/search"
        params = {
            "q": term_id,
            "ontology": self.ontology,
            "exact": "true",
            "queryFields": "obo_id,short_form,label"
        }
        try:
            response = requests.get(search_url, params=params)
            response.raise_for_status()
            data = response.json()
            if data['response']['numFound'] > 0:
                return data['response']['docs'][0]['iri']
            print(f"No IRI found for ID '{term_id}'")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Error resolving ID: {e}")
            return None

    def get_term_details(self, iri):
        encoded_iri = urllib.parse.quote(urllib.parse.quote(iri, safe=''), safe='')
        url = f"{self.base_url}/ontologies/{self.ontology}/terms/{encoded_iri}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching details: {e}")
            return None

    def get_term_graph(self, iri):
        """
        Fetches the graph structure (nodes and edges) for the term.
        This is essential for finding INCOMING relationships ('Related from').
        """
        encoded_iri = urllib.parse.quote(urllib.parse.quote(iri, safe=''), safe='')
        url = f"{self.base_url}/ontologies/{self.ontology}/terms/{encoded_iri}/graph"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching graph: {e}")
            return None

    def get_readable_relation(self, rel_iri):
        """Helper to map IRIs to names"""
        if rel_iri in self.RELATION_MAP:
            return self.RELATION_MAP[rel_iri]
        # Fallback: extract last part of URL
        return rel_iri.split('/')[-1].replace('_', ' ')

    def extract_data(self, term_data, graph_data, focus_iri):
        if not term_data:
            return None

        result = {}

        # --- 1. Core Info ---
        result['Label'] = term_data.get('label')
        result['Id'] = term_data.get('obo_id')
        result['IRI'] = term_data.get('iri')
        
        # --- 2. Definitions ---
        defs = term_data.get('description', [])
        annotation = term_data.get('annotation', {})
        if 'definition' in annotation:
            defs.extend(annotation['definition'])
        result['Definition'] = list(set(defs)) if defs else ["N/A"]

        # --- 3. Synonyms ---
        result['Synonyms'] = term_data.get('synonyms', [])

        # --- 4. Annotations (Xrefs, etc) ---
        custom_annotations = {k: v for k, v in annotation.items() if k != 'definition'}
        result['Annotations'] = custom_annotations

        # --- 5. Outgoing Relationships (What this term IS) ---
        links = term_data.get('_links', {})
        outgoing = {}
        ignored_links = {'self', 'first', 'last', 'prev', 'next', 'collection', 'jstree', 'graph', 'curies', 'ontology'}

        for link_key, link_value in links.items():
            if link_key in ignored_links: continue
            
            rel_name = self.get_readable_relation(link_key)
            targets = []
            
            items = link_value if isinstance(link_value, list) else [link_value]
            for item in items:
                title = item.get('title') or item.get('href', '').split('/')[-1]
                targets.append(urllib.parse.unquote(title))
            
            outgoing[rel_name] = targets

        result['Relationships (Outgoing)'] = outgoing

        # --- 6. Incoming Relationships (Related From) ---
        # We parse the graph edges to find nodes pointing TO our focus IRI
        incoming = {}
        
        if graph_data:
            nodes = {n['iri']: n.get('label', n['iri']) for n in graph_data.get('nodes', [])}
            edges = graph_data.get('edges', [])
            
            for edge in edges:
                source = edge['source']
                target = edge['target']
                relation_iri = edge['uri']
                
                # We only care if the target is our current term
                if target == focus_iri:
                    readable_rel = self.get_readable_relation(relation_iri)
                    source_label = nodes.get(source, source)
                    
                    if readable_rel not in incoming:
                        incoming[readable_rel] = []
                    
                    if source_label not in incoming[readable_rel]:
                        incoming[readable_rel].append(source_label)

        result['Related From (Incoming)'] = incoming

        return result

    def flatten_entry(self, entry):
        """
        Flattens the nested dictionary structure into a single-level dictionary 
        suitable for CSV writing.
        """
        flat = {}
        
        # 1. Core Fields
        flat['Id'] = entry.get('Id')
        flat['Label'] = entry.get('Label')
        flat['IRI'] = entry.get('IRI')
        
        # 2. Lists to Strings (Synonyms, Definitions)
        flat['Synonyms'] = " ; ".join(entry.get('Synonyms', []))
        flat['Definition'] = " | ".join(entry.get('Definition', []))
        
        # 3. Annotations (e.g., external_definition -> Annotation: external_definition)
        for k, v in entry.get('Annotations', {}).items():
            if isinstance(v, list):
                v = " ; ".join(str(x) for x in v)
            flat[f"Annotation: {k}"] = v
            
        # 4. Outgoing Relations (e.g. part of -> Outgoing: part of)
        for k, v in entry.get('Relationships (Outgoing)', {}).items():
            flat[f"Outgoing: {k}"] = " ; ".join(v)
            
        # 5. Incoming Relations (e.g. part of -> Incoming: part of)
        for k, v in entry.get('Related From (Incoming)', {}).items():
            flat[f"Incoming: {k}"] = " ; ".join(v)
            
        return flat

    def save_to_csv(self, data_list, filename="lookup_results.csv"):
        if not data_list:
            print("No data to save.")
            return

        # Flatten all entries
        flattened_data = [self.flatten_entry(entry) for entry in data_list]
        
        # Determine all unique headers (some terms might have relations others don't)
        headers = set()
        for item in flattened_data:
            headers.update(item.keys())
        
        # Sort headers for consistent output
        # Force ID and Label to be first
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

    def run(self, input_id):
        print(f"--- Looking up: {input_id} ---")
        
        iri = self.get_iri_from_id(input_id)
        if not iri: return None

        # Fetch Term Details and Graph Structure
        data = self.get_term_details(iri)
        graph = self.get_term_graph(iri)
        
        parsed_data = self.extract_data(data, graph, iri)
        return parsed_data



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
    tool = WhiteMatterLookup()
    
    # List of IDs to query
    # UBERON:0002305 -> Corpus Callosum
    ids_to_query = ["UBERON:0002305", "UBERON:0015510"] 
    
    results = []
    for term_id in uberon_ids:
        lookup_result = tool.run(term_id)
        if lookup_result:
            results.append(lookup_result)
            # Optional: Print JSON to console as well
            # print(json.dumps(lookup_result, indent=4))
    
    # Save to CSV
    tool.save_to_csv(results, "WMT_LUT_UBERON_results.csv")