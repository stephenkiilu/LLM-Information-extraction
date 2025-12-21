#%%
import requests
import json
import urllib.parse
import csv
import sys

class SnomedOLSLookup:
    def __init__(self):
        # Using the standard OLS4 API
        self.base_url = "https://www.ebi.ac.uk/ols4/api"
        # Default to 'snomed', but we will validate this
        self.ontology = "snomed"

    def check_ontology_status(self):
        """
        Checks if the 'snomed' ontology exists on this OLS instance.
        If not, it searches for any ontology containing 'snomed' in the title.
        """
        url = f"{self.base_url}/ontologies/{self.ontology}"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print(f"✅ Ontology '{self.ontology}' found on server.")
                return True
            else:
                print(f"⚠️ Ontology '{self.ontology}' NOT found (Status: {response.status_code}).")
                print("   Searching for available SNOMED-related ontologies...")
                
                # Search for any ontology with 'snomed' in the name
                search_url = f"{self.base_url}/ontologies"
                params = {"page": 0, "size": 20}
                r = requests.get(search_url, params=params)
                found = False
                if r.status_code == 200:
                    for onto in r.json().get('_embedded', {}).get('ontologies', []):
                        if 'snomed' in onto['ontologyId'].lower() or 'snomed' in onto.get('config', {}).get('title', '').lower():
                            print(f"   -> Found alternative: {onto['ontologyId']} ({onto.get('config', {}).get('title')})")
                            self.ontology = onto['ontologyId']
                            found = True
                            break
                
                if found:
                    print(f"   -> Switching to use '{self.ontology}'")
                    return True
                else:
                    print("❌ No SNOMED ontologies found on this public OLS instance.")
                    print("   (Note: Public EBI OLS often restricts SNOMED due to licensing.)")
                    return False
        except Exception as e:
            print(f"Error checking ontology: {e}")
            return False

    def get_iri_from_id(self, term_id):
        """
        Resolves SNOMED ID to IRI.
        """
        # 1. Try exact search first
        search_url = f"{self.base_url}/search"
        params = {
            "q": term_id,
            "ontology": self.ontology,
            "exact": "true",
            "queryFields": "obo_id,short_form,label,iri"
        }
        
        try:
            response = requests.get(search_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data['response']['numFound'] > 0:
                return data['response']['docs'][0]['iri']
            
            # 2. If exact fails, try standard SNOMED IRI construction
            # This is the standard format for SNOMED CT
            print(f"   Search failed for {term_id}, attempting direct IRI construction...")
            return f"http://snomed.info/id/{term_id}"

        except requests.exceptions.RequestException as e:
            print(f"   Error searching ID: {e}")
            return None

    def get_term_details(self, iri):
        encoded_iri = urllib.parse.quote(urllib.parse.quote(iri, safe=''), safe='')
        url = f"{self.base_url}/ontologies/{self.ontology}/terms/{encoded_iri}"
        
        try:
            response = requests.get(url)
            if response.status_code in [403, 404]:
                print(f"   ❌ API Error {response.status_code}: Could not access term at {url}")
                return None
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"   Error fetching details: {e}")
            return None

    def get_term_graph(self, iri):
        encoded_iri = urllib.parse.quote(urllib.parse.quote(iri, safe=''), safe='')
        url = f"{self.base_url}/ontologies/{self.ontology}/terms/{encoded_iri}/graph"
        try:
            response = requests.get(url)
            if response.status_code in [403, 404]:
                return None
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException:
            return None

    def extract_data(self, term_data, graph_data, focus_iri):
        if not term_data:
            return None

        result = {}
        result['Label'] = term_data.get('label')
        result['Id'] = term_data.get('short_form')
        result['IRI'] = term_data.get('iri')
        result['Definition'] = term_data.get('description', ["N/A"])
        result['Synonyms'] = term_data.get('synonyms', [])
        result['Annotations'] = term_data.get('annotation', {})

        # Outgoing
        links = term_data.get('_links', {})
        outgoing = {}
        ignored_links = {'self', 'first', 'last', 'prev', 'next', 'collection', 'jstree', 'graph', 'curies', 'ontology', 'parents', 'children', 'descendants', 'ancestors'}

        for link_key, link_value in links.items():
            if link_key in ignored_links: continue
            
            rel_name = link_key.split('/')[-1]
            if rel_name == "hierarchicalParents": rel_name = "Subclass Of"

            targets = []
            items = link_value if isinstance(link_value, list) else [link_value]
            for item in items:
                title = item.get('title') or item.get('href', '').split('/')[-1]
                targets.append(urllib.parse.unquote(title))
            outgoing[rel_name] = targets

        result['Relationships (Outgoing)'] = outgoing

        # Incoming
        incoming = {}
        if graph_data:
            nodes = {n['iri']: n.get('label', n['iri']) for n in graph_data.get('nodes', [])}
            edges = graph_data.get('edges', [])
            
            for edge in edges:
                source = edge['source']
                target = edge['target']
                uri = edge['uri']
                
                label = edge.get('label')
                if not label: label = uri.split('/')[-1]

                if target == focus_iri:
                    source_label = nodes.get(source, source)
                    if label not in incoming: incoming[label] = []
                    if source_label not in incoming[label]: incoming[label].append(source_label)

        result['Related From (Incoming)'] = incoming
        return result

    def flatten_entry(self, entry):
        flat = {}
        flat['Id'] = entry.get('Id')
        flat['Label'] = entry.get('Label')
        flat['Synonyms'] = " ; ".join(entry.get('Synonyms', []))
        
        for k, v in entry.get('Annotations', {}).items():
            if isinstance(v, list): v = " ; ".join(str(x) for x in v)
            flat[f"Annotation: {k}"] = v

        for k, v in entry.get('Relationships (Outgoing)', {}).items():
            flat[f"Outgoing: {k}"] = " ; ".join(v)
            
        for k, v in entry.get('Related From (Incoming)', {}).items():
            flat[f"Incoming: {k}"] = " ; ".join(v)
        return flat

    def save_to_csv(self, data_list, filename="WMT_LUT_SNOMED.csv"):
        if not data_list: return
        flattened = [self.flatten_entry(e) for e in data_list]
        headers = sorted(list(set().union(*(d.keys() for d in flattened))))
        
        prio = ['Id', 'Label', 'Synonyms']
        for p in reversed(prio):
            if p in headers: headers.insert(0, headers.pop(headers.index(p)))
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(flattened)
        print(f"Saved to {filename}")

    def run(self, query):
        print(f"--- Looking up: {query} ---")
        iri = self.get_iri_from_id(query)
        if not iri: return None

        data = self.get_term_details(iri)
        if not data: return None

        graph = self.get_term_graph(iri)
        return self.extract_data(data, graph, iri)

snomed_ids = ['62872008', '89202009', '9000002', '36159002', '37035000', '88442005', '16746009', '1296738007', '279300007', '84013004', '42932006', '80049006', '87463005', '35664009', '55233005', '85637007', '28390009', '3960005', '70105001', '13958008', '89202009', '60105000', '26230003', '80434005']


# --- Main Execution ---
if __name__ == "__main__":
    tool = SnomedOLSLookup()
    
    # 1. Verify Ontology Exists first
    if tool.check_ontology_status():
        queries = snomed_ids    
        results = []
        for q in queries:
            res = tool.run(q)
            if res:
                results.append(res)
        
        tool.save_to_csv(results)
    else:
        print("\nScript Aborted: SNOMED ontology is not accessible on this OLS instance.") 