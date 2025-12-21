#%%
import os
import pronto
import pronto
from collections import OrderedDict
import json
# original_path = "data/raw/uberon.obo"      # original Uberon file
# fixed_path = "data/raw/uberon_fixed.obo"   

# removed_cob_lines = 0
#some lines in the original file mention COB:0000013, which causes pronto to error on load
# with open(original_path, "r", encoding="utf-8") as fin, open(fixed_path, "w", encoding="utf-8") as fout:
#     for line in fin:
#         # Drop any line that mentions COB:0000013
#         if "COB:0000013" in line:
#             removed_cob_lines += 1
#             continue
#         # Also normalise bad escapes in xref http lines
#         if line.startswith("xref: http") and ("\\" in line):
#             line = line.replace("\\:", ":").replace("\\,", ",")
#         fout.write(line)

# print("Removed lines mentioning COB:0000013:", removed_cob_lines)
# print("File exists:", os.path.exists(fixed_path))


#####
ONTOLOGY_PATH = "data/raw/uberon_fixed.obo"


with open(ONTOLOGY_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                print(line)
                

                



# print("Loading Uberon ontology with Pronto...")
onto = pronto.Ontology(ONTOLOGY_PATH)
# print(onto)
# # inspect ontology object
# print(type(onto))
# [att for att in dir(onto) if not att.startswith("_")]
# # inspect a single term in the ontology
# terms=onto.terms()
# term = next(iter(terms))
# print(term)
# print(type(term))
# [attr for attr in dir(term) if not attr.startswith("_")]
# # inspecting synonnys of a single term e.g cc
# cc=onto["UBERON:0015510"]
# syn = next(iter(cc.synonyms))
# print(type(syn))
# [attr for attr in dir(syn) if not attr.startswith("_")]


### some inspection using corpus callosum
UBERON_ID_CORPUS_CALLOSUM = "UBERON:0015510"
term = onto[UBERON_ID_CORPUS_CALLOSUM]
print("Basic info")
print("ID:", term.id)
print("Name:", term.name)
print("Definition:", term.definition)
print("synonyms:", term.synonyms)
print("xrefs:", term.xrefs)
print("subsets:", term.subsets)
print("subclass_of:", term.subclasses)
print("\nSynonyms from Uberon")
for syn in term.synonyms:
    print(f"{syn.description}  [{syn.scope}]")

#############################################################################
path = ONTOLOGY_PATH 

print("Loading Uberon ontology with Pronto...")
onto = pronto.Ontology(path)
print("Loaded terms:", len(onto))

PART_OF_RELS = {"BFO:0000050", "RO:0002524"}
OVERLAPS_RELS = {"RO:0002131"}


def dedup(seq):
    seen = set()
    out = []
    for s in seq:
        if not s:
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def get_synonym_buckets(t: pronto.Term):
    exact = []
    related = []
    other = []
    for syn in t.synonyms:
        if syn.scope == "EXACT":
            exact.append(syn.description)
        elif syn.scope == "RELATED":
            related.append(syn.description)
        else:
            other.append(syn.description)
    return {
        "exact": dedup(exact),
        "related": dedup(related),
        "other": dedup(other),
    }


def get_db_xrefs_pretty(t: pronto.Term):
    items = []
    for x in t.xrefs:
        xid = getattr(x, "id", None)
        desc = getattr(x, "description", None)
        if xid is None:
            xid = str(x)
        if desc:
            prefix = xid.split(":", 1)[0].lower()
            pretty = f"{desc}{prefix}"
            items.append(pretty)
        else:
            items.append(str(xid))
    return dedup(items)


def get_subclass_of_labels(t: pronto.Term):
    parents = []
    this_id = str(t.id)
    for p in t.superclasses(distance=1):
        # compare IDs instead of object identity
        if str(p.id) == this_id:
            continue
        parents.append(p.name)
    return dedup(parents)


def get_part_of_and_overlaps(t: pronto.Term):
    part_of_targets = []
    overlaps_targets = []
    for rel, targets in t.relationships.items():
        rid = getattr(rel, "id", None)
        if not rid:
            continue
        if rid in PART_OF_RELS:
            for target in targets:
                part_of_targets.append(target.name)
        if rid in OVERLAPS_RELS:
            for target in targets:
                overlaps_targets.append(target.name)
    return {
        "part_of": dedup(part_of_targets),
        "overlaps": dedup(overlaps_targets),
    }


def get_incoming_related_groups(
    t: pronto.Term,
    onto: pronto.Ontology,
    include_part_of: bool = True,
    include_overlaps: bool = True,
):
    """
    Group incoming relationships by relation label.

    Example structure:
      {
        "part of": [
            "tapetum of corpus callosum",
            "forceps minor of corpus callosum",
            ...
        ],
        "results in morphogenesis of": [
            "corpus callosum morphogenesis"
        ],
        "in anterior side of": [
            "genu of corpus callosum"
        ],
        "adjacent to": [
            "quadrigeminal cistern"
        ],
      }
    """
    groups = OrderedDict()
    this_id = str(t.id)

    for other in onto.terms():
        # optional skip by id; object identity is not reliable
        if str(other.id) == this_id:
            continue

        try:
            for rel, targets in other.relationships.items():
                rid = getattr(rel, "id", None)
                if not rid:
                    continue

                if not include_part_of and rid in PART_OF_RELS:
                    continue
                if not include_overlaps and rid in OVERLAPS_RELS:
                    continue

                for target in targets:
                    # crucial fix: compare IDs, not "is"
                    if str(target.id) != this_id:
                        continue

                    rel_label = getattr(rel, "name", rid)

                    if rel_label not in groups:
                        groups[rel_label] = []

                    groups[rel_label].append(other.name)
        except KeyError:
            continue

    for label, names in list(groups.items()):
        groups[label] = dedup(names)

    return groups


def format_related_groups(groups: OrderedDict):
    """
    Turn grouped relations into a multi line block like:

      part of
      tapetum of corpus callosum
      forceps minor of corpus callosum
      radiation of corpus callosum
      body of corpus callosum
      genu of corpus callosum
      rostrum of corpus callosum
      splenium of the corpus callosum

      results in morphogenesis of
      corpus callosum morphogenesis

      results in development of
      corpus callosum development

      in anterior side of
      genu of corpus callosum

      in posterior side of
      splenium of the corpus callosum

      adjacent to
      quadrigeminal cistern
    """
    if not groups:
        return ""

    blocks = []
    for label, names in groups.items():
        if not names:
            continue
        block_lines = [label] + names
        blocks.append("\n".join(block_lines))
    return "\n\n".join(blocks)


def build_row_from_uberon_term(t: pronto.Term, onto: pronto.Ontology):
    syns = get_synonym_buckets(t)
    db_xrefs_pretty = get_db_xrefs_pretty(t)
    parents = get_subclass_of_labels(t)
    rels = get_part_of_and_overlaps(t)

    groups = get_incoming_related_groups(
        t,
        onto,
        include_part_of=True,
        include_overlaps=True,
    )

    related_from_text = format_related_groups(groups)
    definition_text = t.definition or ""
    dbxref_for_row = "; ".join(db_xrefs_pretty)

    row = {
        "canonical_name": t.name,
        "ontology_id_uberon": t.id,
        "ontology_id_homba": "",
        "primary_abbreviation": "CC",
        "exact_synonyms": "; ".join(syns["exact"]),
        "related_synonyms": "; ".join(syns["related"] + syns["other"]),
        "abbreviations_all": "CC",
        "subcomponents": "genu; body; splenium; rostrum",
        "cross_references": dbxref_for_row,
        "noise_variants": "",
        "ambiguous_terms": "CC can also mean central canal in some contexts",
        "notes": "",
        "definition": definition_text,
        "database_cross_reference": dbxref_for_row,
        "subclass_of": "; ".join(parents),
        "part_of": "; ".join(rels["part_of"]),
        "overlaps": "; ".join(rels["overlaps"]),
    }

    if related_from_text:
        row["related_from"] = related_from_text

    row = {k: v for k, v in row.items() if v}
    return row


UBERON_ID_CORPUS_CALLOSUM = "UBERON:0002707"
term = onto[UBERON_ID_CORPUS_CALLOSUM]

print("Basic info")
print("ID:", term.id)
print("Name:", term.name)
print("Definition:", term.definition)

print("\nSynonyms from Uberon")
for syn in term.synonyms:
    print(f"{syn.description}  [{syn.scope}]")

print("\nBuilding lookup row")
row = build_row_from_uberon_term(term, onto)

for k, v in row.items():
    if k == "related_from":
        print("Related from")
        print(v)
    else:
        print(f"{k}: {v}")



###############

import requests

BASE_OLS = "https://www.ebi.ac.uk/ols4"
SNOMED = "snomed"


def get_term_details_by_id(sctid: str) -> dict:
    """Fetch a SNOMED term from OLS by SCTID."""
    iri = f"http://snomed.info/id/{sctid}"
    url = f"{BASE_OLS}/api/ontologies/{SNOMED}/terms"
    resp = requests.get(url, params={"iri": iri, "size": 1})
    resp.raise_for_status()
    data = resp.json()
    terms = data.get("_embedded", {}).get("terms", [])
    if not terms:
        raise ValueError(f"No SNOMED term found for SCTID {sctid}")
    return terms[0]


def extract_alt_labels(term: dict) -> list:
    """
    Extract alternative labels from the annotation block.
    OLS usually uses the SKOS altLabel IRI:
    http://www.w3.org/2004/02/skos/core#altLabel
    """
    annotation = term.get("annotation", {}) or {}
    alt_labels = []

    for key, values in annotation.items():
        if key.endswith("altLabel") or key.endswith("#altLabel"):
            if isinstance(values, list):
                alt_labels.extend(values)
            else:
                alt_labels.append(values)

    seen = set()
    result = []
    for v in alt_labels:
        if v not in seen:
            seen.add(v)
            result.append(v)
    return result


def get_parent_labels(term: dict) -> list:
    """Follow the parents link and return a list of parent labels."""
    parents_link = term.get("_links", {}).get("parents", {}).get("href")
    if not parents_link:
        return []

    resp = requests.get(parents_link)
    resp.raise_for_status()
    data = resp.json()
    parents = data.get("_embedded", {}).get("terms", [])
    return [p["label"] for p in parents]


if __name__ == "__main__":
    # Corpus callosum structure
    sctid = "88442005"
    details = get_term_details_by_id(sctid)
    alt_labels = extract_alt_labels(details)
    parent_labels = get_parent_labels(details)

    print(f"LABEL: {details['label']}")
    print(f"IRI: {details['iri']}")
    print(f"OBO ID: {details.get('obo_id')}")
    print()

    print("Information")
    print("preferred label")
    print(details["label"])

    if alt_labels:
        print()
        print("alternative label")
        for a in alt_labels:
            print(a)

    if parent_labels:
        print()
        print("class Relations")
        print("Subclass of")
        for p in parent_labels:
            print(p)

    print()
    print("============================================================")

#########
import requests

BASE_OLS = "https://www.ebi.ac.uk/ols4"
SNOMED = "snomed"


def get_term_details_by_id(sctid: str) -> dict:
    """Fetch a SNOMED term from OLS by SCTID."""
    iri = f"http://snomed.info/id/{sctid}"
    url = f"{BASE_OLS}/api/ontologies/{SNOMED}/terms"
    resp = requests.get(url, params={"iri": iri, "size": 1})
    resp.raise_for_status()
    data = resp.json()
    terms = data.get("_embedded", {}).get("terms", [])
    if not terms:
        raise ValueError(f"No SNOMED term found for SCTID {sctid}")
    return terms[0]


def extract_alt_labels(term: dict) -> list:
    """
    Extract alternative labels from the annotation block and, if needed,
    from the synonyms field. This is tuned for how OLS exposes SNOMED.
    """
    annotation = term.get("annotation", {}) or {}
    alt_labels = []

    # Look in annotation for keys that look like "altLabel" or "alternative_term"
    for key, values in annotation.items():
        key_l = key.lower()
        if "altlabel" in key_l or "alternative" in key_l:
            if isinstance(values, list):
                alt_labels.extend(values)
            else:
                alt_labels.append(values)

    # Also fall back to synonyms if present
    syns = term.get("synonyms") or []
    for s in syns:
        alt_labels.append(s)

    # Clean up duplicates
    seen = set()
    result = []
    for v in alt_labels:
        if not v:
            continue
        if v not in seen:
            seen.add(v)
            result.append(v)

    return result


def get_parent_labels(term: dict) -> list:
    """Follow the parents link and return a list of parent labels."""
    parents_link = term.get("_links", {}).get("parents", {}).get("href")
    if not parents_link:
        return []

    resp = requests.get(parents_link)
    resp.raise_for_status()
    data = resp.json()
    parents = data.get("_embedded", {}).get("terms", [])
    return [p["label"] for p in parents]


if __name__ == "__main__":

    # Corpus callosum structure
    sctid = "85637007"
    details = get_term_details_by_id(sctid)
    alt_labels = extract_alt_labels(details)
    parent_labels = get_parent_labels(details)

    print(f"LABEL: {details['label']}")
    print(f"IRI: {details['iri']}")
    print(f"OBO ID: {details.get('obo_id')}")
    print()

    print("Information")
    print("preferred label")
    print(details["label"])

    if alt_labels:
        print()
        print("alternative label")
        for a in alt_labels:
            print(a)

    if parent_labels:
        print()
        print("class Relations")
        print("Subclass of")
        for p in parent_labels:
            print(p)

    print()
    print("============================================================")

####################

