# %%
from difflib import SequenceMatcher
from typing import Tuple, Dict, List
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score, recall_score, f1_score
import plotly.express as px
import plotly.graph_objects as go
import warnings

# -------------------------------
# CONFIG
# -------------------------------
SIM_THRESH = 0.95
USE_SEMANTIC_MATCHING = True  # set False to disable semantic mapping
DEDUP_PER_SAMPLE = False      # if True, deduplicate predictions per sample (one occurrence per label)
DATA_RAW = "data/raw/WM_full_600.csv"
DATA_PRED = "data/processed/whitematter_full_predicted_data_GPT_5_mini.csv"

# -------------------------------
# LOAD & MERGE
# -------------------------------
print("Loading datasets...")
golden_data = pd.read_csv(DATA_RAW)
predicted_data = pd.read_csv(DATA_PRED)



drop_cols = [
    "PMCID",'Open Source?', 'Authors', 'Citation', 'First Author', "Which imaging modality was used? e.g electroencephalogram (EEG), Positron emission tomography (PET), Anatomical MRI, fMRI, diffusion MRI (dMRI) etc",
    'Journal/Book', 'Publication Year', 'Create Date', 'PMCID.1',
    'NIHMS ID', 'DOI', 'Group Difference Explored?' ,  'Do they present results using x,y,z coordinates?', 'Other notes',
    'Unnamed: 27', 'Unnamed: 28'
]
for c in drop_cols:
    if c in golden_data.columns:
        golden_data = golden_data.drop(c, axis=1)

assert len(golden_data) == len(predicted_data), (
    f"Row count mismatch: golden={len(golden_data)}, predicted={len(predicted_data)}"
)

df = pd.concat(
    [golden_data.reset_index(drop=True),
     predicted_data.reset_index(drop=True)],
    axis=1,
    join='inner'
).copy()

print(f"Concatenated {len(df)} rows and {df.shape[1]} columns")



golden_data.columns


# Fixed rename (no stray string)
rename_map = {
    "Is this DTI?": "DTI_gt",
    "Is this a single study or a review?": "Study_type_gt",
    "Human study or not?": "Human_study_gt",
    "Does this study dementia, alzheimers, or related disease? \n": "Dementia_study_gt",
    "Which one?": "Disease_study_gt",
    "How are results presented? T-tests, beta effect sizes, ANOVA, Chi test, regression etc": "Results_method_gt",
    "What analysis software libraries are being used e.g dipy.org, fsl, freesurfer, PRIDE ..?": "Analysis_software_gt",
    "What was the diffusion measure e.g Fractional Anisotropy (FA), Mean Diffusivity (MD), Axial diffusivity (AD), Radial diffusivity (RD)?": "Diffusion_measure_gt",
    "Were increases or decreases in WM integrity reported relative to the disease population? Eg: AD> HC= RD increase (Corpus Callosum, Cingulum) , AD<HC = FA decreases (cingulum). If Column T is NA enter NA here. (What it means, comparing Alzheimer's disease with Health Control, there was increase in Radial diffusivity in the Corpus Callosum and Cingulum white matter tracts, etc).": "WM_integrity_gt",
    "Brain Template/ Space e.g MNI, JHU'": "Brain_template_gt",
    "What tracts were studied?": "Whitematter_tracts_gt",
    "DTI_study": "DTI_pred",
    "study_type": "Study_type_pred",
    "Human_study": "Human_study_pred",
    "Dementia_study": "Dementia_study_pred",
    "Disease_study": "Disease_study_pred",
    "whitematter_tracts": "Whitematter_tracts_pred",
    "imaging_modalities": "Imaging_modalities_pred",
    "results_method": "Results_method_pred",
    "analysis_software": "Analysis_software_pred",
    "diffusion_measures": "Diffusion_measure_pred",
    "template_space": "Brain_template_pred",
    "question_of_study": "Question_of_study "
}
df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
print(f"Merged rows {len(df)}")

# Keep only columns that exist for ordering and exporting
column_order = [
    "Title","PMCID",
    "DTI_gt", "DTI_pred",
    "Study_type_gt", "Study_type_pred",
    "Human_study_gt", "Human_study_pred",
    "Dementia_study_gt", "Dementia_study_pred",
    "Disease_study_gt", "Disease_study_pred",
    "Whitematter_tracts_gt", "Whitematter_tracts_pred",
    "Imaging_modalities_pred",
    "Results_method_gt", "Results_method_pred",
    "Analysis_software_gt", "Analysis_software_pred",
    "Diffusion_measure_gt", "Diffusion_measure_pred",
    "Brain_template_gt", "Brain_template_pred",
    "Question_of_study "
]
column_order = [c for c in column_order if c in df.columns]
df = df[column_order]
df.to_csv("data/processed/preds_ref_combined.csv", index=False, encoding='utf-8')

# -------------------------------
# TEXT NORMALIZATION & HELPERS
# -------------------------------
EMPTY_TOKENS = {
    "", "none", "n.a.", "na", "n a", "n/a", 'Na', 'N/A', "null", "_", "-", "nan",
    "not reported", "unknown"
}

def normalize_text(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return " ".join(str(x).lower().strip().split())

def normalize_cmap_keys_values(cmap: Dict[str, str]) -> Dict[str, str]:
    """Normalize both keys and values for a canonical map."""
    return {normalize_text(k): normalize_text(v) for k, v in cmap.items()}

def is_empty_token(s: str) -> bool:
    return normalize_text(s) in EMPTY_TOKENS

def clean_split(x) -> List[str]:
    """Split on commas/semicolons, normalize, remove empty tokens."""
    s = normalize_text(x)
    if is_empty_token(s):
        return []
    parts = [p.strip() for p in s.replace(";", ",").split(",")]
    out = [normalize_text(p) for p in parts if p and not is_empty_token(p)]
    return out

def canonicalize(value: str, cmap: Dict[str, str]) -> str:
    v = normalize_text(value)
    return cmap.get(v, v)

def canonicalize_list(values: List[str], cmap: Dict[str, str]) -> List[str]:
    return [canonicalize(v, cmap) for v in values if not is_empty_token(v)]

def seq_sim(a: str, b: str) -> float:
    a = normalize_text(a)
    b = normalize_text(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

def best_semantic_match(pred: str, refs: List[str], cmap: Dict[str, str], thresh: float = SIM_THRESH) -> Tuple[str, float]:
    """
    Return the canonical gold label (value) and best similarity score.
    If no match >= thresh, returns (None, best_score).
    """
    if not refs:
        return None, 0.0
    p_can = normalize_text(pred)
    best_ref = None
    best_score = 0.0
    for r in refs:
        r_can = canonicalize(r, cmap)
        # direct equality with canonicalized gold
        if p_can == r_can:
            return r_can, 1.0
        score = seq_sim(p_can, r_can)
        if score > best_score:
            best_score = score
            best_ref = r_can
    if best_score >= thresh:
        return best_ref, best_score
    return None, best_score

# -------------------------------
# CANONICAL MAPS (keys and values will be normalized)
# -------------------------------
canon_dti = {"yes": "yes", "no": "no"}
canon_human = {"yes": "yes", "no": "no", "human": "yes"}
canon_dementia = {"yes": "yes", "no": "no"}
canon_type = {"single study": "single study", "single": "single study", "meta analysis": "meta analysis", "review": "review"}
canon_disease = {
    "alzheimers disease": "alzheimers disease", "ad": "alzheimers disease",
    "parkinson disease": "parkinson disease"
}

# White-matter canonical map: extend as needed (keys and values will be normalized)
canon_white_matter_tracts = {
    "corpus callosum": "corpus callosum",
    "corpus callosum - splenium": "corpus callosum - splenium",
    "cingulum": "cingulum",
    "uncinate fasciculus": "uncinate fasciculus",
    "fornix": "fornix",
    "genu": "genu",
    "inferior fronto occipital fasciculus": "inferior fronto occipital fasciculus",
    "superior longitudinal fasciculus": "superior longitudinal fasciculus",
    "corticospinal tract": "corticospinal tract",
    "forceps minor": "forceps minor",
    # example abbreviations you likely want in the map
    "ilf": "inferior longitudinal fasciculus",
    "ifo": "inferior fronto occipital fasciculus",
    "uncinate fasc.": "uncinate fasciculus",
    "slf": "superior longitudinal fasciculus",
    "cc": "corpus callosum",
    "cc- corpus callosum": "corpus callosum"
}

# normalize keys & values
canon_dti = normalize_cmap_keys_values(canon_dti)
canon_human = normalize_cmap_keys_values(canon_human)
canon_dementia = normalize_cmap_keys_values(canon_dementia)
canon_type = normalize_cmap_keys_values(canon_type)
canon_disease = normalize_cmap_keys_values(canon_disease)
canon_white_matter_tracts = normalize_cmap_keys_values(canon_white_matter_tracts)

# -------------------------------
# FIELDS CONFIG
# -------------------------------
fields = [
    ("Does it use DTI?", "DTI_gt", "DTI_pred", canon_dti),
    ("Review or single study?", "Study_type_gt", "Study_type_pred", canon_type),
    ("Human_vs_non_human_study", "Human_study_gt", "Human_study_pred", canon_human),
    ("Does it study dementia or related diseases?", "Dementia_study_gt", "Dementia_study_pred", canon_dementia),
    ("Which diseases are studied", "Disease_study_gt", "Disease_study_pred", canon_disease),
    ("WM tracts studied", "Whitematter_tracts_gt", "Whitematter_tracts_pred", canon_white_matter_tracts),
]

# -------------------------------
# NORMALIZE DATA (returns dataframe)
# -------------------------------
def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    df_norm = df.copy()
    for name, gt_col, pred_col, cmap in fields:
        for col in [gt_col, pred_col]:
            if col in df_norm.columns:
                df_norm[col] = df_norm[col].apply(
                    lambda x: ", ".join(canonicalize_list(clean_split(x), cmap)) if pd.notna(x) else ""
                )
    df_norm.to_csv("data/processed/normalized_data.csv", index=False, encoding='utf-8')
    return df_norm

df = normalize_data(df)

# -------------------------------
# BUILD PRED AND REF LISTS (no dropping; mapping replaces preds when matched)
# -------------------------------
field_predictions = {name: {"references": [], "predictions": [], "preds_raw": []} for name, *_ in fields}

for _, row in df.iterrows():
    for name, gtf, prf, cmap in fields:
        if gtf not in df.columns or prf not in df.columns:
            continue

        # references: canonicalized (strings)
        g_list = canonicalize_list(clean_split(row.get(gtf)), cmap)

        # raw predictions: we use clean_split to split and normalize whitespace/punctuation but do not canonicalize values
        p_list_raw = [normalize_text(x) for x in clean_split(row.get(prf))]

        # mapping step: for each prediction replace with matched canonical gold label if sem-match >= thresh
        if USE_SEMANTIC_MATCHING and p_list_raw and g_list:
            p_mapped = []
            for p_raw in p_list_raw:
                match_label, score = best_semantic_match(p_raw, g_list, cmap, SIM_THRESH)
                if match_label is not None:
                    p_mapped.append(match_label)
                else:
                    p_mapped.append(normalize_text(p_raw))
        else:
            # no semantic mapping -> keep normalized predictions
            p_mapped = [normalize_text(p) for p in p_list_raw]

        # Keep all predictions (optionally deduplicate per sample)
        if DEDUP_PER_SAMPLE:
            p_list_to_store = sorted(set(p_mapped))
        else:
            p_list_to_store = p_mapped[:]  # keep order and duplicates if any

        g_list_to_store = g_list[:]  # keep canonical references as-is

        # Add even empty rows so lengths are aligned
        field_predictions[name]["references"].append(g_list_to_store)
        field_predictions[name]["predictions"].append(p_list_to_store)
        field_predictions[name]["preds_raw"].append(p_list_raw)

# -------------------------------
# MULTILABEL BINARIZATION
# -------------------------------
def multilabel_binarize(preds: List[List[str]], refs: List[List[str]]):
    all_labels = sorted(set(x for sub in preds + refs for x in sub))
    if not all_labels:
        return np.zeros((0, 0), int), np.zeros((0, 0), int), []
    mlb = MultiLabelBinarizer(classes=all_labels)
    Y_true = mlb.fit_transform(refs)
    Y_pred = mlb.transform(preds)
    return Y_pred, Y_true, all_labels

# -------------------------------
# METRICS
# -------------------------------
def macro_labelwise(y_true: np.ndarray, y_pred: np.ndarray, scorer) -> float:
    vals = []
    for j in range(y_true.shape[1]):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        if yt.sum() + yp.sum() == 0:
            continue
        vals.append(scorer(yt, yp, zero_division=0))
    return float(np.mean(vals)) if vals else 0.0

def _jaccard_samples_from_lists(preds, refs) -> float:
    vals = []
    for p, r in zip(preds, refs):
        ps, rs = set(p), set(r)
        union = ps | rs
        inter = ps & rs
        if not union:
            vals.append(1.0)
        else:
            vals.append(len(inter) / len(union))
    return float(np.mean(vals)) if vals else 0.0

### Calculate metrics for field with at least one gold label
#### commenting this means we will not skip empty rows
# def filter_empty_gold(preds, refs):
#     """
#     Remove samples where the gold reference list is empty.
#     """
#     preds_filt = []
#     refs_filt = []

#     for p, r in zip(preds, refs):
#         if len(r) == 0:
#             continue
#         preds_filt.append(p)
#         refs_filt.append(r)

#     return preds_filt, refs_filt
# #########

def compute_for_field(preds, refs):
    Y_pred, Y_true, labels = multilabel_binarize(preds, refs)
    if Y_true.size == 0:
        return dict(macro_p=0, macro_r=0, macro_f1=0,
                    micro_p=0, micro_r=0, micro_f1=0,
                    subset_acc=0, jaccard_accuracy=0, exact_match=0, labels=labels)

    macro_p = macro_labelwise(Y_true, Y_pred, precision_score)
    macro_r = macro_labelwise(Y_true, Y_pred, recall_score)
    macro_f1 = macro_labelwise(Y_true, Y_pred, f1_score)

    micro_p = precision_score(Y_true, Y_pred, average="micro", zero_division=0)
    micro_r = recall_score(Y_true, Y_pred, average="micro", zero_division=0)
    micro_f1 = f1_score(Y_true, Y_pred, average="micro", zero_division=0)

    subset_acc = float((Y_pred == Y_true).all(axis=1).mean())
    jaccard_accuracy = _jaccard_samples_from_lists(preds, refs)

    return dict(
        macro_p=macro_p, macro_r=macro_r, macro_f1=macro_f1,
        micro_p=micro_p, micro_r=micro_r, micro_f1=micro_f1,
        subset_acc=subset_acc, jaccard_accuracy=jaccard_accuracy,
        exact_match=subset_acc, labels=labels
    )

# -------------------------------
# RUN EVAL
# -------------------------------
results = {}
for name, *_ in fields:
    preds = field_predictions[name]["predictions"]
    refs  = field_predictions[name]["references"]
    results[name] = compute_for_field(preds, refs)

# handle skip empty rows
# for name, *_ in fields:
#     preds = field_predictions[name]["predictions"]
#     refs  = field_predictions[name]["references"]

#     preds_filt, refs_filt = filter_empty_gold(preds, refs)

#     if len(refs_filt) == 0:
#         results[name] = dict(
#             macro_p=np.nan, macro_r=np.nan, macro_f1=np.nan,
#             micro_p=np.nan, micro_r=np.nan, micro_f1=np.nan,
#             subset_acc=np.nan, jaccard_accuracy=np.nan,
#             exact_match=np.nan, labels=[]
#         )
#     else:
#         results[name] = compute_for_field(preds_filt, refs_filt)

# Pretty print a quick view
for name in results:
    r = results[name]
    print(f"\nField  {name}")
    print(f"macro P  {r['macro_p']:.3f}   macro R  {r['macro_r']:.3f}   macro F1  {r['macro_f1']:.3f}")
    print(f"micro P  {r['micro_p']:.3f}   micro R  {r['micro_r']:.3f}   micro F1  {r['micro_f1']:.3f}")
    print(f"subset acc  {r['subset_acc']:.3f}   jaccard accuracy  {r['jaccard_accuracy']:.3f}   exact match  {r['exact_match']:.3f}")

# -------------------------------
# Prepare metric DataFrames & CSV audit
# -------------------------------
metrics_df = pd.DataFrame.from_dict(results, orient="index").reset_index().rename(columns={"index": "field"})
COLS = ["precision", "recall", "f1", "accuracy", "exact_match", "jaccard_accuracy"]

macro_metrics_df = (
    metrics_df[["field", "macro_p", "macro_r", "macro_f1", "subset_acc", "exact_match", "jaccard_accuracy"]]
    .rename(columns={"macro_p": "precision", "macro_r": "recall", "macro_f1": "f1", "subset_acc": "accuracy"})
)[["field"] + COLS]

micro_metrics_df = (
    metrics_df[["field", "micro_p", "micro_r", "micro_f1", "subset_acc", "exact_match", "jaccard_accuracy"]]
    .rename(columns={"micro_p": "precision", "micro_r": "recall", "micro_f1": "f1", "subset_acc": "accuracy"})
)[["field"] + COLS]

samples_metrics_df = metrics_df.merge(
    micro_metrics_df[["field", "precision", "recall", "f1"]], on="field"
)[["field", "precision", "recall", "f1"]].merge(
    metrics_df[["field", "subset_acc", "exact_match", "jaccard_accuracy"]].rename(columns={"subset_acc": "accuracy"}),
    on="field"
)[ ["field"] + COLS ]

for df_view in [macro_metrics_df, micro_metrics_df, samples_metrics_df]:
    for c in COLS:
        df_view[c] = pd.to_numeric(df_view[c], errors="coerce").round(3)

macro_metrics_df.to_csv("data/processed/evaluation_metrics_macro.csv", index=False)
micro_metrics_df.to_csv("data/processed/evaluation_metrics_micro.csv", index=False)
samples_metrics_df.to_csv("data/processed/evaluation_metrics_samples.csv", index=False)

print("\nMACRO AVERAGING RESULTS")
print(macro_metrics_df.to_string(index=False))
print("\nMICRO AVERAGING RESULTS")
print(micro_metrics_df.to_string(index=False))
print("\nSAMPLES VIEW")
print(samples_metrics_df.to_string(index=False))

# Audit CSV with mapped vs raw preds
audit_rows = []
for name in field_predictions:
    for refs_row, preds_mapped_row, preds_raw_row in zip(field_predictions[name]["references"],
                                                        field_predictions[name]["predictions"],
                                                        field_predictions[name]["preds_raw"]):
        audit_rows.append({
            "field": name,
            "references": "; ".join(refs_row),
            "predictions_mapped": "; ".join(preds_mapped_row),
            "predictions_raw": "; ".join(preds_raw_row)
        })
audit_df = pd.DataFrame(audit_rows)
audit_df.to_csv("data/processed/prediction_mapping_audit.csv", index=False, encoding='utf-8')
print("Wrote prediction mapping audit to data/processed/prediction_mapping_audit.csv")

# -------------------------------
# PLOTS (unchanged visually)
# -------------------------------
print("\nGenerating plots\n")
fig1 = go.Figure()
fig1.add_trace(go.Bar(x=micro_metrics_df["field"], y=micro_metrics_df["f1"], name="F1"))
fig1.add_trace(go.Bar(x=micro_metrics_df["field"], y=micro_metrics_df["jaccard_accuracy"], name="Jaccard Accuracy"))
fig1.update_layout(barmode="group", template="plotly_white",
                   title="F1 vs Jaccard Accuracy",
                   yaxis=dict(range=[0,1], title="Score"), xaxis=dict(title="Field"))
fig1.show()

fig2 = px.scatter(micro_metrics_df, x="f1", y="accuracy", text="field", color="f1",
                  color_continuous_scale="Viridis", range_x=[0,1], range_y=[0,1],
                  title="Micro F1 vs Subset accuracy")
fig2.update_traces(textposition="top center", marker=dict(size=12))
fig2.update_layout(template="plotly_white")
fig2.show()

fig4 = px.imshow(micro_metrics_df.set_index("field")[["f1", "jaccard_accuracy"]],
                 text_auto=".2f", color_continuous_scale="YlGnBu", range_color=[0,1],
                 title="Heatmap")
fig4.update_layout(template="plotly_white", height=500)
fig4.show()

# =====================
# build_state / main (lightweight API)
# =====================
def build_state():
    """
    Return evaluation state (field_predictions, fields, compute_for_field)
    Useful for unit tests or importing this module.
    """
    return field_predictions, fields, compute_for_field

def main():
    """
    When run directly, the inline code above already executed. This function is a placeholder
    if you want to modularize execution into functions.
    """
    pass

if __name__ == "__main__":
    main()

# %%
