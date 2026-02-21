# %%
"""
WMT Extraction Error Analysis — Hallucination Levels

For each of the 4 model/input conditions computes at the corpus level:
  - Correct (TP)      : predicted tract that exists in gold
  - False Positive    : predicted tract NOT in gold   (hallucination)
  - False Negative    : gold tract NOT in predictions (missed detection)

Plots a 100 % stacked bar chart matching the example figure.
"""

from difflib import SequenceMatcher
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── FILE PATHS ────────────────────────────────────────────────────────────────
DATA_RAW      = "data/raw/WM_full_600.csv"
GPT4_ABSTRACT = "data/processed/whitematter_abstract_predicted_data_GPT_4_mini.csv"
GPT5_ABSTRACT = "data/processed/whitematter_abstract_predicted_data_GPT_5_mini.csv"
GPT4_FULL     = "data/processed/whitematter_full_predicted_data_GPT_4o_mini_data.csv"
GPT5_FULL     = "data/processed/whitematter_full_predicted_data_GPT_5_mini.csv"

# ── NORMALISATION ─────────────────────────────────────────────────────────────
EMPTY_TOKENS = {
    "", "none", "n.a.", "na", "n a", "n/a", "null", "_", "-", "nan",
    "not reported", "unknown",
}
SIM_THRESH = 0.85   # slightly relaxed to handle minor spelling variants

def normalize_text(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return " ".join(str(x).lower().strip().split())

def is_empty(s: str) -> bool:
    return normalize_text(s) in EMPTY_TOKENS

def clean_split(x) -> List[str]:
    s = normalize_text(x)
    if is_empty(s):
        return []
    parts = [p.strip() for p in s.replace(";", ",").split(",")]
    return [normalize_text(p) for p in parts if p and not is_empty(p)]

def normalize_cmap(cmap: Dict[str, str]) -> Dict[str, str]:
    return {normalize_text(k): normalize_text(v) for k, v in cmap.items()}

def canonicalize(value: str, cmap: Dict[str, str]) -> str:
    return cmap.get(normalize_text(value), normalize_text(value))

def canonicalize_list(values: List[str], cmap: Dict[str, str]) -> List[str]:
    return [canonicalize(v, cmap) for v in values if not is_empty(v)]

def seq_sim(a: str, b: str) -> float:
    a, b = normalize_text(a), normalize_text(b)
    if not a and not b: return 1.0
    if not a or not b:  return 0.0
    return SequenceMatcher(None, a, b).ratio()

# ── WMT CANONICAL MAP ─────────────────────────────────────────────────────────
canon_wmt = normalize_cmap({
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
    "ilf": "inferior longitudinal fasciculus",
    "ifo": "inferior fronto occipital fasciculus",
    "uncinate fasc.": "uncinate fasciculus",
    "slf": "superior longitudinal fasciculus",
    "cc": "corpus callosum",
    "cc- corpus callosum": "corpus callosum",
})

# ── TP / FP / FN PER ROW ─────────────────────────────────────────────────────
def row_tp_fp_fn(gold: List[str], pred: List[str]) -> Tuple[int, int, int]:
    """
    Match predictions to gold using fuzzy similarity.
    Returns (true_positives, false_positives, false_negatives).
    """
    if not gold and not pred:
        return 0, 0, 0

    gold_set  = set(gold)
    pred_copy = list(pred)
    matched_gold = set()

    tp, fp = 0, 0
    for p in pred_copy:
        best_score = 0.0
        best_g     = None
        for g in gold_set - matched_gold:
            s = seq_sim(p, g)
            if s > best_score:
                best_score, best_g = s, g
        if best_score >= SIM_THRESH and best_g is not None:
            tp += 1
            matched_gold.add(best_g)
        else:
            fp += 1

    fn = len(gold_set - matched_gold)
    return tp, fp, fn


# ── EVALUATE ONE CONDITION ────────────────────────────────────────────────────
def evaluate_errors(pred_path: str, label: str, golden_data: pd.DataFrame) -> dict:
    print(f"\nEvaluating: {label}")
    predicted_data = pd.read_csv(pred_path)

    assert len(golden_data) == len(predicted_data), (
        f"Row mismatch: gold={len(golden_data)}, pred={len(predicted_data)}"
    )

    df = pd.concat(
        [golden_data.reset_index(drop=True),
         predicted_data.reset_index(drop=True)],
        axis=1, join="inner"
    ).copy()

    df = df.rename(columns={
        "What tracts were studied?": "wmt_gt",
        "whitematter_tracts":        "wmt_pred",
    })

    total_tp = total_fp = total_fn = 0

    for _, row in df.iterrows():
        gold = canonicalize_list(clean_split(row.get("wmt_gt",   "")), canon_wmt)
        pred = canonicalize_list(clean_split(row.get("wmt_pred", "")), canon_wmt)

        # Skip rows where both are empty (no signal from either side)
        if not gold and not pred:
            continue

        tp, fp, fn = row_tp_fp_fn(gold, pred)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    total = total_tp + total_fp + total_fn
    if total == 0:
        tp_pct = fp_pct = fn_pct = 0.0
    else:
        tp_pct = total_tp / total * 100
        fp_pct = total_fp / total * 100
        fn_pct = total_fn / total * 100

    print(f"  TP={total_tp}  FP={total_fp}  FN={total_fn}")
    print(f"  Correct={tp_pct:.1f}%  FP={fp_pct:.1f}%  FN={fn_pct:.1f}%")
    return {"condition": label, "Correct": tp_pct, "False Positive": fp_pct, "False Negative": fn_pct}


# ── LOAD GROUND TRUTH ─────────────────────────────────────────────────────────
golden_data = pd.read_csv(DATA_RAW)
drop_cols = [
    "PMCID", "Open Source?", "Authors", "Citation", "First Author",
    "Which imaging modality was used? e.g electroencephalogram (EEG), Positron emission tomography (PET), Anatomical MRI, fMRI, diffusion MRI (dMRI) etc",
    "Journal/Book", "Publication Year", "Create Date", "PMCID.1",
    "NIHMS ID", "DOI", "Group Difference Explored?",
    "Do they present results using x,y,z coordinates?", "Other notes",
    "Unnamed: 27", "Unnamed: 28",
]
for c in drop_cols:
    if c in golden_data.columns:
        golden_data = golden_data.drop(c, axis=1)

# ── RUN ALL FOUR CONDITIONS ───────────────────────────────────────────────────
conditions = [
    (GPT4_ABSTRACT, "GPT-4 Abs"),
    (GPT4_FULL,     "GPT-4 Full"),
    (GPT5_ABSTRACT, "GPT-5 Abs"),
    (GPT5_FULL,     "GPT-5 Full"),
]

rows = [evaluate_errors(path, label, golden_data) for path, label in conditions]
err_df = pd.DataFrame(rows).set_index("condition")
err_df.to_csv("data/processed/wmt_error_analysis.csv")

print("\n" + "=" * 55)
print("WMT ERROR SUMMARY")
print("=" * 55)
print(err_df.to_string())

# ── STACKED BAR PLOT (100 %) ─────────────────────────────────────────────────
COLORS = {
    "Correct":        "#2ecc71",   # green
    "False Positive": "#e74c3c",   # red
    "False Negative": "#3498db",   # blue
}
outcome_order = ["Correct", "False Positive", "False Negative"]
labels = err_df.index.tolist()
x      = np.arange(len(labels)) * 0.04  # tighter spacing between bars
bar_w  = 0.025

fig, ax = plt.subplots(figsize=(9, 6))

bottoms = np.zeros(len(labels))
for outcome in outcome_order:
    vals = err_df[outcome].values
    ax.bar(x, vals, bar_w, bottom=bottoms,
           color=COLORS[outcome], label=outcome, edgecolor="white", linewidth=0.4)
    bottoms += vals

ax.set_ylabel("Percentage of Cases (%)", fontsize=12, fontweight="bold")
ax.set_xlabel("Condition",               fontsize=12, fontweight="bold")
ax.set_title("WMT Extraction Error Analysis",
             fontsize=13, fontweight="bold", pad=12)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11, fontweight="bold", rotation=20, ha="right")
ax.set_xlim(x[0] - bar_w, x[-1] + bar_w)
ax.set_ylim(0, 105)
for lbl in ax.get_yticklabels():
    lbl.set_fontweight("bold")
ax.tick_params(axis="y", labelsize=10)

legend_patches = [mpatches.Patch(color=COLORS[o], label=o) for o in outcome_order]
ax.legend(handles=legend_patches, title="", title_fontsize=10,
          fontsize=10, loc="upper left", bbox_to_anchor=(0.95, 1),
          frameon=True, prop={"weight": "bold"})

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
out_path = "data/processed/wmt_error_analysis_gpt4_vs_gpt5.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"\nPlot saved → {out_path}")
plt.show()