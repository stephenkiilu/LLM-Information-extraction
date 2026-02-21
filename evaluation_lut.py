# %%
"""
Evaluation pipeline — WM Tracts (WMT) only.

Compares GPT-4 and GPT-5 with and without a Look-Up Table (LUT)
on the White Matter Tracts multilabel field and produces a
publication-ready grouped bar chart (mirrors the accuracy screenshot).
"""

from difflib import SequenceMatcher
from typing import Tuple, Dict, List
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# ── CONFIG ─────────────────────────────────────────────────────────────────────
SIM_THRESH            = 0.95
USE_SEMANTIC_MATCHING = True
DEDUP_PER_SAMPLE      = False
SKIP_EMPTY_GOLD       = True

DATA_RAW    = "data/raw/WM_full_600.csv"
GPT4_no_LUT = "data/processed/whitematter_no_lut_predicted_data_GPT_4o_mini.csv"
GPT5_no_LUT = "data/processed/whitematter_no_lut_predicted_data_GPT_5_mini.csv"
GPT4_LUT    = "data/processed/whitematter_full_predicted_data_GPT_4o_mini_data.csv"
GPT5_LUT    = "data/processed/whitematter_full_predicted_data_GPT_5_mini.csv"

# ── TEXT NORMALIZATION ─────────────────────────────────────────────────────────
EMPTY_TOKENS = {
    "", "none", "n.a.", "na", "n a", "n/a", "null", "_", "-", "nan",
    "not reported", "unknown",
}

def normalize_text(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return " ".join(str(x).lower().strip().split())

def normalize_cmap(cmap: Dict[str, str]) -> Dict[str, str]:
    return {normalize_text(k): normalize_text(v) for k, v in cmap.items()}

def is_empty(s: str) -> bool:
    return normalize_text(s) in EMPTY_TOKENS

def clean_split(x) -> List[str]:
    s = normalize_text(x)
    if is_empty(s):
        return []
    parts = [p.strip() for p in s.replace(";", ",").split(",")]
    return [normalize_text(p) for p in parts if p and not is_empty(p)]

def canonicalize(value: str, cmap: Dict[str, str]) -> str:
    return cmap.get(normalize_text(value), normalize_text(value))

def canonicalize_list(values: List[str], cmap: Dict[str, str]) -> List[str]:
    return [canonicalize(v, cmap) for v in values if not is_empty(v)]

def seq_sim(a: str, b: str) -> float:
    a, b = normalize_text(a), normalize_text(b)
    if not a and not b: return 1.0
    if not a or not b:  return 0.0
    return SequenceMatcher(None, a, b).ratio()

def best_semantic_match(pred: str, refs: List[str], cmap: Dict[str, str],
                        thresh: float = SIM_THRESH) -> Tuple[str, float]:
    if not refs:
        return None, 0.0
    p_can = normalize_text(pred)
    best_ref, best_score = None, 0.0
    for r in refs:
        r_can = canonicalize(r, cmap)
        if p_can == r_can:
            return r_can, 1.0
        score = seq_sim(p_can, r_can)
        if score > best_score:
            best_score, best_ref = score, r_can
    return (best_ref, best_score) if best_score >= thresh else (None, best_score)

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

# ── METRIC HELPERS ────────────────────────────────────────────────────────────
def _jaccard_samples(preds, refs) -> float:
    vals = []
    for p, r in zip(preds, refs):
        ps, rs = set(p), set(r)
        union = ps | rs
        vals.append(len(ps & rs) / len(union) if union else 1.0)
    return float(np.mean(vals)) if vals else 0.0

def compute_wmt_f1(preds: List[List[str]], refs: List[List[str]]) -> float:
    """Return micro-F1 for the WMT multilabel field."""
    if SKIP_EMPTY_GOLD:
        preds, refs = zip(*[(p, r) for p, r in zip(preds, refs) if r]) if any(refs) else ([], [])
        preds, refs = list(preds), list(refs)

    if not refs:
        return 0.0

    all_labels = sorted(set(x for sub in list(preds) + list(refs) for x in sub))
    if not all_labels:
        return 0.0

    mlb    = MultiLabelBinarizer(classes=all_labels)
    Y_true = mlb.fit_transform(refs)
    Y_pred = mlb.transform(preds)

    return float(f1_score(Y_true, Y_pred, average="micro", zero_division=0))

# ── EVALUATE ONE MODEL (WMT only) ─────────────────────────────────────────────
def evaluate_model(pred_path: str, model_label: str) -> float:
    """
    Load predictions, match against ground truth on the WM Tracts field,
    and return the micro-F1 score.
    """
    print(f"\n{'=' * 60}")
    print(f"Evaluating: {model_label}")
    print(f"  File: {pred_path}")
    print("=" * 60)

    golden_data    = pd.read_csv(DATA_RAW)
    predicted_data = pd.read_csv(pred_path)

    # Drop metadata columns from ground truth
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

    assert len(golden_data) == len(predicted_data), (
        f"Row count mismatch: golden={len(golden_data)}, predicted={len(predicted_data)}"
    )

    df = pd.concat(
        [golden_data.reset_index(drop=True),
         predicted_data.reset_index(drop=True)],
        axis=1, join="inner"
    ).copy()

    # Rename ground-truth WMT column
    df = df.rename(columns={
        "What tracts were studied?": "Whitematter_tracts_gt",
        "whitematter_tracts":        "Whitematter_tracts_pred",
    })

    if "Whitematter_tracts_gt" not in df.columns or "Whitematter_tracts_pred" not in df.columns:
        print("  WARNING: WMT columns not found — returning F1=0")
        return 0.0

    # Build prediction/reference lists per row
    references, predictions = [], []
    for _, row in df.iterrows():
        g_list = canonicalize_list(clean_split(row.get("Whitematter_tracts_gt")),  canon_wmt)
        p_raw  = [normalize_text(x) for x in clean_split(row.get("Whitematter_tracts_pred"))]

        if USE_SEMANTIC_MATCHING and p_raw and g_list:
            p_mapped = []
            for p in p_raw:
                m, _ = best_semantic_match(p, g_list, canon_wmt, SIM_THRESH)
                p_mapped.append(m if m is not None else p)
        else:
            p_mapped = p_raw

        if DEDUP_PER_SAMPLE:
            p_mapped = sorted(set(p_mapped))

        references.append(g_list)
        predictions.append(p_mapped)

    micro_f1 = compute_wmt_f1(predictions, references)
    print(f"  WMT micro-F1 = {micro_f1:.3f}")
    return micro_f1

# ── RUN ALL FOUR CONDITIONS ───────────────────────────────────────────────────
f1_gpt4_no_lut = evaluate_model(GPT4_no_LUT, "GPT-4  No LUT")
f1_gpt4_lut    = evaluate_model(GPT4_LUT,    "GPT-4  With LUT")
f1_gpt5_no_lut = evaluate_model(GPT5_no_LUT, "GPT-5  No LUT")
f1_gpt5_lut    = evaluate_model(GPT5_LUT,    "GPT-5  With LUT")

print("\n" + "=" * 60)
print("WMT F1 SUMMARY")
print("=" * 60)
print(f"  GPT-4  No LUT  : {f1_gpt4_no_lut:.3f}")
print(f"  GPT-4  With LUT: {f1_gpt4_lut:.3f}")
print(f"  GPT-5  No LUT  : {f1_gpt5_no_lut:.3f}")
print(f"  GPT-5  With LUT: {f1_gpt5_lut:.3f}")

# Save CSV
summary_df = pd.DataFrame([
    {"model": "GPT-4", "condition": "No LUT",   "wmt_f1": f1_gpt4_no_lut},
    {"model": "GPT-4", "condition": "With LUT",  "wmt_f1": f1_gpt4_lut},
    {"model": "GPT-5", "condition": "No LUT",   "wmt_f1": f1_gpt5_no_lut},
    {"model": "GPT-5", "condition": "With LUT",  "wmt_f1": f1_gpt5_lut},
])
summary_df.to_csv("data/processed/f1_wmt_lut_comparison.csv", index=False)

# ── BAR PLOT ── mirrors screenshot (2 model groups × 2 condition bars) ─────────
COLORS = {"No LUT": "#EF553B", "With LUT": "#636EFA"}
models = ["GPT-4", "GPT-5"]
x      = np.arange(len(models))
bar_w  = 0.3

no_lut_vals   = [f1_gpt4_no_lut,  f1_gpt5_no_lut]
with_lut_vals = [f1_gpt4_lut,     f1_gpt5_lut]

fig, ax = plt.subplots(figsize=(7, 6))

bars_no  = ax.bar(x - bar_w / 2, no_lut_vals,   bar_w,
                  color=COLORS["No LUT"],   label="No LUT",   edgecolor="white")
bars_yes = ax.bar(x + bar_w / 2, with_lut_vals, bar_w,
                  color=COLORS["With LUT"], label="With LUT", edgecolor="white")

for bars in (bars_no, bars_yes):
    for bar in bars:
        v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012,
                f"{v:.2f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color="#333333")

ax.set_ylabel("F1 Score", fontsize=13, fontweight="bold")
ax.set_xlabel("Model",    fontsize=13, fontweight="bold")
ax.set_title("Look-Up Table (LUT) on WMT Extraction",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12, fontweight="bold")
ax.set_ylim(0, 1.0)
for lbl in ax.get_yticklabels():
    lbl.set_fontweight("bold")
ax.tick_params(axis="y", labelsize=10)
ax.legend(title="", fontsize=11, title_fontsize=11,
          frameon=True, prop={"weight": "bold"})
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.35)

plt.tight_layout()
out_path = "data/processed/f1_wmt_lut_comparison_gpt4_vs_gpt5.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"\nPlot saved → {out_path}")
plt.show()