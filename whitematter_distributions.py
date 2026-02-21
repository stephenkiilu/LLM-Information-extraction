# %%
"""
WMT Distribution Analysis

For each of the 4 model/input conditions, categorise every row's
whitematter_tracts prediction as:
  - NA       : empty / null / "na" / NaN
  - Global   : contains the word "global" (but no specific tract)
  - Specific : a named tract is mentioned

Produces a grouped bar chart (% of studies) matching the example figure.
"""

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

EMPTY_TOKENS = {
    "", "none", "n.a.", "na", "n a", "n/a", "null", "_", "-", "nan",
    "not reported", "unknown",
}

# ── CATEGORISATION ────────────────────────────────────────────────────────────
def categorise(cell) -> str:
    """
    Given a raw cell value from the whitematter_tracts column, return
    one of: 'NA', 'Global', 'Specific'.
    """
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return "NA"

    raw = str(cell).strip()
    if not raw or raw.lower() in EMPTY_TOKENS:
        return "NA"

    # Split on comma or semicolon
    parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]

    # Check if *all* non-empty parts resolve to global/empty
    global_keywords = {
        "global", "white matter", "global white matter",
        "all tracts", "all white matter", "wm",
    }

    has_specific = False
    has_global   = False

    for p in parts:
        pl = p.lower()
        if pl in EMPTY_TOKENS:
            continue
        if pl in global_keywords or pl.startswith("global"):
            has_global = True
        else:
            has_specific = True   # anything else = specific tract name

    if has_specific:
        return "Specific"
    if has_global:
        return "Global"
    return "NA"


def compute_distribution(pred_path: str, label: str) -> dict:
    """Load a prediction CSV, categorise each row, return % breakdown."""
    print(f"\nProcessing: {label}  ({pred_path})")
    df = pd.read_csv(pred_path)

    col = "whitematter_tracts"
    if col not in df.columns:
        print(f"  WARNING: '{col}' not found — all NA")
        return {"model": label, "Specific": 0.0, "Global": 0.0, "NA": 100.0}

    cats = df[col].apply(categorise)
    n    = len(cats)
    counts = cats.value_counts()

    specific = counts.get("Specific", 0) / n * 100
    global_  = counts.get("Global",   0) / n * 100
    na       = counts.get("NA",       0) / n * 100

    print(f"  n={n}  Specific={specific:.1f}%  Global={global_:.1f}%  NA={na:.1f}%")
    return {"model": label, "Specific": specific, "Global": global_, "NA": na}


# ── RUN ALL FOUR CONDITIONS ───────────────────────────────────────────────────
conditions = [
    (GPT4_ABSTRACT, "GPT-4 Abstract"),
    (GPT4_FULL,     "GPT-4 Full"),
    (GPT5_ABSTRACT, "GPT-5 Abstract"),
    (GPT5_FULL,     "GPT-5 Full"),
]

rows = [compute_distribution(path, label) for path, label in conditions]
dist_df = pd.DataFrame(rows).set_index("model")
dist_df.to_csv("data/processed/wmt_distribution.csv")
print("\n", dist_df.to_string())

# ── BAR PLOT ──────────────────────────────────────────────────────────────────
# Mirrors the example screenshot:
#   X-axis = 4 model conditions
#   3 bars per group (Specific, Global, NA)
#   Y-axis = Percentage of Studies (%)

COLORS = {
    "Specific": "#3bba91",   # teal-green
    "Global":   "#f4845f",   # orange-salmon
    "NA":       "#8b93c9",   # muted blue-purple
}
categories = ["Specific", "Global", "NA"]
labels     = dist_df.index.tolist()   # 4 model conditions

n_cond  = len(labels)
n_cats  = len(categories)
bar_w   = 0.22
x       = np.arange(n_cond)

# Centre the 3 bars around each x tick
offsets = np.array([-1, 0, 1]) * bar_w

fig, ax = plt.subplots(figsize=(10, 6))

for i, cat in enumerate(categories):
    vals = dist_df[cat].values
    bars = ax.bar(x + offsets[i], vals, bar_w,
                  color=COLORS[cat], label=cat,
                  edgecolor="white", linewidth=0.5)


ax.set_ylabel("Percentage of Studies (%)", fontsize=12, fontweight="bold")
ax.set_xlabel("Model",                      fontsize=12, fontweight="bold")
ax.set_title("Distribution of WMT Extractions",
             fontsize=13, fontweight="bold", pad=12)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
ax.set_ylim(0, 105)
for lbl in ax.get_yticklabels():
    lbl.set_fontweight("bold")
ax.tick_params(axis="y", labelsize=10)

# Legend 
legend_patches = [mpatches.Patch(color=COLORS[c], label=c) for c in categories]
ax.legend(handles=legend_patches, title="",
          title_fontsize=10, fontsize=10,
          frameon=True, prop={"weight": "bold"})

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.35)

plt.tight_layout()
out_path = "data/processed/wmt_distribution_gpt4_vs_gpt5.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"\nPlot saved → {out_path}")
plt.show()