
# %%
# eval_boot.py (patched plotting + correct mapping)
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from evaluation import build_state
field_predictions, fields, compute_for_field = build_state()

BOOTSTRAP_ITERATIONS = 10
CONFIDENCE_LEVEL = 0.95
OUTPUT_CSV = "bootstrapped_summary_all_fields.csv"

METRIC_MAP: Dict[str, str] = {
    "F1": "micro_f1",             
    "Accuracy": "subset_acc",
    "Jaccard": "jaccard_accuracy"  
}

def _bootstrap_distribution(preds, refs, metric_key, n_iterations, rng):
    if not preds or not refs:
        return [], float("nan")
    if len(preds) != len(refs):
        raise ValueError("Predictions and references must have equal length.")
    point_est = float(compute_for_field(preds, refs)[metric_key])
    n = len(preds)
    idx_all = np.arange(n)
    samples = []
    for _ in range(n_iterations):
        idx = rng.choice(idx_all, size=n, replace=True)
        res = compute_for_field([preds[i] for i in idx], [refs[i] for i in idx])
        samples.append(float(res[metric_key]))
    return samples, point_est

def _ci(values, level):
    if not values:
        return float("nan"), float("nan")
    a = 1.0 - level
    return (float(np.percentile(values, 100*(a/2))),
            float(np.percentile(values, 100*(1-a/2))))

def compute_bootstrap_summary(n_iterations=BOOTSTRAP_ITERATIONS,
                              confidence_level=CONFIDENCE_LEVEL,
                              random_seed: Optional[int]=42) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    rows = []
    field_order = [name for name, *_ in fields]
    for field_name in field_order:
        ds = field_predictions.get(field_name, {})
        preds = ds.get("predictions", [])
        refs  = ds.get("references", [])
        for label, key in METRIC_MAP.items():
            samples, est = _bootstrap_distribution(preds, refs, key, n_iterations, rng)
            lo, hi = _ci(samples, confidence_level)
            rows.append({
                "field": field_name, "metric": label,
                "estimate": est, "ci_lower": lo, "ci_upper": hi,
                "n_bootstrap": len(samples)
            })
    df = pd.DataFrame(rows)
    df["field"] = pd.Categorical(df["field"], categories=field_order, ordered=True)
    df["metric"] = pd.Categorical(df["metric"], categories=["F1","Accuracy","Jaccard"], ordered=True)
    df.sort_values(["field","metric"], inplace=True)
    df["err_up"] = df["ci_upper"] - df["estimate"]
    df["err_dn"] = df["estimate"] - df["ci_lower"]
    df["label"]  = df["estimate"].round(3)
    return df

def plot_grouped_bar(summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        print("No data available to plot.")
        return

    field_order  = [name for name, *_ in fields]
    metric_order = ["F1", "Accuracy", "Jaccard"]

    fig = go.Figure()

    for metric in metric_order:
        mdf = (summary_df[summary_df["metric"] == metric]
               .set_index("field")
               .reindex(field_order)
               .reset_index())

        fig.add_bar(
            x=mdf["field"],
            y=mdf["estimate"],
            name=metric,
            error_y=dict(
                type="data",
                array=(mdf["ci_upper"] - mdf["estimate"]),
                arrayminus=(mdf["estimate"] - mdf["ci_lower"]),
                thickness=1.8,
                width=4
            ),
            # text=mdf["estimate"].apply(lambda x: f"{x:.2f}"),
            # textposition="inside",          # keep labels inside the bars
            # textangle=0,                    # force horizontal
            # insidetextanchor="end", 
            # textfont=dict(
            #     color="white",
            #     size=13,
            #     family="Arial Black"  # bold white labels
            # ),
            hovertemplate=(
                "Field=%{x}<br>"
                f"Metric={metric}<br>"
                "Value=%{y:.2f}<br>"
                "CI=[%{customdata[0]:.2f}, %{customdata[1]:.2f}]<extra></extra>"
            ),
            customdata=np.stack((mdf["ci_lower"], mdf["ci_upper"]), axis=-1)
        )

    fig.update_layout(
        barmode="group",
        bargap=0.1,
        bargroupgap=0.12,
        template="plotly_white",
        yaxis=dict(
            range=[0, 1],
            title=dict(
                text="Score",
                font=dict(size=16, family="Arial Black", color="black")
            ),
            tickformat=".2f"
        ),
        xaxis=dict(
            title=dict(
                text="Field",
                font=dict(size=16, family="Arial Black", color="black")
            ),
            categoryorder="array",
            categoryarray=field_order
        ),
        title=dict(
            text="Bootstrapped Metrics with 95% CIs",
            x=0.5,
            xanchor="center",
            font=dict(size=16, family="Arial Black", color="black")
        ),
        legend_title_text="Metric",
        legend=dict(font=dict(size=12, family="Arial")),
        width=750,
        height=550,
        margin=dict(t=70, b=70, l=60, r=40),
        plot_bgcolor="white"
    )

    fig.show()

def plot_point_with_ci(summary_df: pd.DataFrame) -> None:
    """
    Create a point plot showing estimates with confidence interval error bars.
    Each metric gets its own subplot for clarity.
    """
    if summary_df.empty:
        print("No data available to plot.")
        return

    field_order  = [name for name, *_ in fields]
    metric_order = ["F1", "Accuracy", "Jaccard"]

    fig = go.Figure()

    for metric in metric_order:
        mdf = (summary_df[summary_df["metric"] == metric]
               .set_index("field")
               .reindex(field_order)
               .reset_index())

        fig.add_scatter(
            x=mdf["field"],
            y=mdf["estimate"],
            name=metric,
            mode="markers",
            marker=dict(size=10, symbol="square"),
            line=dict(width=2),
            error_y=dict(
                type="data",
                array=(mdf["ci_upper"] - mdf["estimate"]),
                arrayminus=(mdf["estimate"] - mdf["ci_lower"]),
                thickness=2,
                width=6
            ),
            hovertemplate=(
                "Field=%{x}<br>"
                f"Metric={metric}<br>"
                "Estimate=%{y:.3f}<br>"
                "CI=[%{customdata[0]:.3f}, %{customdata[1]:.3f}]<extra></extra>"
            ),
            customdata=np.stack((mdf["ci_lower"], mdf["ci_upper"]), axis=-1)
        )

    fig.update_layout(
        scattermode="group",
        bargap=0.1,
        bargroupgap=0.12,
        template="plotly_white",
        yaxis=dict(
            range=[0, 1],
            title=dict(
                text="Score",
                font=dict(size=16, family="Arial Black", color="black")
            ),
            tickformat=".2f"
        ),
        xaxis=dict(
            title=dict(
                text="Field",
                font=dict(size=16, family="Arial Black", color="black")
            ),
            categoryorder="array",
            categoryarray=field_order
        ),
        title=dict(
            text="Metrics with 95% CIs",
            x=0.5,
            xanchor="center",
            font=dict(size=16, family="Arial Black", color="black")
        ),
        legend_title_text="Metric",
        legend=dict(font=dict(size=12, family="Arial")),
        width=750,
        height=550,
        margin=dict(t=70, b=70, l=60, r=40),
        plot_bgcolor="white"
    )

    fig.show()

def main():
    df = compute_bootstrap_summary()
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved bootstrap summary to {OUTPUT_CSV}")
    print(df[["field", "metric", "estimate", "ci_lower", "ci_upper", "n_bootstrap"]]
          .round(3)
          .to_string(index=False))
    plot_grouped_bar(df)
    plot_point_with_ci(df)

if __name__ == "__main__":
    main()
# %%
