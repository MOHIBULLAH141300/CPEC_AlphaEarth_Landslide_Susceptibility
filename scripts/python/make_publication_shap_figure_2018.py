"""Create publication-ready SHAP tree-importance figures for 2018 models."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(r"D:\DING PROJECT")
SHAP_DIR = PROJECT_ROOT / "03_models" / "shap_2018_tree_importance"
OUT_DIR = PROJECT_ROOT / "05_reports" / "figures"
REPORT = PROJECT_ROOT / "05_reports" / "publication_shap_figure_2018.md"

FILES = [
    (
        "Conventional",
        SHAP_DIR / "shap_importance_conventional_xgboost.csv",
    ),
    (
        "AlphaEarth Embeddings",
        SHAP_DIR / "shap_importance_alphaearth_embeddings_xgboost.csv",
    ),
    (
        "Conventional + AlphaEarth Embeddings",
        SHAP_DIR / "shap_importance_conventional_alphaearth_embeddings_xgboost.csv",
    ),
]


def short_label(name: str) -> str:
    replacements = {
        "Maximum 1-Day Rainfall": "Max 1-day rainfall",
        "Mean Daytime LST": "Mean daytime LST",
        "Monsoon Rainfall": "Monsoon rainfall",
        "MODIS Land Cover": "MODIS land cover",
        "NDVI Median": "NDVI median",
        "NDVI Amplitude": "NDVI amplitude",
    }
    return replacements.get(name, name)


def read_top(path: Path, top_n: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["feature_name"] = df["feature_name"].map(short_label)
    return df.sort_values("mean_abs_shap", ascending=False).head(top_n).iloc[::-1]


def make_three_panel(top_n: int = 12) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 6.2), sharex=False)
    colors = ["#4477AA", "#228833", "#CC6677"]

    for idx, (ax, (title, path), color) in enumerate(zip(axes, FILES, colors), start=1):
        top = read_top(path, top_n)
        ax.barh(top["feature_name"], top["mean_abs_shap"], color=color, edgecolor="none")
        ax.set_title(f"({chr(96 + idx)}) {title}", loc="left", fontsize=11, fontweight="bold")
        ax.set_xlabel("Mean |SHAP value|", fontsize=10)
        ax.tick_params(axis="both", labelsize=9)
        ax.grid(axis="x", color="#d9d9d9", linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)

    fig.suptitle("XGBoost SHAP Variable Importance for 2018 Landslide Susceptibility Models", fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=2.0)
    out = OUT_DIR / "figure_shap_tree_importance_2018_three_panel.png"
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return out


def make_fused_single(top_n: int = 20) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    title, path = FILES[-1]
    top = read_top(path, top_n)
    fig, ax = plt.subplots(figsize=(7.2, 7.8))
    ax.barh(top["feature_name"], top["mean_abs_shap"], color="#CC6677", edgecolor="none")
    ax.set_title("XGBoost SHAP Variable Importance\nConventional + AlphaEarth Embeddings", fontsize=12, fontweight="bold")
    ax.set_xlabel("Mean |SHAP value|", fontsize=10)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    fig.tight_layout()
    out = OUT_DIR / "figure_shap_tree_importance_2018_fused_model.png"
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    panel = make_three_panel()
    fused = make_fused_single()
    lines = [
        "# Publication SHAP Figure 2018",
        "",
        "Publication-ready SHAP tree-importance figures were generated from the final XGBoost models.",
        "",
        "## Files",
        "",
        f"- Three-panel figure: `{panel}`",
        f"- Fused model figure: `{fused}`",
        "",
        "## Suggested Caption",
        "",
        "SHAP variable importance for XGBoost landslide susceptibility models in 2018. Bars show mean absolute SHAP values, indicating each predictor's average contribution to model output magnitude. AlphaEarth variables represent latent annual satellite embedding dimensions.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(panel)
    print(fused)


if __name__ == "__main__":
    main()
