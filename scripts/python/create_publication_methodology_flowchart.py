from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle


OUT_DIR = Path(
    r"D:\DING PROJECT\00_READ_ME_FIRST_2018_V3_RESULTS"
    r"\03_figures\04_methodology_flowchart"
)
REPORT_DIR = Path(r"D:\DING PROJECT\05_reports\figures")
SCRIPT_OUT = OUT_DIR / "figure_cpec_kkh_publishable_methodology_flowchart_2026-05-10.png"


def wrap(text: str, width: int) -> str:
    lines: list[str] = []
    for part in text.split("\n"):
        if not part.strip():
            lines.append("")
        else:
            lines.extend(textwrap.wrap(part, width=width, break_long_words=False))
    return "\n".join(lines)


def box(
    ax,
    xy: tuple[float, float],
    wh: tuple[float, float],
    title: str,
    body: str,
    face: str,
    edge: str = "#334155",
    title_color: str = "#0f172a",
    fontsize: float = 7.2,
    z: int = 3,
):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.05",
        linewidth=1.15,
        edgecolor=edge,
        facecolor=face,
        zorder=z,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h - 0.15,
        wrap(title, 27),
        ha="center",
        va="top",
        fontsize=fontsize + 0.9,
        fontweight="bold",
        color=title_color,
        zorder=z + 1,
    )
    ax.text(
        x + 0.12,
        y + h - 0.44,
        wrap(body, 32),
        ha="left",
        va="top",
        fontsize=fontsize,
        color="#1f2937",
        linespacing=1.15,
        zorder=z + 1,
    )
    return patch


def group_band(ax, x: float, y: float, w: float, h: float, label: str, color: str) -> None:
    rect = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.01,rounding_size=0.05",
        linewidth=0.9,
        edgecolor=color,
        facecolor=color,
        alpha=0.08,
        zorder=1,
    )
    ax.add_patch(rect)
    ax.text(
        x + 0.12,
        y + h - 0.08,
        label,
        ha="left",
        va="top",
        fontsize=8.5,
        fontweight="bold",
        color=color,
        zorder=2,
    )


def arrow(ax, start: tuple[float, float], end: tuple[float, float], color="#475569", lw=1.25, rad=0.0):
    a = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=lw,
        color=color,
        shrinkA=3,
        shrinkB=3,
        connectionstyle=f"arc3,rad={rad}",
        zorder=2,
    )
    ax.add_patch(a)


def small_label(ax, xy: tuple[float, float], text: str, color: str = "#475569") -> None:
    ax.text(
        xy[0],
        xy[1],
        wrap(text, 40),
        ha="center",
        va="center",
        fontsize=6.9,
        color=color,
        fontstyle="italic",
        zorder=4,
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(17.5, 10.2))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10.5)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Title
    ax.text(
        9,
        10.15,
        "Transferability-aware dynamic landslide susceptibility framework for the CPEC/KKH corridor",
        ha="center",
        va="center",
        fontsize=15.5,
        fontweight="bold",
        color="#0f172a",
    )
    ax.text(
        9,
        9.78,
        "Physical conditioning factors + AlphaEarth annual embeddings + spatial-CV stacked ensembles + uncertainty + infrastructure exposure",
        ha="center",
        va="center",
        fontsize=9.7,
        color="#475569",
    )

    # Background phase groups
    group_band(ax, 0.35, 6.05, 5.45, 3.2, "A. Data foundation and factor preparation", "#2563eb")
    group_band(ax, 6.05, 6.05, 5.65, 3.2, "B. 2018 baseline modelling and explanation", "#047857")
    group_band(ax, 11.95, 6.05, 5.7, 3.2, "C. Robustness and transferability", "#b45309")
    group_band(ax, 0.35, 1.1, 5.45, 4.55, "D. Annual dynamic mapping, 2017-2024", "#7c3aed")
    group_band(ax, 6.05, 1.1, 5.65, 4.55, "E. Uncertainty, scenarios, and infrastructure relevance", "#be123c")
    group_band(ax, 11.95, 1.1, 5.7, 4.55, "F. Manuscript-ready products", "#0f766e")

    # Phase A
    box(
        ax,
        (0.6, 8.05),
        (2.35, 0.9),
        "Official study area",
        "CPEC boundary\nPakistan + Kashgar\n250 m modelling grid",
        "#eef6ff",
        edge="#2563eb",
    )
    box(
        ax,
        (3.15, 8.05),
        (2.35, 0.9),
        "Landslide inventory",
        "CPEC + HMA dated events\npresence / control samples\nspatial folds",
        "#eef6ff",
        edge="#2563eb",
    )
    box(
        ax,
        (0.6, 6.55),
        (2.35, 1.18),
        "Conventional factors",
        "Terrain, rainfall, NDVI,\nland cover, lithology,\nsoil, roads, rivers, faults,\nseismicity",
        "#eef6ff",
        edge="#2563eb",
    )
    box(
        ax,
        (3.15, 6.55),
        (2.35, 1.18),
        "AlphaEarth embeddings",
        "Annual 64-band Satellite\nEmbedding from GEE\nA00-A63, 2017-2024",
        "#eef6ff",
        edge="#2563eb",
    )

    # Phase B
    box(
        ax,
        (6.35, 8.0),
        (2.4, 0.98),
        "Screening gate",
        "VIF + correlation checks\nremove redundant factors\nfinal Conventional set: 19",
        "#edfdf5",
        edge="#047857",
    )
    box(
        ax,
        (9.0, 8.0),
        (2.4, 0.98),
        "Three feature sets",
        "Conventional\nAlphaEarth Embeddings\nConventional + AlphaEarth",
        "#edfdf5",
        edge="#047857",
    )
    box(
        ax,
        (6.35, 6.48),
        (2.4, 1.2),
        "Spatial-CV model family",
        "LR, RF, Extra Trees,\nXGBoost, LightGBM, CatBoost\n5-fold spatial block CV",
        "#edfdf5",
        edge="#047857",
    )
    box(
        ax,
        (9.0, 6.48),
        (2.4, 1.2),
        "Stacked ensemble + XAI",
        "OOF meta-learner\nROC, PR, calibration, Brier\nSHAP + grouped embeddings",
        "#edfdf5",
        edge="#047857",
    )

    # Phase C
    box(
        ax,
        (12.25, 8.0),
        (2.45, 0.98),
        "CPEC subdomains",
        "KKH/high mountains\nIndus basin, west Pakistan\nsouthern corridor",
        "#fff7ed",
        edge="#b45309",
    )
    box(
        ax,
        (14.95, 8.0),
        (2.45, 0.98),
        "Transferability tests",
        "Leave-one-domain-out\nsingle/multi-source checks\nfeature-shift diagnostics",
        "#fff7ed",
        edge="#b45309",
    )
    box(
        ax,
        (12.25, 6.48),
        (2.45, 1.2),
        "Area of applicability",
        "KL / JS / MMD similarity\ndissimilarity raster\ntransfer confidence map",
        "#fff7ed",
        edge="#b45309",
    )
    box(
        ax,
        (14.95, 6.48),
        (2.45, 1.2),
        "Sampling robustness",
        "Repeated control sampling\nprobability mean and SD\nconfidence masks",
        "#fff7ed",
        edge="#b45309",
    )

    # Phase D
    box(
        ax,
        (0.6, 4.28),
        (2.35, 0.98),
        "Fixed 2018 model",
        "Hold trained V3 model constant\nannual changes reflect predictors",
        "#f5f3ff",
        edge="#7c3aed",
    )
    box(
        ax,
        (3.15, 4.28),
        (2.35, 0.98),
        "Annual predictors",
        "Rainfall, NDVI, land cover\nAlphaEarth embeddings\n2017-2024",
        "#f5f3ff",
        edge="#7c3aed",
    )
    box(
        ax,
        (0.6, 2.72),
        (2.35, 1.12),
        "Annual probability maps",
        "Conventional\nAlphaEarth Embeddings\nConventional + AlphaEarth\n250 m rasters",
        "#f5f3ff",
        edge="#7c3aed",
    )
    box(
        ax,
        (3.15, 2.72),
        (2.35, 1.12),
        "Temporal products",
        "Previous-year change\ntrend 2017-2024\nmean susceptibility\npersistent hotspots",
        "#f5f3ff",
        edge="#7c3aed",
    )
    small_label(ax, (3.0, 1.62), "Use careful wording: annual dynamic-factor maps; full temporal validation needs dated events.", "#6d28d9")

    # Phase E
    box(
        ax,
        (6.35, 4.25),
        (2.4, 1.03),
        "AlphaEarth added value",
        "Fused minus Conventional\nfeature-set disagreement\nembedding-change diagnostics",
        "#fff1f2",
        edge="#be123c",
    )
    box(
        ax,
        (9.0, 4.25),
        (2.4, 1.03),
        "Reliability and uncertainty",
        "Feature-set disagreement\nsampling uncertainty\ncalibration and Brier",
        "#fff1f2",
        edge="#be123c",
    )
    box(
        ax,
        (6.35, 2.68),
        (2.4, 1.14),
        "Counterfactual checks",
        "Road-distance ablation\nrainfall scenario\nPGA/seismic scenario",
        "#fff1f2",
        edge="#be123c",
    )
    box(
        ax,
        (9.0, 2.68),
        (2.4, 1.14),
        "KKH/CPEC exposure",
        "Road-segment hotspots\npersistent risk sections\nfield-verification priorities",
        "#fff1f2",
        edge="#be123c",
    )

    # Phase F
    box(
        ax,
        (12.25, 4.23),
        (2.45, 1.05),
        "Model evidence",
        "AUC, AP/PR-AUC, F1\nbalanced accuracy, Brier\ncalibration, spatial CV",
        "#ecfdf5",
        edge="#0f766e",
    )
    box(
        ax,
        (14.95, 4.23),
        (2.45, 1.05),
        "Maps and decision outputs",
        "2018 baseline, annual maps\nchange, trend, persistence\nuncertainty/reliability",
        "#ecfdf5",
        edge="#0f766e",
    )
    box(
        ax,
        (12.25, 2.62),
        (2.45, 1.18),
        "Interpretation",
        "SHAP/ALE mechanisms\nphysical factor explanation\nAlphaEarth contribution",
        "#ecfdf5",
        edge="#0f766e",
    )
    box(
        ax,
        (14.95, 2.62),
        (2.45, 1.18),
        "Publication story",
        "Transferable, dynamic\nuncertainty-aware, explainable\nCPEC/KKH relevant",
        "#ecfdf5",
        edge="#0f766e",
    )

    # Arrows within and between phases.
    arrows = [
        ((2.95, 8.5), (3.15, 8.5)),
        ((1.78, 8.05), (1.78, 7.73)),
        ((4.33, 8.05), (4.33, 7.73)),
        ((5.5, 7.15), (6.35, 8.45)),
        ((8.75, 8.5), (9.0, 8.5)),
        ((7.55, 8.0), (7.55, 7.68)),
        ((10.2, 8.0), (10.2, 7.68)),
        ((11.4, 7.1), (12.25, 8.42)),
        ((14.7, 8.5), (14.95, 8.5)),
        ((13.47, 8.0), (13.47, 7.68)),
        ((16.17, 8.0), (16.17, 7.68)),
        ((1.78, 4.28), (1.78, 3.84)),
        ((2.95, 4.77), (3.15, 4.77)),
        ((4.33, 4.28), (4.33, 3.84)),
        ((2.95, 3.27), (3.15, 3.27)),
        ((5.5, 3.3), (6.35, 4.78)),
        ((8.75, 4.77), (9.0, 4.77)),
        ((7.55, 4.25), (7.55, 3.82)),
        ((10.2, 4.25), (10.2, 3.82)),
        ((11.4, 3.28), (12.25, 4.75)),
        ((14.7, 4.75), (14.95, 4.75)),
        ((13.47, 4.23), (13.47, 3.8)),
        ((16.17, 4.23), (16.17, 3.8)),
    ]
    for s, e in arrows:
        arrow(ax, s, e)

    # Cross-links: baseline model into annual maps and robustness into final outputs.
    arrow(ax, (10.2, 6.48), (2.0, 5.25), color="#7c3aed", lw=1.0, rad=0.15)
    small_label(ax, (5.35, 5.58), "Fixed 2018 trained model", "#6d28d9")
    arrow(ax, (13.45, 6.48), (13.45, 5.28), color="#0f766e", lw=1.0)
    arrow(ax, (16.15, 6.48), (16.15, 5.28), color="#0f766e", lw=1.0)

    # Footer note
    footer = (
        "Figure: Proposed next methodology after teacher-sent literature review. "
        "Core novelty = annual AlphaEarth foundation embeddings + physically interpretable factors + transferability + uncertainty + CPEC/KKH exposure."
    )
    ax.add_patch(Rectangle((0.35, 0.18), 17.3, 0.55, facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=0.8))
    ax.text(9, 0.46, wrap(footer, 155), ha="center", va="center", fontsize=8.1, color="#334155")

    for ext in ["png", "svg", "pdf"]:
        out = OUT_DIR / f"figure_cpec_kkh_publishable_methodology_flowchart_2026-05-10.{ext}"
        if ext == "png":
            fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white")
            fig.savefig(REPORT_DIR / out.name, dpi=600, bbox_inches="tight", facecolor="white")
        else:
            fig.savefig(out, bbox_inches="tight", facecolor="white")
            fig.savefig(REPORT_DIR / out.name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(SCRIPT_OUT)


if __name__ == "__main__":
    main()
