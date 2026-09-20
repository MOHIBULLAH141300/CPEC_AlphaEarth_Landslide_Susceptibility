"""Generate the publication-ready methodology flowchart for manuscript V6."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PROJECT_ROOT = Path(r"D:\DING PROJECT")
OUTPUT_DIR = (
    PROJECT_ROOT
    / "ding review"
    / "v6_2026-08-01"
    / "figures"
)
OUTPUT_STEM = OUTPUT_DIR / "Figure_2_methodological_workflow_refined"


PALETTE = {
    "blue": ("#2F6FB0", "#EAF2FB"),
    "teal": ("#258B88", "#E8F6F4"),
    "purple": ("#7652A5", "#F1ECF8"),
    "orange": ("#D9841F", "#FFF3E4"),
    "green": ("#2F7D45", "#EAF5EC"),
}


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )


def rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    facecolor: str,
    edgecolor: str,
    title: str,
    body: str = "",
    title_color: str = "#17212B",
    body_color: str = "#303840",
    title_size: float = 9.4,
    body_size: float = 8.0,
    title_position: float = 0.72,
    body_position: float = 0.25,
    linewidth: float = 1.0,
    radius: float = 0.012,
    padding: float = 0.004,
    zorder: int = 5,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad={padding},rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(patch)
    title_y = y + height * (title_position if body else 0.50)
    title_text = ax.text(
        x + width / 2,
        title_y,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=title_color,
        linespacing=1.05,
        zorder=zorder + 1,
    )
    body_text = None
    if body:
        body_text = ax.text(
            x + width / 2,
            y + height * body_position,
            body,
            ha="center",
            va="center",
            fontsize=body_size,
            color=body_color,
            linespacing=1.25,
            zorder=zorder + 1,
        )
    if not hasattr(ax, "_box_text_groups"):
        ax._box_text_groups = []
    ax._box_text_groups.append((patch, title_text, body_text))


def fit_and_validate_box_text(fig: plt.Figure, ax: plt.Axes) -> None:
    """Fit long labels to their boxes and reject any remaining overflow."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    groups = getattr(ax, "_box_text_groups", [])

    for patch, title_text, body_text in groups:
        patch_box = patch.get_window_extent(renderer)
        for text in (title_text, body_text):
            if text is None:
                continue
            while (
                text.get_window_extent(renderer).width > patch_box.width * 0.90
                and text.get_fontsize() > 6.6
            ):
                text.set_fontsize(text.get_fontsize() - 0.15)
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    issues: list[str] = []
    for patch, title_text, body_text in groups:
        patch_box = patch.get_window_extent(renderer)
        for role, text in (("title", title_text), ("body", body_text)):
            if text is None:
                continue
            text_box = text.get_window_extent(renderer)
            if (
                text_box.x0 < patch_box.x0 + 1
                or text_box.x1 > patch_box.x1 - 1
                or text_box.y0 < patch_box.y0 + 1
                or text_box.y1 > patch_box.y1 - 1
            ):
                issues.append(f"{role} outside box: {text.get_text()!r}")

        if body_text is not None:
            title_box = title_text.get_window_extent(renderer)
            body_box = body_text.get_window_extent(renderer)
            if title_box.y0 < body_box.y1 + 2:
                issues.append(
                    f"title/body collision: {title_text.get_text()!r} / "
                    f"{body_text.get_text()!r}"
                )

    if issues:
        raise RuntimeError("Figure text QA failed:\n" + "\n".join(issues))


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = "#39434D",
    linewidth: float = 1.35,
    mutation_scale: float = 12,
    zorder: int = 4,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=linewidth,
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=zorder,
        )
    )


def phase_container(
    ax: plt.Axes,
    y: float,
    height: float,
    number: int,
    label: str,
    color_key: str,
) -> tuple[str, str]:
    accent, pale = PALETTE[color_key]
    container = FancyBboxPatch(
        (0.075, y),
        0.85,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.014",
        facecolor=pale,
        edgecolor=accent,
        linewidth=1.05,
        zorder=1,
    )
    ax.add_patch(container)
    rounded_box(
        ax,
        0.10,
        y + height - 0.037,
        0.80,
        0.027,
        accent,
        accent,
        f"{number}. {label}",
        title_color="white",
        title_size=9.6,
        linewidth=0.8,
        radius=0.010,
        zorder=7,
    )
    return accent, pale


def build() -> None:
    configure()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.25, 9.55), facecolor="white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Short title suitable for a manuscript figure.
    rounded_box(
        ax,
        0.11,
        0.925,
        0.78,
        0.061,
        "#245F9D",
        "#245F9D",
        "Transferability-aware CPEC landslide\nsusceptibility framework",
        title_color="white",
        title_size=11.4,
        linewidth=0.8,
        radius=0.012,
        zorder=10,
    )

    # Phase 1
    y1, h1 = 0.772, 0.124
    blue, blue_pale = phase_container(ax, y1, h1, 1, "DATA PREPARATION", "blue")
    box_y, box_h, box_w = y1 + 0.015, 0.060, 0.225
    xs = [0.105, 0.3875, 0.670]
    rounded_box(
        ax,
        xs[0],
        box_y,
        box_w,
        box_h,
        "white",
        blue,
        "Study area",
        "Official CPEC boundary\nTwo five-domain partitions",
        body_size=7.7,
    )
    rounded_box(
        ax,
        xs[1],
        box_y,
        box_w,
        box_h,
        "white",
        blue,
        "Samples",
        "1,508 slope failures\n1,808 controls",
    )
    rounded_box(
        ax,
        xs[2],
        box_y,
        box_w,
        box_h,
        "white",
        blue,
        "Harmonised data",
        "3,316 samples\n250 m grid",
    )
    arrow(ax, (xs[0] + box_w, box_y + box_h / 2), (xs[1] - 0.012, box_y + box_h / 2), blue)
    arrow(ax, (xs[1] + box_w, box_y + box_h / 2), (xs[2] - 0.012, box_y + box_h / 2), blue)

    # Phase 2
    y2, h2 = 0.602, 0.136
    teal, teal_pale = phase_container(ax, y2, h2, 2, "PREDICTOR CONFIGURATIONS", "teal")
    rounded_box(
        ax,
        0.185,
        y2 + 0.061,
        0.63,
        0.029,
        "white",
        teal,
        "VIF < 5 | training-fold preprocessing",
        title_size=9.1,
        linewidth=0.9,
        radius=0.009,
    )
    feature_y, feature_h, feature_w = y2 - 0.003, 0.055, 0.225
    rounded_box(
        ax,
        xs[0],
        feature_y,
        feature_w,
        feature_h,
        teal,
        teal,
        "Conventional",
        "19 factors",
        title_color="white",
        body_color="white",
        title_size=9.2,
        body_size=8.1,
    )
    rounded_box(
        ax,
        xs[1],
        feature_y,
        feature_w,
        feature_h,
        teal,
        teal,
        "AlphaEarth\nEmbeddings",
        "64 axes",
        title_color="white",
        body_color="white",
        title_size=8.3,
        body_size=7.8,
        title_position=0.74,
        body_position=0.19,
    )
    rounded_box(
        ax,
        xs[2],
        feature_y,
        feature_w,
        feature_h,
        teal,
        teal,
        "Fusion",
        "Conventional + AlphaEarth\n83 variables",
        title_color="white",
        body_color="white",
        title_size=9.2,
        body_size=7.3,
        title_position=0.80,
        body_position=0.25,
    )

    # Phase 3
    y3, h3 = 0.425, 0.137
    purple, purple_pale = phase_container(ax, y3, h3, 3, "NESTED SPATIAL-CV STACKING", "purple")
    model_y, model_h = y3 + 0.014, 0.075
    rounded_box(
        ax,
        xs[0],
        model_y,
        box_w,
        model_h,
        "white",
        purple,
        "Buffered Spatial-CV",
        "5 outer + 5 inner folds\n20 km buffer",
    )
    rounded_box(
        ax,
        xs[1],
        model_y,
        box_w,
        model_h,
        "white",
        purple,
        "Six base learners",
        "LR, RF, Extra Trees\nXGBoost, LightGBM\nCatBoost",
        title_size=9.2,
        body_size=7.1,
        title_position=0.76,
        body_position=0.28,
    )
    rounded_box(
        ax,
        xs[2],
        model_y,
        box_w,
        model_h,
        "white",
        purple,
        "Stacked model",
        "Logistic meta-learner\nOut-of-fold predictions",
        title_size=9.2,
        body_size=7.7,
    )
    arrow(ax, (xs[0] + box_w, model_y + model_h / 2), (xs[1] - 0.012, model_y + model_h / 2), purple)
    arrow(ax, (xs[1] + box_w, model_y + model_h / 2), (xs[2] - 0.012, model_y + model_h / 2), purple)

    # Phase 4
    y4, h4 = 0.254, 0.132
    orange, orange_pale = phase_container(ax, y4, h4, 4, "EVALUATION AND ROBUSTNESS", "orange")
    eval_y, eval_h = y4 + 0.014, 0.071
    rounded_box(
        ax,
        xs[0],
        eval_y,
        box_w,
        eval_h,
        "white",
        orange,
        "Performance",
        "ROC-AUC, PR-AUC, F1\nBrier score, ECE",
    )
    rounded_box(
        ax,
        xs[1],
        eval_y,
        box_w,
        eval_h,
        "white",
        orange,
        "Uncertainty and\nexplanation",
        "2,000 block bootstraps\nTreeSHAP + stack weights",
        title_size=8.4,
        body_size=7.6,
        title_position=0.75,
        body_position=0.22,
    )
    rounded_box(
        ax,
        xs[2],
        eval_y,
        box_w,
        eval_h,
        "white",
        orange,
        "Design sensitivity",
        "Matched controls + typology\nBlocks, buffers + road ablation",
        title_size=8.4,
        body_size=7.0,
    )

    # Phase 5
    y5, h5 = 0.088, 0.132
    green, green_pale = phase_container(ax, y5, h5, 5, "TRANSFERABILITY, SUPPORT, AND APPLICATION", "green")
    transfer_y, transfer_h = y5 + 0.014, 0.071
    rounded_box(
        ax,
        xs[0],
        transfer_y,
        box_w,
        transfer_h,
        "white",
        green,
        "Comparative transfer",
        "Three feature sets\nTwo five-domain LODO tests",
        title_size=8.4,
        body_size=7.2,
    )
    rounded_box(
        ax,
        xs[1],
        transfer_y,
        box_w,
        transfer_h,
        "white",
        green,
        "Transfer AoA",
        "Source-only target support\n16-setting sensitivity",
        title_size=8.6,
        body_size=7.2,
    )
    rounded_box(
        ax,
        xs[2],
        transfer_y,
        box_w,
        transfer_h,
        "white",
        green,
        "Deployment support",
        "Full-data AoA + disagreement\nCPEC and KKH priorities",
        title_size=8.5,
        body_size=7.0,
    )

    # Main vertical spine.
    for upper_y, lower_y in [
        (y1, y2 + h2),
        (y2, y3 + h3),
        (y3, y4 + h4),
        (y4, y5 + h5),
    ]:
        arrow(
            ax,
            (0.5, upper_y - 0.006),
            (0.5, lower_y + 0.006),
            color="#3C4650",
            linewidth=1.6,
            mutation_scale=14,
            zorder=3,
        )

    # Final output.
    rounded_box(
        ax,
        0.125,
        0.003,
        0.75,
        0.054,
        "#2F7D45",
        "#2F7D45",
        "TRANSFERABILITY-AWARE SUSCEPTIBILITY PRODUCTS",
        "Three 250 m score maps | comparative LODO + AoA diagnostics |\n"
        "supported and verification-priority road segments",
        title_color="white",
        body_color="white",
        title_size=9.8,
        body_size=7.8,
        title_position=0.82,
        body_position=0.27,
        linewidth=0.9,
        radius=0.012,
        zorder=8,
    )
    arrow(
        ax,
        (0.5, y5 - 0.006),
        (0.5, 0.065),
        color="#3C4650",
        linewidth=1.6,
        mutation_scale=10,
        zorder=3,
    )

    fit_and_validate_box_text(fig, ax)

    fig.savefig(
        OUTPUT_STEM.with_suffix(".png"),
        dpi=600,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.08,
    )
    fig.savefig(
        OUTPUT_STEM.with_suffix(".pdf"),
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.08,
    )
    fig.savefig(
        OUTPUT_STEM.with_suffix(".svg"),
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.08,
    )
    plt.close(fig)

    for suffix in [".png", ".pdf", ".svg"]:
        print(OUTPUT_STEM.with_suffix(suffix))


if __name__ == "__main__":
    build()
