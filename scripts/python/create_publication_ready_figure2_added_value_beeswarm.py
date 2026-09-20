from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import TwoSlopeNorm
from PIL import Image, ImageChops


ROOT = Path(r"D:\DING PROJECT\00_READ_ME_FIRST_2018_V3_RESULTS")
OUTDIR = Path(r"D:\DING PROJECT\05_reports\manuscript_drafts\assembled_figures")

ADDED_VALUE = (
    ROOT
    / "04_probability_maps_250m/02_uncertainty_added_value_reliability/"
    / "cpec_2018_fused_minus_conventional_added_value_250m.tif"
)
BEESWARM = (
    ROOT
    / "03_figures/02_shap_and_explainability/"
    / "figure_factor_shap_beeswarm_2018_conventional_alphaearth_embeddings.png"
)

OUT_PNG = OUTDIR / "figure_R2_added_value_and_beeswarm_pubready.png"
OUT_PDF = OUTDIR / "figure_R2_added_value_and_beeswarm_pubready.pdf"


def read_raster(path, max_dim=1400):
    with rasterio.open(path) as src:
        scale = max(src.width, src.height) / max_dim
        if scale < 1:
            out_h, out_w = src.height, src.width
        else:
            out_h = int(src.height / scale)
            out_w = int(src.width / scale)
        data = src.read(1, out_shape=(out_h, out_w), masked=True)
        bounds = src.bounds
    arr = data.filled(np.nan).astype("float32")
    arr[~np.isfinite(arr)] = np.nan
    return arr, (bounds.left, bounds.right, bounds.bottom, bounds.top)


def trim_white_margin(image):
    image = image.convert("RGB")
    background = Image.new("RGB", image.size, "white")
    diff = ImageChops.difference(image, background)
    bbox = diff.getbbox()
    if bbox:
        pad = 16
        left = max(0, bbox[0] - pad)
        upper = max(0, bbox[1] - pad)
        right = min(image.size[0], bbox[2] + pad)
        lower = min(image.size[1], bbox[3] + pad)
        return image.crop((left, upper, right, lower))
    return image


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    added, extent = read_raster(ADDED_VALUE)
    q = np.nanpercentile(np.abs(added), 99)
    limit = float(max(q, 0.10))
    limit = min(limit, 0.65)

    beeswarm = trim_white_margin(Image.open(BEESWARM))

    fig = plt.figure(figsize=(12.0, 5.15), dpi=300)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.02, 1.20], wspace=0.10)

    ax_map = fig.add_subplot(gs[0, 0])
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad((1, 1, 1, 0))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    im = ax_map.imshow(added, extent=extent, cmap=cmap, norm=norm, interpolation="nearest")
    ax_map.set_title("(a) Fused minus Conventional probability", loc="left", fontsize=9.6, weight="bold")
    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")
    ax_map.tick_params(labelsize=8)
    ax_map.grid(color="0.75", linewidth=0.35, alpha=0.6)
    cbar = fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.02)
    cbar.set_label("Probability difference", fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.5)

    ax_shap = fig.add_subplot(gs[0, 1])
    ax_shap.imshow(beeswarm)
    ax_shap.set_title("(b) Factor-level TreeSHAP beeswarm", loc="left", fontsize=9.6, weight="bold")
    ax_shap.axis("off")

    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.075, top=0.93)
    fig.savefig(OUT_PNG, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(OUT_PNG)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
