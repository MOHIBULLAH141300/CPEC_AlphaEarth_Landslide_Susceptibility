"""Render a PDF to page PNGs and compact contact sheets for visual QA."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--scale", type=float, default=1.55)
    parser.add_argument("--pages-per-sheet", type=int, default=6)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pages_dir = args.out_dir / "pages"
    pages_dir.mkdir(exist_ok=True)

    pdf = pdfium.PdfDocument(str(args.pdf))
    page_paths = []
    for index in range(len(pdf)):
        bitmap = pdf[index].render(scale=args.scale)
        image = bitmap.to_pil().convert("RGB")
        path = pages_dir / f"page-{index + 1:02d}.png"
        image.save(path, quality=95)
        page_paths.append(path)

    thumb_width = 520
    cols = 3
    rows = math.ceil(args.pages_per_sheet / cols)
    for sheet_index, start in enumerate(range(0, len(page_paths), args.pages_per_sheet), start=1):
        subset = page_paths[start : start + args.pages_per_sheet]
        thumbs = []
        for page_number, path in enumerate(subset, start=start + 1):
            image = Image.open(path).convert("RGB")
            height = round(image.height * thumb_width / image.width)
            thumb = image.resize((thumb_width, height), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (thumb_width + 20, height + 42), "white")
            canvas.paste(thumb, (10, 30))
            ImageDraw.Draw(canvas).text((12, 8), f"Page {page_number}", fill="black")
            thumbs.append(canvas)
        cell_w = max(t.width for t in thumbs)
        cell_h = max(t.height for t in thumbs)
        sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "#D8D8D8")
        for position, thumb in enumerate(thumbs):
            x = (position % cols) * cell_w
            y = (position // cols) * cell_h
            sheet.paste(thumb, (x, y))
        sheet.save(args.out_dir / f"contact-{sheet_index:02d}.jpg", quality=88)

    print(f"Rendered {len(page_paths)} pages to {args.out_dir}")


if __name__ == "__main__":
    main()
