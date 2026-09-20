"""Create a Word DOCX for the updated Results section.

The builder converts the manuscript-ready Markdown Results section into a
formatted Word document with journal-style headings, tables, figures, and
captions. It intentionally keeps the layout formal and submission-friendly.
"""

from __future__ import annotations

import re
from pathlib import Path

from PIL import Image
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


PROJECT_ROOT = Path(r"D:\DING PROJECT")
MD_PATH = (
    PROJECT_ROOT
    / "05_reports"
    / "manuscript_drafts"
    / "geoscience_frontiers_results_section_updated_2026-06-06.md"
)
OUT_DOCX = (
    PROJECT_ROOT
    / "05_reports"
    / "manuscript_drafts"
    / "geoscience_frontiers_results_section_updated_2026-06-06.docx"
)
FIG_CACHE = (
    PROJECT_ROOT
    / "05_reports"
    / "manuscript_drafts"
    / "docx_figure_cache_2026-06-06"
)

CONTENT_WIDTH_IN = 6.5
CONTENT_WIDTH_DXA = 9360
TABLE_INDENT_DXA = 120


def set_cell_margins(table, top=80, start=120, bottom=80, end=120):
    tbl_pr = table._tbl.tblPr
    tbl_cell_mar = tbl_pr.first_child_found_in("w:tblCellMar")
    if tbl_cell_mar is None:
        tbl_cell_mar = OxmlElement("w:tblCellMar")
        tbl_pr.append(tbl_cell_mar)
    for m, v in [("top", top), ("start", start), ("bottom", bottom), ("end", end)]:
        node = tbl_cell_mar.find(qn(f"w:{m}"))
        if node is None:
            node = OxmlElement(f"w:{m}")
            tbl_cell_mar.append(node)
        node.set(qn("w:w"), str(v))
        node.set(qn("w:type"), "dxa")


def set_table_fixed_width(table, widths_dxa: list[int]):
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    table.autofit = False
    tbl = table._tbl
    tbl_pr = tbl.tblPr
    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:type"), "dxa")
    tbl_w.set(qn("w:w"), str(CONTENT_WIDTH_DXA))
    tbl_ind = tbl_pr.find(qn("w:tblInd"))
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tbl_pr.append(tbl_ind)
    tbl_ind.set(qn("w:w"), str(TABLE_INDENT_DXA))
    tbl_ind.set(qn("w:type"), "dxa")
    tbl_layout = tbl_pr.find(qn("w:tblLayout"))
    if tbl_layout is None:
        tbl_layout = OxmlElement("w:tblLayout")
        tbl_pr.append(tbl_layout)
    tbl_layout.set(qn("w:type"), "fixed")

    grid = tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width in widths_dxa:
        col = OxmlElement("w:gridCol")
        col.set(qn("w:w"), str(width))
        grid.append(col)

    for row in table.rows:
        for idx, cell in enumerate(row.cells):
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_w = tc_pr.find(qn("w:tcW"))
            if tc_w is None:
                tc_w = OxmlElement("w:tcW")
                tc_pr.append(tc_w)
            tc_w.set(qn("w:w"), str(widths_dxa[idx]))
            tc_w.set(qn("w:type"), "dxa")


def set_table_borders(table, color="D9DEE7"):
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.find(qn("w:tblBorders"))
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ["top", "left", "bottom", "right", "insideH", "insideV"]:
        tag = f"w:{edge}"
        node = borders.find(qn(tag))
        if node is None:
            node = OxmlElement(tag)
            borders.append(node)
        node.set(qn("w:val"), "single")
        node.set(qn("w:sz"), "4")
        node.set(qn("w:space"), "0")
        node.set(qn("w:color"), color)


def shade_cell(cell, fill="F2F4F7"):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_paragraph_spacing(paragraph, before=0, after=6, line=1.10):
    fmt = paragraph.paragraph_format
    fmt.space_before = Pt(before)
    fmt.space_after = Pt(after)
    fmt.line_spacing = line


def add_formatted_text(paragraph, text: str, *, bold_default=False, italic_default=False):
    """Add text with simple Markdown bold/italic markers."""
    token_re = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*)")
    pos = 0
    for match in token_re.finditer(text):
        if match.start() > pos:
            run = paragraph.add_run(text[pos : match.start()])
            run.bold = bold_default
            run.italic = italic_default
        token = match.group(0)
        inner = token.strip("*")
        run = paragraph.add_run(inner)
        run.bold = bold_default or token.startswith("**")
        run.italic = italic_default or (token.startswith("*") and not token.startswith("**"))
        pos = match.end()
    if pos < len(text):
        run = paragraph.add_run(text[pos:])
        run.bold = bold_default
        run.italic = italic_default


def set_run_font(run, size=None, color=None, bold=None, italic=None):
    run.font.name = "Calibri"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    if size is not None:
        run.font.size = Pt(size)
    if color is not None:
        run.font.color.rgb = RGBColor.from_string(color)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


def configure_styles(doc: Document):
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Calibri"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    normal.font.size = Pt(11)
    normal.font.color.rgb = RGBColor(0, 0, 0)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.10

    for style_name, size, color, before, after in [
        ("Heading 1", 16, "2E74B5", 16, 8),
        ("Heading 2", 13, "2E74B5", 12, 6),
        ("Heading 3", 12, "1F4D78", 8, 4),
    ]:
        style = styles[style_name]
        style.font.name = "Calibri"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(color)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.line_spacing = 1.10
        style.paragraph_format.keep_with_next = True


def table_widths(headers: list[str]) -> list[int]:
    n = len(headers)
    if n == 4:
        return [600, 2200, 5300, 1260]
    if n == 5:
        return [600, 2800, 1600, 1600, 3160]
    if n == 6:
        return [600, 2200, 1250, 1250, 1250, 2510]
    if n == 7:
        return [520, 2700, 1100, 1100, 1100, 1100, 1440]
    if n == 8:
        return [480, 2450, 1000, 1000, 1000, 950, 950, 1530]
    if n == 9:
        return [460, 600, 2200, 760, 1100, 1050, 800, 800, 1590]
    return [CONTENT_WIDTH_DXA // n] * n


def add_table(doc: Document, rows: list[list[str]]):
    if not rows:
        return
    headers = rows[0]
    widths = table_widths(headers)
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    set_table_fixed_width(table, widths)
    set_table_borders(table)
    set_cell_margins(table)

    for col_idx, text in enumerate(headers):
        cell = table.rows[0].cells[col_idx]
        shade_cell(cell, "F2F4F7")
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        header_text = "No" if text.strip() == "S.No." else text
        run = p.add_run(header_text)
        set_run_font(run, size=7.6 if len(headers) >= 8 else 8.0, bold=True)

    for row_values in rows[1:]:
        row = table.add_row()
        for col_idx, text in enumerate(row_values):
            cell = row.cells[col_idx]
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.05
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if col_idx in [1, 2] else WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(text)
            set_run_font(run, size=8.0)

    after = doc.add_paragraph()
    set_paragraph_spacing(after, before=0, after=4)


def add_image(doc: Document, path_text: str):
    image_path = Path(path_text)
    if not image_path.exists():
        p = doc.add_paragraph()
        add_formatted_text(p, f"[Missing figure: {path_text}]")
        return
    image_path = optimize_image_for_docx(image_path)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run()
    run.add_picture(str(image_path), width=Inches(6.25))


def optimize_image_for_docx(image_path: Path) -> Path:
    """Create a Word-friendly JPEG copy capped at 1800 px wide."""
    FIG_CACHE.mkdir(parents=True, exist_ok=True)
    out_path = FIG_CACHE / f"{image_path.stem}.jpg"
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            max_width = 1800
            if img.width > max_width:
                ratio = max_width / float(img.width)
                new_size = (max_width, max(1, int(img.height * ratio)))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            img.save(out_path, "JPEG", quality=90, optimize=True, progressive=False)
        return out_path
    except Exception:
        return image_path


def add_caption(doc: Document, text: str):
    p = doc.add_paragraph()
    set_paragraph_spacing(p, before=0, after=8, line=1.05)
    p.paragraph_format.keep_together = True
    run = p.add_run(text)
    set_run_font(run, size=9, italic=True, color="333333")


def add_table_caption(doc: Document, text: str):
    p = doc.add_paragraph()
    set_paragraph_spacing(p, before=8, after=4, line=1.05)
    p.paragraph_format.keep_with_next = True
    stripped = text.strip("*")
    run = p.add_run(stripped)
    set_run_font(run, size=10, bold=True, color="1F4D78")


def parse_markdown_table(lines: list[str], start: int) -> tuple[list[list[str]], int]:
    table_lines = []
    i = start
    while i < len(lines) and lines[i].strip().startswith("|"):
        table_lines.append(lines[i].strip())
        i += 1
    rows = []
    for idx, line in enumerate(table_lines):
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if idx == 1 and all(re.fullmatch(r":?-+:?", c) for c in cells):
            continue
        rows.append(cells)
    return rows, i


def add_paragraph(doc: Document, text: str):
    p = doc.add_paragraph()
    set_paragraph_spacing(p, before=0, after=6, line=1.10)
    add_formatted_text(p, text)


def build_docx():
    doc = Document()
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)
    section.header_distance = Inches(0.492)
    section.footer_distance = Inches(0.492)
    configure_styles(doc)

    # Quiet footer page number placeholder; Word will render field on open/update.
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = footer.add_run("Results section")
    set_run_font(run, size=8, color="666666")

    lines = MD_PATH.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        if line.startswith("# "):
            p = doc.add_paragraph(style="Heading 1")
            add_formatted_text(p, line[2:].strip(), bold_default=True)
            i += 1
            continue
        if line.startswith("## "):
            p = doc.add_paragraph(style="Heading 2")
            add_formatted_text(p, line[3:].strip(), bold_default=True)
            i += 1
            continue
        if line.startswith("### "):
            p = doc.add_paragraph(style="Heading 3")
            add_formatted_text(p, line[4:].strip(), bold_default=True)
            i += 1
            continue

        image_match = re.match(r"!\[[^\]]+\]\((.+)\)", line)
        if image_match:
            add_image(doc, image_match.group(1))
            i += 1
            continue

        if re.match(r"^Figure\s+\d+", line):
            add_caption(doc, line)
            i += 1
            continue

        if re.match(r"^\*\*Table\s+\d+\.", line):
            add_table_caption(doc, line)
            i += 1
            continue

        if line.startswith("|"):
            rows, i = parse_markdown_table(lines, i)
            add_table(doc, rows)
            continue

        add_paragraph(doc, line)
        i += 1

    OUT_DOCX.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT_DOCX)
    print(OUT_DOCX)


if __name__ == "__main__":
    build_docx()
