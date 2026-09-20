"""Build the corrected Geoscience Frontiers manuscript-v3 DOCX from Markdown."""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT, WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor
from PIL import Image


PROJECT = Path(r"D:\DING PROJECT")
PAPER = PROJECT / "FINAL PAPER" / "Manuscript_single_file" / "v3_1_2026-07-17"
SOURCE = PAPER / "manuscript_v31_source.md"
OUTPUT = PAPER / "CPEC_manuscript_v3_1_expanded_results_2026-07-17.docx"
FIGURES = PAPER / "figures"

FIGURE_FILES = {
    1: FIGURES / "Figure_1_study_area.png",
    2: FIGURES / "Figure_2_methodological_workflow.png",
    3: FIGURES / "Figure_3_nested_spatial_cv_validation.png",
    4: FIGURES / "Figure_4_baseline_susceptibility_maps.png",
    5: FIGURES / "Figure_5_alphaearth_added_value_and_explainability.png",
    6: FIGURES / "Figure_6_sampling_typology_and_road_robustness.png",
    7: FIGURES / "Figure_7_subdomain_scores_and_transferability.png",
    8: FIGURES / "Figure_8_harmonised_area_of_applicability.png",
    9: FIGURES / "Figure_9_road_exposure_and_ablation.png",
}


def set_cell_margins(cell, top=70, start=85, bottom=70, end=85):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for side, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{side}"))
        if node is None:
            node = OxmlElement(f"w:{side}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_table_borders(table):
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.first_child_found_in("w:tblBorders")
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    settings = {
        "top": ("single", "8", "444444"),
        "bottom": ("single", "8", "444444"),
        "insideH": ("single", "4", "B5B5B5"),
        "left": ("nil", "0", "FFFFFF"),
        "right": ("nil", "0", "FFFFFF"),
        "insideV": ("nil", "0", "FFFFFF"),
    }
    for edge, (val, size, color) in settings.items():
        element = borders.find(qn(f"w:{edge}"))
        if element is None:
            element = OxmlElement(f"w:{edge}")
            borders.append(element)
        element.set(qn("w:val"), val)
        element.set(qn("w:sz"), size)
        element.set(qn("w:color"), color)


def set_repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    header = OxmlElement("w:tblHeader")
    header.set(qn("w:val"), "true")
    tr_pr.append(header)


def set_cell_width(cell, width_inches: float):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_w = tc_pr.first_child_found_in("w:tcW")
    if tc_w is None:
        tc_w = OxmlElement("w:tcW")
        tc_pr.append(tc_w)
    tc_w.set(qn("w:w"), str(int(width_inches * 1440)))
    tc_w.set(qn("w:type"), "dxa")


def set_table_width(table, width_inches: float):
    tbl_pr = table._tbl.tblPr
    tbl_w = tbl_pr.first_child_found_in("w:tblW")
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(int(width_inches * 1440)))
    tbl_w.set(qn("w:type"), "dxa")
    autofit = tbl_pr.first_child_found_in("w:tblLayout")
    if autofit is None:
        autofit = OxmlElement("w:tblLayout")
        tbl_pr.append(autofit)
    autofit.set(qn("w:type"), "fixed")


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instr, separate, end])


def configure_section(section, landscape=False):
    section.page_width = Cm(29.7 if landscape else 21.0)
    section.page_height = Cm(21.0 if landscape else 29.7)
    section.orientation = WD_ORIENT.LANDSCAPE if landscape else WD_ORIENT.PORTRAIT
    section.top_margin = Cm(2.25)
    section.bottom_margin = Cm(2.15)
    section.left_margin = Cm(2.25)
    section.right_margin = Cm(2.25)
    section.header_distance = Cm(1.0)
    section.footer_distance = Cm(1.0)
    section.footer.is_linked_to_previous = False
    paragraph = section.footer.paragraphs[0]
    for child in list(paragraph._p):
        paragraph._p.remove(child)
    add_page_number(paragraph)


def configure_styles(doc: Document):
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    normal.font.size = Pt(10.5)
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    normal.paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    normal.paragraph_format.line_spacing = 1.12
    normal.paragraph_format.space_after = Pt(4.5)

    for style_name, size, before, after in (
        ("Heading 1", 13.0, 12, 5),
        ("Heading 2", 11.5, 9, 3),
        ("Heading 3", 10.5, 7, 2),
    ):
        style = styles[style_name]
        style.font.name = "Arial"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True

    caption = styles["Caption"]
    caption.font.name = "Times New Roman"
    caption._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    caption.font.size = Pt(9)
    caption.font.italic = False
    caption.font.color.rgb = RGBColor(0, 0, 0)
    caption.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    caption.paragraph_format.space_before = Pt(3)
    caption.paragraph_format.space_after = Pt(7)

    table_text = styles.add_style("Table Text", 1)
    table_text.font.name = "Arial"
    table_text._element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")
    table_text.font.size = Pt(7.6)
    table_text.paragraph_format.space_after = Pt(0)
    table_text.paragraph_format.line_spacing = 1.0

    equation = styles.add_style("Equation V3", 1)
    equation.font.name = "Cambria Math"
    equation._element.rPr.rFonts.set(qn("w:eastAsia"), "Cambria Math")
    equation.font.size = Pt(10.5)
    equation.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    equation.paragraph_format.space_before = Pt(5)
    equation.paragraph_format.space_after = Pt(5)


def strip_markdown(text: str) -> str:
    text = text.replace("`", "")
    text = re.sub(r"\^([^\^]+)\^", r"\1", text)
    return text.replace("**", "")


def add_rich_paragraph(doc, text: str, style=None):
    p = doc.add_paragraph(style=style)
    if text.startswith("**Keywords:**"):
        p.add_run("Keywords:").bold = True
        p.add_run(strip_markdown(text[len("**Keywords:**") :]))
        return p
    if text.startswith("**") and "**" in text[2:]:
        end = text.find("**", 2)
        p.add_run(strip_markdown(text[2:end])).bold = True
        p.add_run(strip_markdown(text[end + 2 :]))
        return p
    if "^" in text:
        for part in re.split(r"(\^[^\^]+\^)", text):
            if not part:
                continue
            if part.startswith("^") and part.endswith("^"):
                run = p.add_run(strip_markdown(part[1:-1]))
                run.font.superscript = True
            else:
                p.add_run(strip_markdown(part))
    else:
        p.add_run(strip_markdown(text))
    return p


def table_column_widths(table_number: int, ncols: int, usable: float):
    preferred = {
        1: [0.35, 0.95, 1.35, 1.55, 2.30],
        2: [0.35, 1.20, 2.05, 2.85],
        3: [0.35, 1.35, 1.25, 1.25, 1.25, 1.15, 1.25, 0.65],
        4: [0.35, 2.35, 1.15, 1.15, 1.70, 0.85],
        5: [0.35, 2.55, 1.15, 0.90, 0.90, 1.05, 0.75, 0.90],
        6: [0.35, 1.45, 0.70, 0.70, 1.55, 1.55, 1.60, 0.60, 0.75],
        7: [0.32, 1.25, 0.72, 0.68, 0.78, 0.78, 0.78, 0.92, 0.68, 0.88, 0.78],
    }
    widths = preferred.get(table_number, [usable / ncols] * ncols)
    scale = usable / sum(widths)
    return [w * scale for w in widths]


def add_table(doc, caption: str, rows: list[list[str]], table_number: int):
    landscape = len(rows[0]) >= 8
    if landscape:
        section = doc.add_section(WD_SECTION.NEW_PAGE)
        configure_section(section, landscape=True)
        usable = 10.65
    else:
        usable = 6.55

    cap = add_rich_paragraph(doc, caption, style="Caption")
    cap.paragraph_format.keep_with_next = True

    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_table_width(table, usable)
    set_table_borders(table)
    widths = table_column_widths(table_number, len(rows[0]), usable)

    for r_idx, row in enumerate(rows):
        for c_idx, value in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            set_cell_width(cell, widths[c_idx])
            set_cell_margins(cell)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cell.paragraphs[0]
            p.style = doc.styles["Table Text"]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx == 0 or (r_idx > 0 and c_idx >= 2) else WD_ALIGN_PARAGRAPH.LEFT
            run = p.add_run(strip_markdown(value))
            if r_idx == 0:
                run.bold = True
        if r_idx == 0:
            set_repeat_table_header(table.rows[0])

    doc.add_paragraph().paragraph_format.space_after = Pt(0)
    if landscape:
        section = doc.add_section(WD_SECTION.NEW_PAGE)
        configure_section(section, landscape=False)


def parse_table(lines: list[str], index: int):
    rows = []
    while index < len(lines) and lines[index].strip().startswith("|"):
        cells = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
        if not all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            rows.append(cells)
        index += 1
    return rows, index


def add_figure(doc, number: int, caption: str, first=False):
    path = FIGURE_FILES[number]
    if not path.exists():
        raise FileNotFoundError(path)
    p = doc.add_paragraph()
    if not first:
        p.paragraph_format.page_break_before = True
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    with Image.open(path) as image:
        aspect = image.width / image.height
    max_width = 10.20
    max_height = 5.85
    width = min(max_width, max_height * aspect)
    run = p.add_run()
    run.add_picture(str(path), width=Inches(width))
    descr = f"Figure {number}: {strip_markdown(caption)}"
    drawing = run._r.xpath(".//wp:docPr")
    if drawing:
        drawing[0].set("name", f"Figure {number}")
        drawing[0].set("descr", descr[:250])
    cap = add_rich_paragraph(doc, caption, style="Caption")
    cap.paragraph_format.keep_together = True


def build():
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    doc = Document()
    configure_section(doc.sections[0], landscape=False)
    configure_styles(doc)

    # Remove the package's initial empty paragraph only after adding real content.
    in_references = False
    in_figures = False
    first_figure = False
    table_number = 0
    index = 0

    while index < len(lines):
        raw = lines[index]
        text = raw.strip()
        index += 1
        if not text:
            continue

        if text.startswith("# "):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(8)
            run = p.add_run(strip_markdown(text[2:]))
            run.font.name = "Arial"
            run._element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")
            run.font.size = Pt(17)
            run.bold = True
            continue

        if text.startswith("## "):
            heading = strip_markdown(text[3:])
            in_references = heading == "References"
            in_figures = heading == "Figure captions"
            if in_figures:
                section = doc.add_section(WD_SECTION.NEW_PAGE)
                configure_section(section, landscape=True)
                first_figure = True
            else:
                doc.add_heading(heading, level=1)
            continue

        if text.startswith("### "):
            doc.add_heading(strip_markdown(text[4:]), level=2)
            continue

        if text.startswith("$$") and text.endswith("$$"):
            eq = text[2:-2].strip()
            if eq.startswith("VIF_j"):
                eq = "VIF_j = 1 / (1 - R_j²)"
            elif eq.startswith("s(x)"):
                eq = "s(x) = σ[β₀ + Σ(m=1...6) β_m z_m(x)]"
            elif eq.startswith("BS"):
                eq = "BS = (1/n) Σ(i=1...n) (s_i - y_i)²"
            elif eq.startswith("DI(x)"):
                eq = "DI(x) = min(j in T[-b]) ||q(x) - q(x_j)||₂ / Q₀.₉₅(d[-b])"
            elif eq.startswith("L_t"):
                eq = "L_t = Σ(k=1...K) ell_k I(s_k >= t)"
            elif eq.startswith("\\bar{s}_L"):
                eq = "s-bar_L = [Σ(k=1...K) ell_k s_k] / [Σ(k=1...K) ell_k]"
            else:
                replacements = {
                    r"\frac{1}{1-R_j^2}": "1 / (1 - R_j²)",
                    r"\sigma": "σ",
                    r"\left(": "(",
                    r"\right)": ")",
                    r"\sum_{m=1}^{6}": "Σ(m=1..6)",
                    r"\sum_{i=1}^{n}": "Σ(i=1..n)",
                    r"\sum_{k=1}^{K}": "Σ(k=1..K)",
                    r"\min_{j\in T_{-b}}": "min(j in T[-b])",
                    r"\|q(x)-q(x_j)\|_2": "||q(x) - q(x_j)||₂",
                    r"\geq": ">=",
                    r"\ell": "ell",
                    r"\bar{s}_L": "s-bar_L",
                }
                for old, new in replacements.items():
                    eq = eq.replace(old, new)
                eq = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1) / (\2)", eq)
                eq = eq.replace("\\", "").replace("{", "").replace("}", "")
            doc.add_paragraph(eq, style="Equation V3")
            continue

        if text.startswith("**Table "):
            table_number = int(re.search(r"Table (\d+)", text).group(1))
            while index < len(lines) and not lines[index].strip():
                index += 1
            rows, index = parse_table(lines, index)
            add_table(doc, text, rows, table_number)
            continue

        if in_figures and text.startswith("**Figure "):
            number = int(re.search(r"Figure (\d+)", text).group(1))
            add_figure(doc, number, text, first=first_figure)
            first_figure = False
            continue

        if in_references:
            p = add_rich_paragraph(doc, text)
            p.paragraph_format.left_indent = Cm(0.6)
            p.paragraph_format.first_line_indent = Cm(-0.6)
            p.paragraph_format.space_after = Pt(3.5)
            continue

        p = add_rich_paragraph(doc, text)
        if len(doc.paragraphs) <= 5:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if text.startswith("[") and "AUTHOR" in text:
            for run in p.runs:
                run.font.color.rgb = RGBColor(156, 52, 52)

    # Core document properties and clean package metadata.
    props = doc.core_properties
    props.title = "Transferability-aware landslide susceptibility mapping with AlphaEarth embeddings along the CPEC"
    props.subject = "Corrected manuscript version 3"
    props.author = "Mohibullah and co-authors"
    props.keywords = "landslide susceptibility; CPEC; KKH; AlphaEarth; spatial cross-validation; AoA"
    props.comments = "Built from the verified manuscript-v3 evidence package."

    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    build()
