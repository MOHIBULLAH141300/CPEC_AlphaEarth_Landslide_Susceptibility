from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(r"D:\DING PROJECT\00_READ_ME_FIRST_2018_V3_RESULTS")
OUTDIR = Path(r"D:\DING PROJECT\05_reports\manuscript_drafts")
FIGDIR = OUTDIR / "assembled_figures"

OUT_DOCX = OUTDIR / "geoscience_frontiers_full_results_section_pubready_2026-05-15.docx"
OUT_MD = OUTDIR / "geoscience_frontiers_full_results_section_pubready_2026-05-15.md"
AUDIT_MD = OUTDIR / "geoscience_frontiers_results_section_scope_style_audit_pubready_2026-05-15.md"


def load_inputs():
    factors = pd.read_csv(ROOT / "06_samples_factors_and_vif/readable_conventional_factor_list.csv")
    vif = pd.read_csv(ROOT / "06_samples_factors_and_vif/v3_final_selected_vif.csv")
    factors = factors.merge(vif, left_on="Raw field", right_on="factor", how="left").drop(columns=["factor"])

    return {
        "factors": factors,
        "perf": pd.read_csv(ROOT / "02_model_performance_tables/clean_model_performance_summary_for_supervisor.csv"),
        "stacked": pd.read_csv(ROOT / "02_model_performance_tables/stacked_ensemble_metrics_pooled.csv"),
        "domain": pd.read_csv(
            ROOT / "10_domain_specific_2018_results/00_domain_comparison_summary/domain_transferability_mechanism_comparison_table.csv"
        ),
        "road": pd.read_csv(ROOT / "05_road_exposure_outputs/road_exposure_summary_250m.csv"),
        "relroad": pd.read_csv(
            ROOT / "02_model_performance_tables/cpec_2018_reliability_aware_road_exposure_summary_by_domain.csv"
        ),
        "annual": pd.read_csv(
            ROOT / "11_dynamic_temporal_results_2017_2024/02_tables_and_diagnostics/annual_fused_probability_summary_2017_2024.csv"
        ),
        "temporal_diag": pd.read_csv(
            ROOT / "11_dynamic_temporal_results_2017_2024/02_tables_and_diagnostics/temporal_summary_raster_diagnostics.csv"
        ),
        "rel_feature": pd.read_csv(
            ROOT / "02_model_performance_tables/cpec_2018_reliability_aware_feature_set_summary.csv"
        ),
    }


def with_serial(df):
    df = df.copy()
    if "No." in df.columns:
        return df
    df.insert(0, "No.", range(1, len(df) + 1))
    return df


def build_tables(data):
    factors = data["factors"]
    perf = data["perf"]
    stacked = data["stacked"]
    domain = data["domain"]
    road = data["road"]
    relroad = data["relroad"]
    annual = data["annual"]

    t1 = factors[["Group", "Readable label", "vif"]].copy()
    t1 = t1.rename(columns={"Group": "Predictor group", "Readable label": "Predictor name"})
    t1["VIF"] = t1["vif"].map(lambda x: f"{x:.2f}")
    t1 = t1.drop(columns=["vif"])

    selected_perf = []

    def add_perf(feature):
        model = "Spatial-CV Stacked Ensemble"
        pooled = stacked[(stacked["feature_set"] == feature) & (stacked["model"] == model)].iloc[0]
        selected_perf.append(
            {
                "Feature set": feature,
                "ROC-AUC": f"{pooled['roc_auc']:.4f}",
                "PR-AUC": f"{pooled['pr_auc_ap']:.4f}",
                "Balanced accuracy": f"{pooled['balanced_accuracy']:.4f}",
                "F1 score": f"{pooled['f1']:.4f}",
                "Brier score": f"{pooled['brier']:.4f}",
            }
        )

    add_perf("Conventional")
    add_perf("AlphaEarth Embeddings")
    add_perf("Conventional + AlphaEarth Embeddings")
    t2 = pd.DataFrame(selected_perf)

    t3 = domain[
        [
            "domain",
            "roc_auc_conventional_and_alphaearth_embeddings",
            "balanced_accuracy_conventional_and_alphaearth_embeddings",
            "fused_aoa_coverage_percent",
            "mean_fused_probability",
            "mean_fused_transfer_confidence",
            "uncertain_high_area_percent",
            "outside_fused_aoa_percent",
        ]
    ].copy()
    t3.columns = [
        "Subdomain",
        "Fused ROC-AUC",
        "Balanced accuracy",
        "Fused AoA (%)",
        "Mean probability",
        "Mean transfer confidence",
        "Uncertain high area (%)",
        "Outside fused AoA (%)",
    ]
    t3["Subdomain"] = t3["Subdomain"].replace(
        {
            "KP-AJK": "KPK-AJK",
            "Kashgar (Xinjiang, China)": "Kashgar",
            "Punjab-Sindh lowland corridor": "Punjab-Sindh",
        }
    )
    for col in t3.columns[1:]:
        t3[col] = t3[col].map(lambda x, c=col: f"{x:.2f}" if "%" in c else f"{x:.3f}")

    rr_all = relroad[(relroad["road_group"] == "All clipped roads") & (relroad["domain"] == "All domains")].iloc[0]
    rr_kkh = relroad[(relroad["road_group"].str.contains("KKH", regex=False)) & (relroad["domain"] == "All domains")].iloc[0]
    t4 = road.copy()
    t4["Road network"] = t4["road_group"]
    t4["Road network"] = t4["Road network"].replace(
        {"KKH proxy routes (N35 / 314)": "KKH proxy route (N35/314)"}
    )
    t4 = t4[
        [
            "Road network",
            "total_length_km",
            "weighted_length_probability_p80_km",
            "weighted_length_probability_p90_km",
            "mean_probability_length_weighted",
        ]
    ]
    t4["Reliable high weighted km"] = [
        rr_all["reliable_high_weighted_length_km"],
        rr_kkh["reliable_high_weighted_length_km"],
    ]
    t4["Uncertain high weighted km"] = [
        rr_all["uncertain_high_weighted_length_km"],
        rr_kkh["uncertain_high_weighted_length_km"],
    ]
    t4.columns = [
        "Road network",
        "Total length (km)",
        "P80-weighted exposure (km)",
        "P90-weighted exposure (km)",
        "Mean probability",
        "Reliable high exposure (km)",
        "Uncertain high exposure (km)",
    ]
    for c in t4.columns[1:]:
        t4[c] = t4[c].map(
            lambda x, col=c: f"{x:.3f}" if "Mean" in col else f"{x:,.1f}".replace(",", " ")
        )

    t5 = annual[["year", "mean", "p80", "p90", "std"]].copy().sort_values("year")

    def interp(y):
        if y in [2021, 2022]:
            return "highest annual susceptibility phase"
        if y in [2019, 2024]:
            return "lower annual susceptibility phase"
        if y == 2018:
            return "validated baseline year"
        return "intermediate dynamic condition"

    t5["Interpretation"] = t5["year"].map(interp)
    t5.columns = ["Year", "Mean probability", "P80", "P90", "Std. dev.", "Interpretation"]
    for c in ["Mean probability", "P80", "P90", "Std. dev."]:
        t5[c] = t5[c].map(lambda x: f"{x:.3f}")

    return {"T1": with_serial(t1), "T2": with_serial(t2), "T3": with_serial(t3), "T4": with_serial(t4), "T5": with_serial(t5)}


SECTIONS = [
    (
        "4.1. Integrity of the final factor database",
        [
            "The final 2018 modelling database contained 3316 labelled samples across the official China-Pakistan Economic Corridor (CPEC) study area, including Pakistan and Kashgar, Xinjiang. The feature-set comparison separated physically interpretable conditioning factors from annual foundation-model remote-sensing predictors. The Conventional set contained 19 conditioning factors, the AlphaEarth Embeddings set contained 64 annual embedding bands, and the fused Conventional + AlphaEarth Embeddings set contained 83 predictors. Hereafter, AlphaEarth Embeddings are referred to as AE embeddings when brevity is needed.",
            "The final Conventional database represented the main process controls expected across a transboundary mountain-to-lowland corridor: terrain geometry, hydro-climatic forcing, vegetation state, land cover, lithology, soil, seismicity, active structures, fluvial proximity, and infrastructure disturbance. Multicollinearity screening removed high-redundancy factor combinations from the candidate stacks. All retained Conventional variables had variance inflation factor (VIF) values below 4. The largest retained VIF values were monsoon rainfall total (3.88), maximum 1-day rainfall (3.66), NDVI median (2.81), and elevation (2.70). The final factor set therefore retained geomorphic and environmental completeness while limiting redundancy among terrain and climate variables.",
        ],
    ),
    (
        "4.2. Spatial-CV model performance and calibration",
        [
            "Spatial cross-validation (Spatial-CV) showed high discrimination for all three configurations of the stacked ensemble. The Conventional + AlphaEarth Embeddings ensemble produced the strongest performance, with ROC-AUC = 0.9718, PR-AUC = 0.9568, balanced accuracy = 0.9140, F1 score = 0.8969, and Brier score = 0.0603. The Conventional ensemble ranked second, whereas the AlphaEarth Embeddings ensemble ranked third. This ordering shows that AE embeddings were most effective when combined with physically interpretable conditioning factors rather than used as a stand-alone predictor set.",
            "The same ranking was reproduced in pooled out-of-fold evaluation (Table 2; n = 3316). The fused stacked ensemble achieved ROC-AUC = 0.9689, PR-AUC = 0.9612, balanced accuracy = 0.9219, F1 score = 0.9149, and Brier score = 0.0610. The Conventional stacked ensemble ranked second (ROC-AUC = 0.9607; PR-AUC = 0.9544), whereas the AlphaEarth Embeddings ensemble remained informative but lower (ROC-AUC = 0.9510; PR-AUC = 0.9410). The fused model also showed the most consistent balance among discrimination, precision-recall performance, threshold-dependent accuracy, and calibration (Fig. 1).",
        ],
    ),
    (
        "4.3. Spatial distribution of 2018 susceptibility",
        [
            "The fused 2018 susceptibility surface was spatially concentrated rather than uniformly high across the CPEC study area. The mean probability was 0.132, the median was 0.031, the 80th percentile was 0.126, and the 90th percentile was 0.442. This distribution indicates a right-skewed probability surface, with extensive low-probability terrain and a smaller set of coherent high-probability terrain units. The high-probability zones were concentrated in mountainous and structurally complex parts of the corridor, especially where high relief, incised valleys, tectonic structures, and road corridors overlap.",
            "The fused-minus-Conventional map shows where AE embeddings changed the 2018 probability surface (Fig. 2a). Positive values identify locations where annual satellite embedding information increased susceptibility relative to the Conventional model, whereas negative values identify locations where the fused model reduced the Conventional probability. The factor-level TreeSHAP beeswarm in Fig. 2b shows the corresponding sample-level contribution pattern for the fused model. Together, these panels show that the AE contribution was spatially structured rather than uniformly distributed across the CPEC study area.",
        ],
    ),
    (
        "4.4. Explainable landslide controls from TreeSHAP",
        [
            "TreeSHAP, a Shapley-value explanation method for tree-based models, was used to summarize global feature contributions. In the Conventional model, distance to roads was the most influential factor (mean absolute SHAP = 1.718), followed by elevation (1.039), valley depth (0.865), terrain ruggedness (0.774), profile curvature (0.478), distance to active faults (0.361), distance to rivers/streams (0.358), and NDVI median (0.254). These variables correspond to major landslide controls in the CPEC study area: road-cut disturbance, relief energy, valley incision, slope morphology, tectonic weakening, fluvial erosion, and vegetation condition.",
            "In the AlphaEarth-only model, the strongest embedding dimensions were A16, A55, and A44, with mean absolute SHAP values of 1.167, 0.524, and 0.470, respectively. These dimensions are not assigned direct geomorphic labels, but their contributions indicate that the annual embedding space captured remote-sensing patterns relevant to slope instability. In the fused model, distance to roads remained the top predictor (1.578), while AlphaEarth A16 (0.745), A55 (0.437), and A44 (0.262) entered the leading group alongside valley depth, profile curvature, elevation, terrain ruggedness, distance to rivers/streams, and distance to active faults (Fig. 3).",
        ],
    ),
    (
        "4.5. Subdomain transferability and area of applicability",
        [
            "Leave-one-domain-out testing evaluated transferability by holding out each CPEC subdomain in turn and predicting that omitted domain from the remaining subdomains. The fused model obtained ROC-AUC = 0.990 in Punjab-Sindh, ROC-AUC = 0.979 in Gilgit-Baltistan, ROC-AUC = 0.976 in KPK-AJK, ROC-AUC = 0.965 in Kashgar, and ROC-AUC = 0.913 in Balochistan (Table 3; Fig. 4). These differences show that predictive separability varied among geomorphic and administrative subdomains rather than being uniform across the full CPEC study area.",
            "Area of applicability (AoA) analysis identified where prediction pixels remained within the sampled predictor domain. The Conventional-only predictor space placed 34.4% of valid study-area pixels inside the AoA, indicating strong covariate-shift sensitivity when the model relied only on explicit physical predictors. AlphaEarth Embeddings increased AoA coverage to 95.8%, and the fused model increased it to 97.7%. At subdomain scale, fused AoA coverage was 98.9% in Balochistan, 98.9% in Kashgar, 97.8% in Gilgit-Baltistan, 98.2% in Punjab-Sindh, and 90.7% in KPK-AJK. The fused model therefore retained high spatial applicability across most CPEC subdomains while identifying KPK-AJK as the most extrapolation-sensitive subdomain.",
        ],
    ),
    (
        "4.6. Reliability-aware susceptibility and road exposure",
        [
            "Reliability-aware susceptibility separated high predicted probability from high-confidence high predicted probability. For the fused model, the mean study-area probability was 0.132 and the mean transfer confidence was 0.388. The mean reliability-weighted score was 0.0419. This score is not a probability; it is a conservative screening index that downweights high susceptibility where transfer confidence is weak.",
            "Subdomain reliability analysis showed that Balochistan had the highest mean fused probability (0.205), followed by KPK-AJK (0.194), Gilgit-Baltistan (0.184), Kashgar (0.084), and Punjab-Sindh (0.056). Balochistan also contained the largest reliable high-susceptibility area (1130 km²) and the largest uncertain high-probability area (98 491 km²). KPK-AJK had a large uncertain high-probability area (35 476 km²), consistent with its lower AoA coverage. These results separate locations with high susceptibility and high transfer confidence from locations where high probability requires additional field verification.",
            "Road exposure results translate the susceptibility surface into infrastructure decision relevance for the CPEC study area, with a specific focus on the Karakoram Highway (KKH). Across all clipped roads, 17 066 km of road length were assessed, with a mean length-weighted probability of 0.365. Probability-weighted exposure above the 80th and 90th percentile thresholds reached 10 620 km and 5 521 km, respectively. The KKH proxy route, represented by the N35/314 road identifiers, was more exposed: 1 448 km were assessed, the mean length-weighted probability was 0.748, and probability-weighted exposure above the 80th and 90th percentiles reached 1 396 km and 1 078 km. The top hotspot segments were concentrated along N35, with segment-level mean probabilities around 0.994 and reliability scores near 0.975 (Table 4; Fig. 5).",
        ],
    ),
    (
        "4.7. Dynamic susceptibility from 2017 to 2024",
        [
            "The temporal extension used the fixed 2018 stacked ensembles and annual predictors for 2017–2024. The resulting maps are interpreted as dynamic-covariate susceptibility maps, not as independent annual validation experiments.",
            "The fused annual susceptibility sequence showed clear interannual variation. Mean probability was 0.132 in 2017, 0.132 in 2018, declined to 0.113 in 2019, rose to 0.127 in 2020, peaked in 2021 (0.142), remained high in 2022 (0.140), and then declined in 2023 (0.124) and 2024 (0.118). The 90th percentile followed the same broad pattern, increasing from 0.298 in 2019 to 0.483 in 2021 and 0.462 in 2022. Therefore, 2021 and 2022 represent the highest-susceptibility phase in the current annual sequence, while 2019 and 2024 represent lower-susceptibility phases.",
            "The multi-year mean probability map had a study-area mean of 0.128, median of 0.035, and 90th percentile of 0.401. Temporal standard deviation was low for most pixels (median = 0.0039) but higher in the most dynamic areas (90th percentile = 0.0748). Persistent high susceptibility, defined as exceeding the 2018 fused P80 threshold in at least six of eight years, covered 18.1% of valid study-area pixels. The median year of maximum susceptibility was 2021, while the median year of minimum susceptibility was 2019. Year-to-year change maps showed the strongest mean increases during 2019–2020 (+0.0139) and 2020–2021 (+0.0156), and the strongest decreases during 2018–2019 (-0.0194) and 2022–2023 (-0.0162). The overall Theil-Sen trend was close to neutral at the study-area scale (mean slope = -0.00056 probability units yr⁻¹), indicating that the temporal signal is expressed mainly as persistent hotspots, episodic increases, and localized decreases rather than a uniform monotonic increase.",
        ],
    ),
]


CAPTIONS = {
    "R1": "Figure 1. Spatial-CV diagnostics for the three stacked-ensemble feature sets. ROC, precision-recall, and calibration curves are shown for Conventional, AlphaEarth Embeddings, and Conventional + AlphaEarth Embeddings.",
    "R2": "Figure 2. AlphaEarth added value and TreeSHAP explanation for the fused stacked ensemble. Panel (a) maps the difference between fused and Conventional probabilities; positive values indicate pixels where AlphaEarth embeddings increased predicted susceptibility. Panel (b) shows factor-level TreeSHAP values for the fused model; point color indicates feature value.",
    "R3": "Figure 3. Global TreeSHAP interpretation of the fused stacked ensemble. The fused model retained physically interpretable controls such as distance to roads, valley depth, elevation, terrain ruggedness, rivers/streams, and active faults while also using leading AlphaEarth dimensions.",
    "R4": "Figure 4. Domain-specific fused susceptibility and transferability diagnostics. Panels (a)–(e) show 2018 fused stacked-ensemble probability maps for Kashgar, Gilgit-Baltistan, KPK-AJK, Balochistan, and Punjab-Sindh. Panel (f) summarizes leave-one-domain-out ROC-AUC, AoA coverage, mean transfer confidence, and uncertain high-area share.",
    "R5": "Figure 5. Reliability-aware road exposure and Karakoram Highway (KKH) hotspot outputs. The KKH proxy route is represented by N35/314, and the top-ranked hotspot segments are concentrated along N35.",
}


FIG_PATHS = {
    "R1": FIGDIR / "figure_R1_model_validation_curves_grid.png",
    "R2": FIGDIR / "figure_R2_added_value_and_beeswarm_pubready.png",
    "R3": FIGDIR / "figure_R2_fused_shap_explainability_grid.png",
    "R4": FIGDIR / "figure_R4_domain_specific_probability_panels_pubready.png",
    "R5": ROOT / "03_figures/07_reliability_aware_final_outputs/figure_cpec_2018_reliability_aware_road_hotspots.png",
    "R6a": ROOT / "11_dynamic_temporal_results_2017_2024/01_publication_figures/figure_temporal_summary_fused_2017_2024.png",
    "R6b": ROOT / "11_dynamic_temporal_results_2017_2024/01_publication_figures/figure_year_to_year_fused_probability_change_2017_2024.png",
}


def df_to_md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        out.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(out)


def write_markdown(tables):
    md = ["# 4. Results", ""]
    for title, paras in SECTIONS:
        md.append(f"## {title}")
        md.append("")
        for paragraph in paras:
            md.append(paragraph)
            md.append("")
        if title.startswith("4.1"):
            md.extend(["**Table 1. Final Conventional conditioning factors and VIF values.**", df_to_md(tables["T1"]), ""])
        if title.startswith("4.2"):
            md.extend(["**Table 2. Pooled out-of-fold performance of the Spatial-CV Stacked Ensemble for the three feature sets (n = 3316).**", df_to_md(tables["T2"]), "", f"![Figure 1]({FIG_PATHS['R1']})", CAPTIONS["R1"], ""])
        if title.startswith("4.3"):
            md.extend(
                [
                    f"![Figure 2]({FIG_PATHS['R2']})",
                    CAPTIONS["R2"],
                    "",
                ]
            )
        if title.startswith("4.4"):
            md.extend([f"![Figure 3]({FIG_PATHS['R3']})", CAPTIONS["R3"], ""])
        if title.startswith("4.5"):
            md.extend(["**Table 3. Fused Spatial-CV Stacked Ensemble subdomain transferability and area-of-applicability diagnostics.**", df_to_md(tables["T3"]), "", f"![Figure 4]({FIG_PATHS['R4']})", CAPTIONS["R4"], ""])
        if title.startswith("4.6"):
            md.extend(["**Table 4. CPEC study-area road exposure and KKH corridor exposure summary.**", df_to_md(tables["T4"]), "", f"![Figure 5]({FIG_PATHS['R5']})", CAPTIONS["R5"], ""])
        if title.startswith("4.7"):
            md.extend(
                [
                    "**Table 5. Annual fused susceptibility summary, 2017–2024.**",
                    df_to_md(tables["T5"]),
                    "",
                    f"![Figure 6a]({FIG_PATHS['R6a']})",
                    f"![Figure 6b]({FIG_PATHS['R6b']})",
                    "Figure 6. Dynamic fused susceptibility products for 2017–2024.",
                    "",
                ]
            )
    OUT_MD.write_text("\n".join(md), encoding="utf-8")


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_cell_text(cell, text, bold=False, size=8.0, align=None):
    cell.text = ""
    paragraph = cell.paragraphs[0]
    if align is not None:
        paragraph.alignment = align
    run = paragraph.add_run(str(text))
    run.bold = bold
    run.font.name = "Arial"
    run.font.size = Pt(size)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def add_table_doc(doc, df, title, note=None, size=7.6):
    paragraph = doc.add_paragraph()
    run = paragraph.add_run(title)
    run.bold = True
    run.font.name = "Arial"
    run.font.size = Pt(9.5)

    table = doc.add_table(rows=1, cols=len(df.columns))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    for idx, column in enumerate(df.columns):
        set_cell_text(table.rows[0].cells[idx], column, True, size=size, align=WD_ALIGN_PARAGRAPH.CENTER)
        set_cell_shading(table.rows[0].cells[idx], "D9EAF7")

    for _, row in df.iterrows():
        cells = table.add_row().cells
        for idx, column in enumerate(df.columns):
            value = row[column]
            align = WD_ALIGN_PARAGRAPH.CENTER if idx > 0 and len(str(value)) < 16 else WD_ALIGN_PARAGRAPH.LEFT
            set_cell_text(cells[idx], value, False, size=size, align=align)

    if note:
        paragraph = doc.add_paragraph()
        run = paragraph.add_run(note)
        run.italic = True
        run.font.size = Pt(8.5)
        run.font.name = "Arial"
    doc.add_paragraph()


def add_figure_doc(doc, path, caption, width=6.25):
    path = Path(path)
    if not path.exists():
        doc.add_paragraph(f"[Missing figure: {path}]")
        return
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.add_run().add_picture(str(path), width=Inches(width))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = cap.add_run(caption)
    run.bold = True
    run.font.name = "Arial"
    run.font.size = Pt(9)
    cap.paragraph_format.space_after = Pt(6)


def add_body_paragraph(doc, text):
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(5)
    paragraph.paragraph_format.line_spacing = 1.06
    run = paragraph.add_run(text)
    run.font.name = "Arial"
    run.font.size = Pt(10.5)


def write_docx(tables):
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(0.75)
    section.bottom_margin = Inches(0.75)
    section.left_margin = Inches(0.75)
    section.right_margin = Inches(0.75)

    styles = doc.styles
    styles["Normal"].font.name = "Arial"
    styles["Normal"].font.size = Pt(10.5)
    styles["Normal"]._element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")
    for style_name, size, color in [("Title", 18, "1F4E79"), ("Heading 1", 14, "1F4E79"), ("Heading 2", 12, "1F4E79")]:
        style = styles[style_name]
        style.font.name = "Arial"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor.from_string(color)

    header = section.header.paragraphs[0]
    header.text = "CPEC landslide susceptibility | Results section draft for Geoscience Frontiers"
    header.runs[0].font.size = Pt(8)
    header.runs[0].font.color.rgb = RGBColor(90, 90, 90)
    footer = section.footer.paragraphs[0]
    footer.text = "Manuscript results draft | 2026-05-15"
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    footer.runs[0].font.size = Pt(8)
    footer.runs[0].font.color.rgb = RGBColor(90, 90, 90)

    doc.add_paragraph("4. Results", style="Title")

    for title, paragraphs in SECTIONS:
        doc.add_heading(title, level=1)
        for paragraph in paragraphs:
            add_body_paragraph(doc, paragraph)
        if title.startswith("4.1"):
            add_table_doc(
                doc,
                tables["T1"],
                "Table 1. Final Conventional conditioning factors and multicollinearity diagnostics.",
                size=7.5,
            )
        if title.startswith("4.2"):
            add_table_doc(
                doc,
                tables["T2"],
                "Table 2. Pooled out-of-fold performance of the Spatial-CV Stacked Ensemble for the three feature sets (n = 3316).",
                size=6.8,
            )
            add_figure_doc(doc, FIG_PATHS["R1"], CAPTIONS["R1"], width=6.35)
        if title.startswith("4.3"):
            add_figure_doc(doc, FIG_PATHS["R2"], CAPTIONS["R2"], width=6.75)
        if title.startswith("4.4"):
            add_figure_doc(doc, FIG_PATHS["R3"], CAPTIONS["R3"], width=6.35)
        if title.startswith("4.5"):
            doc.add_page_break()
            add_table_doc(
                doc,
                tables["T3"],
                "Table 3. Fused Spatial-CV Stacked Ensemble transferability, area-of-applicability (AoA), and reliability diagnostics by CPEC subdomain.",
                size=6.4,
            )
            add_figure_doc(doc, FIG_PATHS["R4"], CAPTIONS["R4"], width=6.75)
        if title.startswith("4.6"):
            add_table_doc(
                doc,
                tables["T4"],
                "Table 4. CPEC study-area road exposure and KKH corridor exposure summary from the fused 2018 susceptibility surface.",
                size=7.0,
            )
            add_figure_doc(doc, FIG_PATHS["R5"], CAPTIONS["R5"], width=6.2)
        if title.startswith("4.7"):
            add_table_doc(
                doc,
                tables["T5"],
                "Table 5. Annual fused susceptibility summary for the fixed-model dynamic assessment, 2017–2024.",
                size=7.0,
            )
            add_figure_doc(
                doc,
                FIG_PATHS["R6a"],
                "Figure 6a. Multi-year fused temporal summary products for 2017–2024, including mean probability, temporal variability, trend, persistent high-susceptibility zones, and years of maximum/minimum susceptibility.",
                width=5.6,
            )
            add_figure_doc(
                doc,
                FIG_PATHS["R6b"],
                "Figure 6b. Year-to-year fused probability change maps. The strongest mean increases occurred during 2019–2020 and 2020–2021, while the strongest decreases occurred during 2018–2019 and 2022–2023.",
                width=5.8,
            )

    doc.save(OUT_DOCX)


def write_audit():
    AUDIT_MD.write_text(
        """# Geoscience Frontiers results-section scope and style audit

This audit was prepared alongside the assembled Results section.

## Section-level checks

| Results subsection | Journal-style purpose | Audit decision |
| --- | --- | --- |
| 4.1 Factor database and VIF | Establish data quality and geoscientific factor completeness before modelling | Kept. Claims tied to final factor table and VIF values. |
| 4.2 Spatial-CV performance | Present robust model evidence without relying on random-split inflation | Kept. Spatial-CV, PR-AUC, calibration, F1, balanced accuracy, and Brier are all reported. |
| 4.3 Spatial susceptibility pattern | Convert model scores into geoscientific spatial interpretation | Kept. Focuses on mountain/corridor concentration and skewed probability distribution. |
| 4.4 TreeSHAP controls | Show physical plausibility and explain foundation-model contribution | Kept. Avoids claiming physical meanings for individual AlphaEarth bands. |
| 4.5 Transferability and AoA | Address reviewer concerns about extrapolation across CPEC subdomains | Kept. AoA and leave-one-domain-out results are separated from ordinary accuracy. |
| 4.6 Reliability and road exposure | Link CPEC study-area hazard modelling to KKH corridor decision relevance | Kept. Clearly separates probability from reliability-weighted screening score. |
| 4.7 Dynamic susceptibility | Present temporal added value without overclaiming annual retraining | Kept. States fixed 2018 model and dynamic-covariate interpretation. |

## Sentence-level editorial rules applied

- Removed unsupported claims of causality.
- Removed the separate Results-section innovation statement and integrated its useful synthesis into the results narrative.
- Kept only Spatial-CV Stacked Ensemble model results in the manuscript-facing tables and text.
- Added serial-number columns to all manuscript tables.
- Removed drafting notes below tables; essential definitions were moved into captions or section text.
- Avoided calling reliability-weighted scores probabilities.
- Avoided saying susceptibility increased everywhere; the temporal result is described as persistent and spatially heterogeneous.
- Kept AlphaEarth interpretation at embedding/context level instead of assigning false physical labels to bands.
- Used exact values from project CSV outputs wherever a numerical claim is made.
- Standardized terminology: CPEC study area is used for the full mapping region, and KKH corridor is used only for road-exposure interpretation.

## Journal sources checked

- Geoscience Frontiers Guide for Authors: https://www.sciencedirect.com/journal/geoscience-frontiers/publish/guide-for-authors
- Geoscience Frontiers aims and scope: https://geosciencefrontiers.com/as
- Recent Geoscience Frontiers landslide susceptibility paper using SHAP, AUC, uncertainty-aware mapping, and reliability framing: https://www.sciencedirect.com/science/article/pii/S1674987125001938
""",
        encoding="utf-8",
    )


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    data = load_inputs()
    tables = build_tables(data)
    write_markdown(tables)
    write_docx(tables)
    write_audit()
    print(OUT_DOCX)
    print(OUT_MD)
    print(AUDIT_MD)


if __name__ == "__main__":
    main()
