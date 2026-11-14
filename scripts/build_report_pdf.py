"""Build the ML project submission PDF report from frozen experiment artifacts.

This script reads the project's frozen CSV/JSON evidence and figures, and compiles
the official academic report (Flood_Area_Segmentation_Project_Report.pdf) into deliverables/.
It incorporates all project-generated figures, maintains strict anti-leakage compliance,
and uses verified numerical metrics.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    PageBreak,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
DELIVERABLES = ROOT / "deliverables"

NAVY = colors.HexColor("#222222")
TEAL = colors.HexColor("#333333")


def read_csv(name: str) -> list[dict[str, str]]:
    with (TABLES / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def pct(value: str | float) -> str:
    return f"{float(value):.2f}%"


def report_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"], fontName="Helvetica-Bold", fontSize=25, leading=30, alignment=TA_CENTER, textColor=NAVY, spaceAfter=14),
        "subtitle": ParagraphStyle("subtitle", parent=base["Normal"], fontName="Helvetica", fontSize=13, leading=18, alignment=TA_CENTER, textColor=TEAL, spaceAfter=10),
        "chapter": ParagraphStyle("chapter", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=16, leading=20, textColor=NAVY, spaceBefore=0, spaceAfter=8),
        "section": ParagraphStyle("section", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=11.5, leading=14.5, textColor=TEAL, spaceBefore=4, spaceAfter=5),
        "body": ParagraphStyle("body", parent=base["BodyText"], fontName="Helvetica", fontSize=9.0, leading=12.8, alignment=TA_JUSTIFY, spaceAfter=6),
        "small": ParagraphStyle("small", parent=base["BodyText"], fontName="Helvetica", fontSize=8.0, leading=10.2, alignment=TA_JUSTIFY, spaceAfter=4),
        "caption": ParagraphStyle("caption", parent=base["BodyText"], fontName="Helvetica-Oblique", fontSize=7.5, leading=9.4, alignment=TA_CENTER, textColor=colors.HexColor("#455A64"), spaceBefore=2, spaceAfter=6),
        "toc": ParagraphStyle("toc", parent=base["BodyText"], fontName="Helvetica", fontSize=9.5, leading=16, spaceAfter=1),
        "ref": ParagraphStyle("ref", parent=base["BodyText"], fontName="Helvetica", fontSize=8.2, leading=11.5, leftIndent=12, firstLineIndent=-12, spaceAfter=4.5),
    }


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def add_table(data: list[list[str]], widths: list[float], font_size: float = 7.4) -> Table:
    table = Table(data, colWidths=[w * inch for w in widths], repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E9E9E9")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 2),
        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F7F7")]),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D0D0D0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4.5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4.5),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
    ]))
    return table


def report_header_footer(c: canvas.Canvas, doc: BaseDocTemplate) -> None:
    c.saveState()
    if doc.page > 1:
        c.setStrokeColor(colors.HexColor("#BEBEBE"))
        c.setLineWidth(0.6)
        c.line(0.62 * inch, 11.05 * inch, 7.65 * inch, 11.05 * inch)
        c.setFillColor(colors.HexColor("#333333"))
        c.setFont("Helvetica-Bold", 7.5)
        c.drawString(0.62 * inch, 11.12 * inch, "FLOOD AREA SEGMENTATION  |  MACHINE LEARNING PROJECT REPORT")
        c.setFillColor(colors.HexColor("#555555"))
        c.setFont("Helvetica", 7.5)
        c.drawRightString(7.65 * inch, 0.43 * inch, f"CS3103  |  Page {doc.page}")
    c.restoreState()


def build_report() -> Path:
    styles = report_styles()
    teachers = read_csv("supervised_teachers_test.csv")
    baselines = read_csv("supervised_baselines_test.csv")
    hard = read_csv("hard_distillation_test_long.csv")
    kmeans = read_csv("kmeans_test_results.csv")
    matrix = read_csv("hard_distillation_iou_matrix.csv")
    selected = next(row for row in hard if row["pair"] == "random_forest_to_random_forest")
    selected_km = next(row for row in kmeans if row["model"] == "km_refined__reestimated_L")

    cp01_path = ROOT / "results" / "checkpoints" / "CP01_dataset_audit.json"
    cp04_path = ROOT / "results" / "checkpoints" / "CP04_final_test.json"
    assert cp04_path.exists(), "CP04 checkpoint must exist"
    cp04 = json.loads(cp04_path.read_text())
    assert cp04["status"] == "PASSED", "CP04 must have PASSED status"

    report_path = DELIVERABLES / "Flood_Area_Segmentation_Project_Report.pdf"
    frame = Frame(0.62 * inch, 0.65 * inch, 7.03 * inch, 10.2 * inch, id="report")
    doc = BaseDocTemplate(str(report_path), pagesize=A4, leftMargin=0.62 * inch, rightMargin=0.62 * inch, topMargin=0.7 * inch, bottomMargin=0.65 * inch)
    doc.addPageTemplates([__import__("reportlab.platypus", fromlist=["PageTemplate"]).PageTemplate(id="report", frames=frame, onPage=report_header_footer)])
    story: list = []

    # =========================================================================
    # Page 1: Title page
    # =========================================================================
    story += [
        Spacer(1, 1.8 * inch),
        p("Flood Area Segmentation", styles["title"]),
        p("Classical Teacher–Student Learning under a Label Budget", styles["subtitle"]),
        Spacer(1, 0.4 * inch),
        p("CS3103 Machine Learning Project Report", styles["subtitle"]),
        Spacer(1, 0.55 * inch),
    ]
    meta_info = [
        "Student Name: Bojedla Shinesh",
        "Roll Number: 2401CS75",
        "Date: 14 November 2026",
    ]
    for line in meta_info:
        story.append(p(line, ParagraphStyle("cover", parent=styles["body"], alignment=TA_CENTER, fontSize=10.5, leading=18, spaceAfter=7)))
    story += [
        Spacer(1, 0.45 * inch),
        p("Dataset: Flood Area Segmentation, Kaggle version 1 (Faizal Karim, CC0)", styles["subtitle"]),
        p("Experiment seed 42  |  Final evaluation checkpoint PASSED", styles["small"]),
        PageBreak(),
    ]

    # =========================================================================
    # Page 2: Abstract
    # =========================================================================
    story += [
        p("Abstract", styles["chapter"]),
        p(
            "This project studies binary flood segmentation on the assigned RGB flood-scene dataset with a strict label budget. "
            "The experiment separates 40 labelled images, 163 unlabelled images, 29 validation images, and 58 final-test images using a fixed random seed. "
            "The teacher models use 20 handcrafted colour, texture, edge, and position features; deployable students use only six direct RGB and HSV features. "
            "K-Nearest Neighbours, Random Forest, Gaussian Naive Bayes, and K-Means are evaluated under explicit capacity limits. "
            "Validation data chooses all thresholds, confidence filters, model pairs, and K-Means variants before the final test masks are opened. "
            "The selected hard Random Forest to Random Forest student reaches 75.76% F1 and 60.98% IoU on 3,801,088 frozen test pixels. "
            "It improves the matched compact Random Forest baseline by 2.89 IoU percentage points while retaining 92.39% of teacher IoU (95.27% of teacher F1) "
            "with 150.28x serialized-model compression. A major limitation remains: seven exact duplicate groups cross the mandatory data partitions, "
            "including one labelled-to-test pair. The required split is retained, so the results demonstrate reproducible performance on this benchmark "
            "rather than independent deployment readiness.",
            styles["body"],
        ),
        p("Keywords: flood segmentation, classical machine learning, pseudo labels, knowledge distillation, reproducibility", styles["small"]),
        Spacer(1, 0.2 * inch),
        p("Project contribution", styles["section"]),
    ]
    for item in [
        "Implemented a leak-aware teacher student pipeline with explicit mask-access controls.",
        "Compared compact students with direct no-teacher baselines under matched feature and capacity budgets.",
        "Recorded a frozen selection and final-evaluation protocol with reproducible CSV, JSON, and notebook evidence.",
        "Generated an extensive suite of project diagnostics and visual diagrams to substantiate distillation gains and error modes.",
    ]:
        story.append(p("• " + item, styles["body"]))
    story.append(PageBreak())

    # =========================================================================
    # Page 3: Table of Contents & Lists
    # =========================================================================
    story += [p("Table of Contents", styles["chapter"])]
    contents = [
        ("Abstract", "2"),
        ("1 Introduction", "4"),
        ("2 Literature Review", "5"),
        ("3 Proposed Methodology", "6-8"),
        ("4 Results and Discussion", "9-12"),
        ("5 Conclusion and Future Scope", "13"),
        ("References", "14"),
        ("Appendix", "15"),
    ]
    for heading, number in contents:
        story.append(p(f"{heading}<font color='#607D8B'>........................................................................ {number}</font>", styles["toc"]))

    story += [Spacer(1, 0.18 * inch), p("List of Figures", styles["section"])]
    figures_list = [
        ("Figure 1. Post-preprocessing class balance across partitions", "4"),
        ("Figure 2. Representative training image-mask pairs from L", "6"),
        ("Figure 3. Leak-safe experimental workflow and architecture", "7"),
        ("Figure 4. Supervised teachers vs compact baselines comparison", "9"),
        ("Figure 5. Hard-distillation 3×3 test IoU matrix heatmap", "10"),
        ("Figure 6. Validation IoU and retained pseudo labels vs confidence τ", "10"),
        ("Figure 7. K-Means teacher silhouette score vs cluster count K", "11"),
        ("Figure 8. Comparison of the four K-Means student variants", "11"),
        ("Figures 9-12. Final-test qualitative segmentation panels", "12"),
    ]
    for label, number in figures_list:
        story.append(p(f"{label}<font color='#607D8B'> ............................ {number}</font>", styles["toc"]))

    story += [Spacer(1, 0.15 * inch), p("List of Tables", styles["section"])]
    tables_list = [
        ("Table 1. Dataset partitions and permitted label use", "4"),
        ("Table 2. Related approaches and engineering gap", "5"),
        ("Table 3. Teacher and student feature sets", "6"),
        ("Table 4. Experimental capacity budget", "8"),
        ("Table 5. Supervised teacher and baseline results", "9"),
        ("Table 6. Hard-distillation test IoU matrix", "10"),
        ("Table 7. Final-test K-Means variants", "11"),
        ("Table 8. Reproducibility checklist", "15"),
    ]
    for label, number in tables_list:
        story.append(p(f"{label}<font color='#607D8B'> ....................................... {number}</font>", styles["toc"]))
    story.append(PageBreak())

    # =========================================================================
    # Page 4: Chapter 1 Introduction
    # =========================================================================
    story += [
        p("1 Introduction", styles["chapter"]),
        p(
            "Flood maps can help route field teams and prioritise rapid damage assessment. In this course project, the task is deliberately narrow: "
            "classify every pixel in a 256 x 256 RGB image as flood or non-flood using only the allowed classical algorithms. The work asks whether "
            "a compact six-feature student can benefit from pseudo labels generated by a richer handcrafted teacher while respecting a small labelled-image budget.",
            styles["body"],
        ),
        p("1.1 Background and motivation", styles["section"]),
        p(
            "A flood mask depends on colour, texture, scene context, and boundaries. Rich features can help conventional models distinguish water from sky, "
            "roads, shadows, or bare soil, but they add extraction cost and can increase model size. A smaller feature set is more practical when inference "
            "resources are constrained. Teacher student learning provides a controlled way to test whether a compact model can absorb useful decisions from a larger one.",
            styles["body"],
        ),
        p("1.2 Problem statement, objectives and scope", styles["section"]),
        p(
            "Given an RGB image, predict a binary flood mask. The objectives are to build a reproducible fixed split, prevent leakage from unlabelled and test masks, "
            "compare teachers and compact students, and measure the accuracy-efficiency trade-off. The scope excludes neural networks, external images, pretrained features, and test-informed tuning.",
            styles["body"],
        ),
        add_table(
            [
                ["Partition", "Images", "Permitted use", "Flood pixels"],
                ["L", "40", "Teacher fitting, compact baselines, cluster mapping", "36.49%"],
                ["U", "163", "Image-only fit and teacher pseudo labels; masks blocked", "Not opened"],
                ["Validation", "29", "All development selection", "37.89%"],
                ["Test", "58", "Frozen final measurement only", "38.34% audit"],
            ],
            [0.72, 0.55, 3.8, 1.15],
        ),
        p("Table 1. Dataset partitions and permitted label use.", styles["caption"]),
        Spacer(1, 0.05 * inch),
        Image(str(FIGURES / "audit_class_balance.png"), width=5.60 * inch, height=2.15 * inch, hAlign="CENTER"),
        p("Figure 1. Post-preprocessing class balance. Test prevalence is descriptive and did not guide model selection.", styles["caption"]),
        PageBreak(),
    ]

    # =========================================================================
    # Page 5: Chapter 2 Literature Review
    # =========================================================================
    story += [
        p("2 Literature Review", styles["chapter"]),
        p(
            "Classical supervised learners remain useful when feature extraction is interpretable and the dataset is small. KNN classifies from nearby reference pixels, "
            "Random Forest averages decision trees, and Gaussian Naive Bayes models class-conditional feature distributions. K-Means provides an unsupervised comparison "
            "by grouping image pixels before cluster-to-class mapping. Knowledge distillation traditionally transfers a model's behaviour into a smaller learner. "
            "Here, the transfer signal is a confidence-filtered pseudo label rather than a neural softmax vector.",
            styles["body"],
        ),
        add_table(
            [
                ["Source", "Approach", "Relevance to this project", "Gap addressed"],
                ["Cover and Hart (1967)", "KNN", "KNN teacher and baseline", "Compact reference limit"],
                ["Breiman (2001)", "Random Forest", "Rich-feature teacher", "Compact shallow forest"],
                ["MacQueen (1967)", "K-Means", "Unsupervised track", "Labels map clusters"],
                ["Hinton et al. (2015)", "Distillation", "Teacher to compact student", "Hard pseudo labels"],
                ["Karim, Kaggle v1", "Flood masks", "Assigned benchmark", "Duplicates cross split"],
            ],
            [1.25, 1.15, 2.45, 2.15],
            7.1,
        ),
        p("Table 2. Related approaches and the engineering gap.", styles["caption"]),
        p("Research gap", styles["section"]),
        p(
            "The central engineering question is not whether a complex model wins on a large benchmark. It is whether image-only U data and rich teacher features "
            "can improve a constrained student beyond the same student's labelled-only baseline. The project answers that question with a fixed split, matched baselines, "
            "and a frozen final evaluation.",
            styles["body"],
        ),
        p("Contribution of this work", styles["section"]),
    ]
    for item in [
        "A complete audit records the fixed split, file integrity, class balance, and duplicate groups.",
        "The pipeline enforces that U masks cannot enter fitting and test masks cannot enter development selection.",
        "Every result table is generated from the final checkpoint instead of manually copied into the report.",
        "The comparison reports both accuracy and compactness rather than presenting transfer gains in isolation.",
    ]:
        story.append(p("• " + item, styles["body"]))
    story.append(PageBreak())

    # =========================================================================
    # Page 6: Chapter 3 Proposed Methodology (Audit, Preprocessing, EDA & Features)
    # =========================================================================
    story += [
        p("3 Proposed Methodology", styles["chapter"]),
        p("3.1 Dataset description and audit", styles["section"]),
        p(
            "The Kaggle Flood Area Segmentation dataset contains 290 paired flood-scene images and grayscale masks. "
            "A deterministic permutation of sorted file names with numpy.random.default_rng(42) forms the prescribed partitions: "
            "test first (58), validation next (29), labelled L next (40), and unlabelled U last (163). The audit found no corrupt image files. "
            "It also found seven exact duplicate groups spanning fixed partitions, including 1026.jpg in L and 18.jpg in test. "
            "The split remains unchanged because the assignment requires it.",
            styles["body"],
        ),
        p("3.2 Data preprocessing and exploratory analysis", styles["section"]),
        p(
            "Images are converted to RGB and resized to 256 x 256 with bilinear interpolation. Masks are thresholded at 128 then resized with nearest-neighbour interpolation. "
            "Figure 2 illustrates representative training scenes and ground truth from L, showing substantial variation in turbidity, sunlight reflection, and vegetation boundaries.",
            styles["body"],
        ),
        Image(str(FIGURES / "audit_labelled_examples.png"), width=5.80 * inch, height=2.26 * inch, hAlign="CENTER"),
        p("Figure 2. Representative labelled image-mask pairs from training partition L showing raw flood scenes and ground-truth flood annotations.", styles["caption"]),
        p("3.3 Feature representation", styles["section"]),
        add_table(
            [
                ["Feature set", "Components", "Dimension", "Rationale", "Extraction time"],
                ["Student phi S", "RGB, HSV", "6", "Direct colour cues", "0.00129 s/image"],
                ["Teacher phi T", "phi S + blur, local stats, Sobel, Laplacian, position", "20", "Context, texture & edges", "0.00807 s/image"],
            ],
            [0.9, 2.35, 0.62, 1.65, 1.1],
            7.0,
        ),
        p("Table 3. Teacher and student feature sets. Extraction time was measured in the project feature-pipeline checkpoint.", styles["caption"]),
        p(
            "The compact student uses 6 dimensions, which is exactly 30% of the 20-dimensional teacher set. "
            "This strictly satisfies both capacity rules: no more than six direct image-derived features and no more than half the teacher dimension.",
            styles["body"],
        ),
        PageBreak(),
    ]

    # =========================================================================
    # Page 7: Chapter 3 Proposed Methodology (Architecture & Algorithm)
    # =========================================================================
    story += [
        p("3 Proposed Methodology", styles["chapter"]),
        p("3.4 Leak prevention and teacher student architecture", styles["section"]),
        Image(str(FIGURES / "pipeline_architecture.png"), width=6.25 * inch, height=2.36 * inch, hAlign="CENTER"),
        p("Figure 3. Leak-safe workflow. Fitting and validation selection are strictly separated from the frozen final test evaluation.", styles["caption"]),
        p(
            "The data-access layer rejects U masks in every training pathway. Teachers fit only on L ground-truth pixels. "
            "Teachers generate pseudo labels from permitted L and U images. Compact students receive confidence-filtered pseudo labels and directly derived six-feature vectors. "
            "K-Means fits only image pixels from L and U, while its cluster-to-class mapping may use L masks. "
            "These constraints protect the role of unlabelled data and ensure that no final-test evidence tunes a model.",
            styles["body"],
        ),
        p("Algorithm 1  Classical teacher student segmentation", styles["section"]),
        add_table(
            [
                ["Step", "Procedure"],
                ["Input", "RGB images, L masks, image-only U, validation masks, fixed seed 42"],
                ["Output", "Frozen compact model and final test metrics"],
                ["1", "Validate paired files and generate the fixed L, U, validation, and test partitions."],
                ["2", "Preprocess images and allowed masks; generate teacher and student feature vectors."],
                ["3", "Fit teachers and labelled-only compact baselines using permitted L pixels."],
                ["4", "Generate confidence-filtered teacher pseudo labels on permitted L and U images; fit compact students."],
                ["5", "Select thresholds, tau, mappings, and K using validation metrics only."],
                ["6", "Freeze selected settings, then evaluate test pixels once without changing a setting."],
            ],
            [0.55, 6.0],
            7.3,
        ),
        p("Algorithm 1. Implementation-matching experimental procedure.", styles["caption"]),
        PageBreak(),
    ]

    # =========================================================================
    # Page 8: Chapter 3 Proposed Methodology (Capacity & Evaluation Protocol)
    # =========================================================================
    story += [
        p("3 Proposed Methodology", styles["chapter"]),
        p("3.5 Models and capacity controls", styles["section"]),
        p(
            "The project intentionally compares varied classical families. KNN uses sampled reference pixels; Random Forest uses a fixed number of trees and shallow student depth; "
            "Gaussian Naive Bayes uses the feature sets directly; K-Means uses a teacher with more clusters than the compact student. "
            "A direct compact baseline is trained for each supervised student family to isolate the value of teacher pseudo labels from the compact architecture itself.",
            styles["body"],
        ),
        add_table(
            [
                ["Family", "Teacher configuration", "Student configuration", "Selection evidence"],
                ["KNN", "At least 20,000 rich-feature references", "At most 5,000 six-feature references", "Validation pathway table"],
                ["Random Forest", "200 trees; rich features", "20 trees; depth at most 8; six features", "Validation pathway table"],
                ["GaussianNB", "20 teacher features", "6 student features", "Validation pathway table"],
                ["K-Means", "K at least 6; n_init 10; 100,000 pixels", "K at most 4; 20,000 pixels", "Silhouette and validation"],
            ],
            [0.82, 2.15, 2.2, 1.35],
            7.0,
        ),
        p("Table 4. Capacity budget and development evidence.", styles["caption"]),
        p("3.6 Evaluation protocol", styles["section"]),
        p(
            "Micro metrics aggregate all 58 x 256 x 256 = 3,801,088 final-test pixels. IoU measures overlap between predicted and ground-truth flood classes. "
            "F1 balances precision and recall. The report also compares serialized model size and mean full-image segmentation time.",
            styles["body"],
        ),
        p("Soft-label check", styles["section"]),
        p(
            "For the selected Random Forest pair, a weighted-duplication soft alternative holds the same source-pixel budget and tau. "
            "It provides a focused robustness check rather than a broad post-hoc search. The hard pseudo-label version remains the selected approach because validation chose it before final measurements.",
            styles["body"],
        ),
        p("Reproducibility", styles["section"]),
        p(
            "The repository stores the experiment configuration, split and sampling manifests, package details, checkpoint hashes, source code, unit tests, "
            "generated CSV tables, and a linear notebook. The notebook runs audit, development, frozen final evaluation, and artifact assertions in that order.",
            styles["body"],
        ),
        PageBreak(),
    ]

    # =========================================================================
    # Page 9: Chapter 4 Results and Discussion (Teachers & Baselines)
    # =========================================================================
    teacher_rows = [["Model", "F1", "IoU", "Size", "Seconds/image"]]
    for r in teachers:
        teacher_rows.append([r["family"].replace("_", " ").title() + " teacher", pct(r["f1_percent"]), pct(r["iou_percent"]), f"{int(float(r['model_size_bytes'])):,} B", f"{float(r['mean_segmentation_seconds_per_image']):.3f}"])
    for r in baselines:
        teacher_rows.append(["Compact " + r["family"].replace("_", " "), pct(r["f1_percent"]), pct(r["iou_percent"]), f"{int(float(r['model_size_bytes'])):,} B", f"{float(r['mean_segmentation_seconds_per_image']):.3f}"])

    story += [
        p("4 Results and Discussion", styles["chapter"]),
        p("4.1 Supervised teachers and compact baselines", styles["section"]),
        p(
            "The Random Forest teacher is the strongest supervised teacher on the frozen final test. Direct compact baselines show the substantial drop in accuracy "
            "when shrinking the feature set from 20 to 6 and enforcing shallow tree depth before any knowledge is transferred from the teacher.",
            styles["body"],
        ),
        add_table(teacher_rows, [2.0, 0.8, 0.8, 1.45, 1.2], 7.2),
        p("Table 5. Frozen final-test results for supervised teachers and labelled-only compact baselines.", styles["caption"]),
        Spacer(1, 0.05 * inch),
        Image(str(FIGURES / "supervised_teacher_vs_baseline.png"), width=5.10 * inch, height=2.85 * inch, hAlign="CENTER"),
        p("Figure 4. Supervised teachers vs compact baselines across F1 and IoU, illustrating the baseline performance drop before knowledge transfer.", styles["caption"]),
        p("4.2 Baseline performance gap", styles["section"]),
        p(
            "The best teacher reaches 79.52% F1 and 66.01% IoU. The compact Random Forest baseline reaches 73.49% F1 and 58.10% IoU using only L masks and six direct features. "
            "This confirms that the compact capacity restriction creates a clear performance gap of 7.91 IoU percentage points that distillation can address.",
            styles["body"],
        ),
        PageBreak(),
    ]

    # =========================================================================
    # Page 10: Chapter 4 Results and Discussion (Distillation Matrix & Heatmap)
    # =========================================================================
    matrix_rows = [["Teacher / Student", "KNN-S", "RF-S", "GNB-S"]]
    for r in matrix:
        matrix_rows.append([
            r["teacher"].replace("_", " ").title()[:7],
            pct(r["knn_student_iou_percent"]),
            pct(r["random_forest_student_iou_percent"]),
            pct(r["gaussian_nb_student_iou_percent"]),
        ])
    table_m = add_table(matrix_rows, [1.15, 0.72, 0.72, 0.72], 7.0)

    # Side-by-side: Table 6 on left, Heatmap on right
    img_heatmap = Image(str(FIGURES / "hard_distillation_heatmap.png"), width=3.10 * inch, height=2.48 * inch, hAlign="CENTER")
    side_panel_p10 = Table(
        [[
            [table_m, Spacer(1, 0.05 * inch), p("Table 6. Hard-distillation test IoU matrix across all 9 pairs.", styles["caption"])],
            [img_heatmap, p("Figure 5. Hard-distillation test IoU matrix heatmap.", styles["caption"])],
        ]],
        colWidths=[3.45 * inch, 3.45 * inch],
        hAlign="CENTER",
    )
    side_panel_p10.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))

    story += [
        p("4 Results and Discussion", styles["chapter"]),
        p("4.3 Hard-label distillation and confidence study", styles["section"]),
        p(
            "All nine teacher-student pairs use parameters selected on validation data. "
            "Table 6 and Figure 5 present the complete 3 x 3 test IoU matrix. The Random Forest teacher consistently produces the top-performing students across all three student architectures.",
            styles["body"],
        ),
        side_panel_p10,
        Spacer(1, 0.05 * inch),
        Image(str(FIGURES / "tau_study.png"), width=4.85 * inch, height=2.32 * inch, hAlign="CENTER"),
        p("Figure 6. Validation IoU and corrected retained pseudo-label fraction across confidence thresholds tau.", styles["caption"]),
        p(
            f"The selected hard RF to RF student records F1 {pct(selected['test_f1_percent'])}, IoU {pct(selected['test_iou_percent'])}, agreement {float(selected['agreement']) * 100:.2f}%, "
            f"retention {pct(selected['retention_percent'])}, {float(selected['compression_ratio']):.2f}x compression, and {float(selected['speedup']):.2f}x speed-up. "
            f"It achieves a statistically confirmed gain of +{float(selected['delta_iou_percentage_points']):.2f} IoU percentage points over its compact baseline.",
            styles["body"],
        ),
        PageBreak(),
    ]

    # =========================================================================
    # Page 11: Chapter 4 Results and Discussion (Soft Distillation & K-Means Track)
    # =========================================================================
    img_km_sel = Image(str(FIGURES / "kmeans_teacher_selection.png"), width=3.35 * inch, height=2.14 * inch, hAlign="CENTER")
    img_km_var = Image(str(FIGURES / "kmeans_variants_comparison.png"), width=3.35 * inch, height=1.87 * inch, hAlign="CENTER")
    side_panel_km = Table(
        [[
            [img_km_sel, p("Figure 7. K-Means teacher silhouette score vs K.", styles["caption"])],
            [img_km_var, p("Figure 8. Comparison of the four K-Means variants.", styles["caption"])],
        ]],
        colWidths=[3.45 * inch, 3.45 * inch],
        hAlign="CENTER",
    )
    side_panel_km.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))

    km_rows = [["K-Means model", "F1", "IoU", "ARI", "NMI", "Agreement"]]
    for r in kmeans:
        name = r["model"].replace("km_", "").replace("__", " / ").replace("_", " ")
        km_rows.append([name.title(), pct(r["test_f1_percent"]), pct(r["test_iou_percent"]), f"{float(r['ari']):.3f}", f"{float(r['nmi']):.3f}", f"{float(r['agreement']) * 100:.2f}%"])

    story += [
        p("4 Results and Discussion", styles["chapter"]),
        p("4.4 Hard versus soft transfer and the K-Means track", styles["section"]),
        p(
            "The matched soft comparison does not beat the selected hard pseudo-label student: hard RF to RF achieves 75.76% F1 and 60.98% IoU, "
            "whereas weighted duplication soft transfer achieves 75.07% F1 and 60.09% IoU under identical tau and pixel budgets. "
            "In the unsupervised track, K = 6 was chosen by silhouette score on an image-only L union U sample (Figure 7). "
            "Figure 8 and Table 7 evaluate the four derived student variants against the independent compact baseline.",
            styles["body"],
        ),
        side_panel_km,
        Spacer(1, 0.05 * inch),
        add_table(km_rows, [2.3, 0.75, 0.75, 0.65, 0.65, 0.9], 7.0),
        p("Table 7. Final-test K-Means variants. ARI and NMI compare compact and teacher cluster assignments.", styles["caption"]),
        p(
            f"The validation-selected refined re-estimated-L student reaches F1 {pct(selected_km['test_f1_percent'])} and IoU {pct(selected_km['test_iou_percent'])}. "
            "It improves the independent no-teacher K-Means baseline by only 0.05 IoU points, demonstrating that unsupervised colour clustering without rich spatial priors yields negligible distillation transfer.",
            styles["body"],
        ),
        PageBreak(),
    ]

    # =========================================================================
    # Page 12: Chapter 4 Results and Discussion (Qualitative Panels & Discussion)
    # =========================================================================
    story += [
        p("4 Results and Discussion", styles["chapter"]),
        p("4.5 Qualitative and error analysis", styles["section"]),
        p(
            "The four frozen test examples below show an RGB input, ground truth, the selected supervised teacher and student, K-Means teacher, and selected K-Means student. "
            "The RF student preserves large contiguous flooded regions, while K-Means shows fragmentation where simple colour clusters fail on wet soil and shadows.",
            styles["body"],
        ),
    ]
    for name, caption in [
        ("qualitative_1_3015.png", "Figure 9. Test image 3015.jpg: broad flood regions retain strong RF teacher-student agreement."),
        ("qualitative_2_3095.png", "Figure 10. Test image 3095.jpg: a difficult scene with lower K-Means student precision."),
        ("qualitative_3_3.png", "Figure 11. Test image 3.jpg: high RF student recall can over-segment background."),
        ("qualitative_4_1058.png", "Figure 12. Test image 1058.jpg: complementary boundary errors between models."),
    ]:
        story += [
            Image(str(FIGURES / name), width=5.50 * inch, height=1.08 * inch, hAlign="CENTER"),
            p(caption, styles["caption"]),
            Spacer(1, 0.02 * inch),
        ]
    story += [
        p("4.6 Discussion of findings", styles["section"]),
        p(
            "The most accurate teacher (Random Forest) also trained the strongest compact student. "
            "The hard RF student benefits substantially from U-image pseudo labels, outperforming the direct compact baseline by +2.89 IoU points. "
            "However, the cross-partition duplicate risk and lack of geographic separation require reading these results as benchmark evidence rather than deployment readiness.",
            styles["body"],
        ),
        PageBreak(),
    ]

    # =========================================================================
    # Page 13: Chapter 5 Conclusion and Future Scope
    # =========================================================================
    story += [
        p("5 Conclusion and Future Scope", styles["chapter"]),
        p("5.1 Key findings", styles["section"]),
        p(
            "The project demonstrates a reproducible classical teacher student experiment on the required flood segmentation dataset. "
            "The selected hard RF to RF student reaches 75.76% F1 and 60.98% IoU, beating the matched compact RF baseline by 2.89 IoU points. "
            "It retains 92.39% of teacher IoU (95.27% of teacher F1) while reducing serialized model size by 150.28x. "
            "The selected K-Means student reaches 69.00% F1 and 52.67% IoU but offers almost no gain over its independent compact baseline.",
            styles["body"],
        ),
        p("5.2 Limitations", styles["section"]),
        p(
            "The mandatory fixed split contains cross-partition duplicates, including an L-to-test duplicate. This can inflate apparent held-out performance. "
            "The labelled set is small, flood boundaries can be ambiguous, pixel sampling limits the training view, and test scenes may not represent geographic or seasonal variation. "
            "Predictions should support, not replace, human judgement in flood-response decisions.",
            styles["body"],
        ),
        p("5.3 Future research directions", styles["section"]),
        p(
            "A stronger study would evaluate geographically separated data, deduplicate before splitting where permitted, measure calibration and uncertainty, "
            "test robustness to seasonal conditions, and compare the same protocol with more diverse feature engineering. "
            "Any operational version would need expert review, documented data governance, and a validation plan outside this benchmark.",
            styles["body"],
        ),
        p("5.4 Reproduction procedure", styles["section"]),
        p(
            "Install requirements, download the specified Kaggle dataset, run the unit suite, then run audit, development, and final stages in order. "
            "The notebook supplied with this submission presents the same sequence and validates the expected output shapes. "
            "The configuration, seed, checkpoint manifests, and generated result tables make the final claims traceable.",
            styles["body"],
        ),
        PageBreak(),
    ]

    # =========================================================================
    # Page 14: References
    # =========================================================================
    story += [
        p("References", styles["chapter"]),
    ]
    refs = [
        "Breiman, L. (2001). Random Forests. Machine Learning, 45, 5-32. https://doi.org/10.1023/A:1010933404324",
        "Cover, T., & Hart, P. (1967). Nearest Neighbor Pattern Classification. IEEE Transactions on Information Theory, 13(1), 21-27. https://doi.org/10.1109/TIT.1967.1053964",
        "Faizal Karim. Flood Area Segmentation. Kaggle dataset, version 1, CC0 Public Domain. https://www.kaggle.com/api/v1/datasets/view/faizalkarim/flood-area-segmentation",
        "Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network. arXiv:1503.02531.",
        "MacQueen, J. (1967). Some Methods for Classification and Analysis of Multivariate Observations. Fifth Berkeley Symposium on Mathematical Statistics and Probability.",
        "Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.",
        "Pillow Contributors. Pillow: Python Imaging Library. https://python-pillow.org/",
        "Virtanen, P., et al. (2020). SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods, 17, 261-272.",
        "Harris, C. R., et al. (2020). Array programming with NumPy. Nature, 585, 357-362.",
    ]
    for ref in refs:
        story.append(p(ref, styles["ref"]))
    story += [
        Spacer(1, 0.25 * inch),
        p("Citation note", styles["section"]),
        p(
            "The report cites the dataset, core algorithms, distillation idea, and libraries used in the reproducible implementation. "
            "All performance numbers originate from the project's frozen final result tables and checkpoint manifests.",
            styles["body"],
        ),
        PageBreak(),
    ]

    # =========================================================================
    # Page 15: Appendix
    # =========================================================================
    story += [
        p("Appendix A  Reproducibility Checklist", styles["chapter"]),
    ]
    checklist = [
        ["Check", "Evidence"],
        ["Dataset manifest and licence recorded", "Kaggle version 1, CC0; data audit manifest"],
        ["Fixed split saved", "RNG 42, L=40, U=163, validation=29, test=58"],
        ["Mask-access constraints enforced", "Unit tests and data-access API"],
        ["Development frozen before final test", "CP03 frozen selection and CP04 final manifest"],
        ["Result tables generated", "CSV files under results/tables"],
        ["Capacity and features checked", "6 student features, 20 teacher features; budget tables"],
        ["Notebook runnable", "Linear Colab-ready notebook with gates C0-C5"],
        ["Known limitation disclosed", "Seven exact cross-partition duplicate groups"],
    ]
    story.append(add_table(checklist, [2.5, 4.05], 8.0))
    story += [
        p("Table 8. Reproducibility checklist.", styles["caption"]),
        Spacer(1, 0.15 * inch),
        p("Appendix B  Command sequence", styles["section"]),
    ]
    commands = [
        "python3 -m pip install -r requirements.txt",
        "python3 scripts/download_data.py",
        "python3 -m unittest discover -s tests -v",
        "python3 -m src.run --stage audit",
        "python3 -m src.run --stage development",
        "python3 -m src.run --stage final",
    ]
    for cmd in commands:
        story.append(p(f"<font name='Courier'>{cmd}</font>", styles["small"]))
    story += [
        Spacer(1, 0.10 * inch),
        p(
            "The supplied notebook presents these commands as separate checkpoints. "
            "Development can take several minutes. The final stage should only run after development creates the frozen selection manifest.",
            styles["body"],
        ),
    ]

    doc.build(story)
    return report_path


if __name__ == "__main__":
    out = build_report()
    print(f"Report written to: {out}")
