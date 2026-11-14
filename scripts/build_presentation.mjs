/* Build the 9-slide project presentation from frozen experiment results. */
import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "/Users/shinesh/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const ROOT = process.cwd();
const OUT = path.join(ROOT, "output");
const BUILD = path.join(ROOT, ".presentation-build");
const SLIDE_PNGS = path.join(OUT, "slides_rendered_revised");
const SKILL = "/Users/shinesh/.codex/plugins/cache/openai-primary-runtime/presentations/26.1112.11217/skills/presentations";
const NODE = "/Users/shinesh/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node";
const PYTHON = "/Users/shinesh/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3";
const FINAL = path.join(OUT, "Flood_Area_Segmentation_Teacher_Student_Presentation_Revised_v2.pptx");
const DRAFT = path.join(BUILD, "Flood_Area_Segmentation_Teacher_Student_Revised_v2_draft.pptx");
const FIG = path.join(ROOT, "results", "figures");

const C = {
  navy: "#12304A", teal: "#007C83", sea: "#13A8A8", pale: "#EAF4F5",
  mint: "#DDF4E5", gold: "#F5C76A", orange: "#E87527", red: "#CA4B52",
  ink: "#172B3A", gray: "#667784", light: "#F7FAFB", white: "#FFFFFF", line: "#D9E3E7",
};
const FONT = "Arial";

async function writeBlob(file, blob) {
  await fs.writeFile(file, new Uint8Array(await blob.arrayBuffer()));
}
function rect(slide, left, top, width, height, fill, radius = "rounded-lg", name = undefined) {
  return slide.shapes.add({ geometry: "roundRect", ...(name ? { name } : {}), position: { left, top, width, height }, fill,
    line: { style: "solid", fill: "none", width: 0 }, borderRadius: radius });
}
function line(slide, left, top, width, height, fill) {
  return slide.shapes.add({ geometry: "rect", position: { left, top, width, height }, fill,
    line: { style: "solid", fill: "none", width: 0 } });
}
function text(slide, value, left, top, width, height, opts = {}) {
  const s = slide.shapes.add({ geometry: "textbox", ...(opts.name ? { name: opts.name } : {}), position: { left, top, width, height }, fill: "none",
    line: { style: "solid", fill: "none", width: 0 } });
  s.text = value;
  s.text.style = {
    typeface: FONT, fontSize: opts.size ?? 18, color: opts.color ?? C.ink,
    bold: opts.bold ?? false, italic: opts.italic ?? false,
    alignment: opts.align ?? "left", verticalAlignment: opts.valign ?? "top",
    autoFit: "shrinkText", lineSpacing: opts.lineSpacing ?? 1.0,
    insets: opts.insets ?? { left: 0, right: 0, top: 0, bottom: 0 },
  };
  return s;
}
function addFooter(slide, n) {
  line(slide, 72, 678, 1136, 1, C.line);
  text(slide, "CS3103  •  Flood Area Segmentation  •  Seed 42", 72, 686, 750, 18, { size: 10, color: C.gray });
  text(slide, String(n).padStart(2, "0"), 1130, 684, 78, 18, { size: 11, color: C.teal, bold: true, align: "right" });
}
function titleSlide(slide, kicker, title, n, source) {
  slide.background.fill = C.light;
  line(slide, 0, 0, 1280, 10, C.teal);
  text(slide, kicker.toUpperCase(), 72, 34, 800, 20, { size: 12, color: C.teal, bold: true });
  text(slide, title, 72, 58, 1136, 60, { size: 34, color: C.navy, bold: true, name: `title-${n}` });
  slide.speakerNotes.text = source;
  addFooter(slide, n);
}
async function image(slide, file, left, top, width, height, alt, fit = "contain") {
  // Embed image bytes. Path references render as placeholders in a portable PPTX.
  const blob = await fs.readFile(file);
  return slide.images.add({ blob, contentType: "image/png", alt, fit, position: { left, top, width, height }, geometry: "roundRect", borderRadius: "rounded-lg" });
}
function bullet(slide, body, left, top, width, color = C.ink, size = 16, bulletColor = C.teal) {
  text(slide, "•", left, top, 18, 23, { size: size + 2, color: bulletColor, bold: true });
  text(slide, body, left + 25, top + 2, width - 25, 48, { size, color, lineSpacing: 0.95 });
}
function metricCard(slide, label, value, note, left, top, width, accent = C.teal) {
  rect(slide, left, top, width, 106, C.white, "rounded-xl");
  line(slide, left, top, 6, 106, accent);
  text(slide, label.toUpperCase(), left + 20, top + 17, width - 32, 18, { size: 11, color: C.gray, bold: true });
  text(slide, value, left + 20, top + 38, width - 32, 34, { size: 26, color: C.navy, bold: true });
  text(slide, note, left + 20, top + 77, width - 32, 18, { size: 11, color: C.gray });
}
function hBar(slide, label, f1, iou, y, highlighted = false) {
  const base = highlighted ? C.mint : C.white;
  rect(slide, 84, y, 523, 82, base, "rounded-lg");
  text(slide, label, 104, y + 13, 144, 22, { size: 16, color: C.navy, bold: true });
  text(slide, `F1 ${f1.toFixed(2)}%`, 104, y + 45, 130, 18, { size: 12, color: C.gray });
  line(slide, 250, y + 20, 260, 13, "#E5ECEF");
  line(slide, 250, y + 20, Math.round(260 * f1 / 100), 13, highlighted ? C.teal : "#5B8DE7");
  text(slide, `IoU ${iou.toFixed(2)}%`, 250, y + 45, 130, 18, { size: 12, color: C.gray });
  line(slide, 380, y + 20, 195, 13, "#E5ECEF");
  line(slide, 380, y + 20, Math.round(195 * iou / 100), 13, highlighted ? C.orange : "#88B8D9");
}
function matrixCell(slide, value, left, top, width, height, fill = C.white, bold = false) {
  rect(slide, left, top, width - 2, height - 2, fill, "rounded-sm");
  text(slide, value, left + 4, top + 12, width - 10, height - 20, { size: 16, color: fill === C.navy ? C.white : C.navy, bold, align: "center", valign: "middle" });
}

async function build() {
  await fs.mkdir(OUT, { recursive: true });
  await fs.mkdir(BUILD, { recursive: true });
  await fs.mkdir(SLIDE_PNGS, { recursive: true });
  await fs.rm(DRAFT, { force: true });
  const deck = Presentation.create({ slideSize: { width: 1280, height: 720 } });

  // 1. Cover
  {
    const s = deck.slides.add();
    s.background.fill = `linear(135deg, ${C.navy} 0%, #17495C 52%, ${C.teal} 100%)`;
    text(s, "CS3103 MACHINE LEARNING PROJECT", 72, 74, 650, 22, { size: 13, color: "#9DE8E7", bold: true });
    text(s, "Flood Area\nSegmentation", 72, 115, 590, 150, { size: 46, color: C.white, bold: true, lineSpacing: 0.88 });
    text(s, "Classical teacher–student learning\nunder a label budget", 72, 286, 550, 60, { size: 23, color: "#D8F5F3", lineSpacing: 0.95 });
    metricCard(s, "Dataset", "290", "paired source images / masks", 72, 430, 170, "#6DD5D3");
    metricCard(s, "Split seed", "42", "fixed, sorted filenames", 262, 430, 170, "#F5C76A");
    metricCard(s, "Final test", "58", "images • 3.80M pixels", 452, 430, 170, "#B3D0FF");
    rect(s, 690, 70, 518, 536, "#FFFFFF", "rounded-xl");
    await image(s, path.join(FIG, "audit_labelled_examples.png"), 715, 96, 468, 392, "Representative labelled flood images", "contain");
    text(s, "Goal", 724, 515, 100, 22, { size: 13, color: C.teal, bold: true });
    text(s, "Compress richer 20-feature teachers into 6-feature students without touching unlabelled or test masks during development.", 724, 542, 438, 52, { size: 16, color: C.ink, lineSpacing: 0.96 });
    text(s, "Student / Roll number: ____________________", 72, 650, 530, 22, { size: 13, color: "#CFEFED" });
    s.speakerNotes.text = "Source: assigned project brief and Faizal Karim, Flood Area Segmentation dataset (Kaggle version 1, CC0).";
    text(s, "01", 1130, 676, 78, 18, { size: 11, color: "#BFEFEE", bold: true, align: "right" });
  }

  // 2. Audit
  {
    const s = deck.slides.add();
    titleSlide(s, "Part A • audit", "A fixed split with explicit leakage controls", 2,
      "Source: Kaggle dataset audit generated by this project. Dataset source: https://www.kaggle.com/api/v1/datasets/view/faizalkarim/flood-area-segmentation");
    const cards = [["L", "40", "masks: teacher/baseline fit", C.teal], ["U", "163", "masks: never opened", C.gold], ["Validation", "29", "masks: all selection", "#6B9BE8"], ["Test", "58", "masks: frozen final only", C.orange]];
    cards.forEach(([name, count, note, accent], i) => metricCard(s, name, count, note, 72 + i * 280, 145, 255, accent));
    await image(s, path.join(FIG, "audit_class_balance.png"), 72, 284, 542, 286, "Flood class prevalence by fixed split", "contain");
    rect(s, 650, 284, 558, 286, C.white, "rounded-xl");
    text(s, "What the audit verified", 682, 310, 450, 28, { size: 22, color: C.navy, bold: true });
    bullet(s, "290 one-to-one image–mask pairs; no corrupt source images.", 682, 354, 470, C.ink, 16);
    bullet(s, "RGB conversion is deterministic; masks use ≥128 threshold and nearest-neighbour resize.", 682, 405, 470, C.ink, 16);
    bullet(s, "7 exact duplicate groups cross mandatory partitions — retained unchanged, disclosed as a caveat.", 682, 466, 470, C.red, 16, C.red);
    rect(s, 72, 596, 1136, 48, "#FFF5E6", "rounded-lg");
    text(s, "Leakage gate: U masks are blocked before open; test masks can only enter descriptive audit or the frozen final-evaluation stage.", 92, 610, 1096, 22, { size: 15, color: C.navy, bold: true, align: "center" });
  }

  // 3. pipeline
  {
    const s = deck.slides.add();
    titleSlide(s, "Part A/B • methodology", "Compact students inherit knowledge, not labels", 3,
      "Methods source: Hinton, Vinyals and Dean (2015), https://arxiv.org/abs/1503.02531. Implementation and results are project artifacts.");
    await image(s, path.join(FIG, "pipeline_architecture.png"), 72, 142, 1136, 284, "Leak-safe teacher student pipeline", "contain");
    metricCard(s, "Teacher features", "20", "RGB/HSV + local texture + edges + position", 72, 468, 337, C.orange);
    metricCard(s, "Student features", "6", "RGB + HSV (30% of teacher dimension)", 439, 468, 337, C.teal);
    metricCard(s, "Selection boundary", "VAL", "τ, thresholds and K chosen before test", 806, 468, 337, "#6B9BE8");
    text(s, "Permitted families only: KNN • Random Forest • GaussianNB • K-Means", 72, 615, 1136, 20, { size: 16, color: C.gray, bold: true, align: "center" });
  }

  // 4. development protocol
  {
    const s = deck.slides.add();
    titleSlide(s, "Part B • development", "Validation selection and corrected frozen evaluation", 4,
      "Evidence: results/checkpoints/CP01–CP04, frozen_selection.json, and hard_distillation_tau_validation.csv. No test labels were used for selection.");
    rect(s, 72, 148, 423, 468, C.white, "rounded-xl");
    text(s, "Checkpointed workflow", 100, 180, 320, 28, { size: 23, color: C.navy, bold: true });
    const steps = [["C0", "Contract", C.teal], ["C1", "Data + split audit", C.teal], ["C2", "Features + budgets", C.teal], ["C3", "Validation selection", C.orange], ["C4", "Frozen final test", "#6B9BE8"]];
    steps.forEach(([code, label, color], i) => {
      rect(s, 104, 230 + i * 62, 62, 37, color, "rounded-lg");
      text(s, code, 104, 239 + i * 62, 62, 18, { size: 13, color: C.white, bold: true, align: "center" });
      text(s, label, 188, 237 + i * 62, 250, 23, { size: 17, color: C.ink, bold: i >= 3 });
    });
    text(s, "Correction integrity", 104, 553, 230, 18, { size: 13, color: C.red, bold: true });
    text(s, "A reporting denominator was corrected and C3→C4 rerun. Earlier manifests are retained as invalidated. No test result changed selection or retraining.", 104, 577, 350, 40, { size: 12, color: C.gray, lineSpacing: 0.9 });
    await image(s, path.join(FIG, "tau_study.png"), 530, 148, 678, 398, "Validation confidence threshold study", "contain");
    rect(s, 530, 558, 678, 58, C.mint, "rounded-lg");
    text(s, "Selected supervised pair: RF → RF, τ=0.50 • 92.39% of teacher F1 retained on the final frozen test", 550, 578, 638, 20, { size: 15, color: C.navy, bold: true, align: "center" });
  }

  // 5. teacher and compact baseline performance
  {
    const s = deck.slides.add();
    titleSlide(s, "Part D • supervised results", "Random Forest is the accuracy teacher to beat", 5,
      "Evidence: results/tables/supervised_teachers_test.csv and supervised_baselines_test.csv. Metrics are micro-aggregated across 3,801,088 frozen test pixels.");
    text(s, "Teacher test performance", 84, 150, 450, 28, { size: 22, color: C.navy, bold: true });
    hBar(s, "Random Forest", 79.5242, 66.0085, 194, true);
    hBar(s, "KNN", 77.7381, 63.5832, 290, false);
    hBar(s, "GaussianNB", 72.3534, 56.6826, 386, false);
    text(s, "F1", 250, 175, 100, 15, { size: 11, color: C.gray, bold: true });
    text(s, "IoU", 380, 175, 100, 15, { size: 11, color: C.gray, bold: true });
    rect(s, 650, 150, 558, 440, C.white, "rounded-xl");
    text(s, "Compact no-teacher baseline", 686, 182, 420, 28, { size: 23, color: C.navy, bold: true });
    metricCard(s, "Compact RF", "73.49%", "F1 without pseudo-label transfer", 686, 235, 230, C.orange);
    metricCard(s, "Compact RF", "58.10%", "IoU without pseudo-label transfer", 948, 235, 230, C.orange);
    rect(s, 686, 376, 492, 154, C.pale, "rounded-lg");
    text(s, "Baseline definition", 713, 400, 210, 22, { size: 18, color: C.teal, bold: true });
    text(s, "The compact RF is fit directly on L masks with six features and no U pseudo-labels. It provides the fair comparator for RF transfer.", 713, 431, 438, 74, { size: 16, color: C.ink, lineSpacing: 0.95 });
    rect(s, 84, 516, 523, 74, C.navy, "rounded-lg");
    text(s, "Best teacher  •  RF: 79.52% F1 / 66.01% IoU", 104, 539, 483, 23, { size: 18, color: C.white, bold: true, align: "center" });
    text(s, "All reported timing/model-size comparisons use recorded final test artifacts; no test result feeds a new selection decision.", 72, 620, 1136, 18, { size: 13, color: C.gray, align: "center" });
  }

  // 6. hard matrix and soft comparison
  {
    const s = deck.slides.add();
    titleSlide(s, "Part B/D • distillation", "Hard distillation wins the matched soft comparison", 6,
      "Evidence: results/tables/hard_distillation_iou_matrix.csv, hard_distillation_test_long.csv and soft_vs_hard_test.csv. Teacher-student transfer follows Hinton et al. (2015) conceptually within permitted classical models.");
    text(s, "All 3 × 3 hard-distillation test IoU (%)", 72, 145, 575, 30, { size: 23, color: C.navy, bold: true });
    const left = 72, top = 192, cw = 130, ch = 62;
    [["Teacher → Student", 0, 0, C.navy, true], ["KNN", 1, 0, C.navy, true], ["RF", 2, 0, C.navy, true], ["GNB", 3, 0, C.navy, true],
      ["KNN", 0, 1, C.navy, true], ["59.41", 1, 1], ["60.06", 2, 1], ["54.97", 3, 1],
      ["RF", 0, 2, C.navy, true], ["59.37", 1, 2], ["60.98", 2, 2, C.mint, true], ["54.96", 3, 2],
      ["GNB", 0, 3, C.navy, true], ["55.35", 1, 3], ["54.64", 2, 3], ["53.32", 3, 3],
    ].forEach(([v, x, y, fill = C.white, bold = false]) => matrixCell(s, v, left + x * cw, top + y * ch, cw, ch, fill, bold));
    rect(s, 624, 192, 584, 248, C.white, "rounded-xl");
    text(s, "Validation-selected hard student", 656, 220, 450, 23, { size: 16, color: C.teal, bold: true });
    text(s, "RF → RF", 656, 250, 300, 38, { size: 33, color: C.navy, bold: true });
    text(s, "Test F1 75.76%  •  Test IoU 60.98%", 656, 299, 450, 26, { size: 19, color: C.ink, bold: true });
    text(s, "+2.89 IoU points vs compact RF baseline\n92.39% retention • 150.28× compression • 2.12× speedup", 656, 342, 480, 56, { size: 16, color: C.gray, lineSpacing: 0.94 });
    text(s, "Hard vs soft RF → RF at τ=0.50", 72, 500, 440, 26, { size: 21, color: C.navy, bold: true });
    const bX = 98, bY = 560, scale = 4.7;
    text(s, "F1", bX, bY - 23, 40, 18, { size: 12, color: C.gray, bold: true });
    line(s, bX + 46, bY + 8, 360, 20, "#E5ECEF"); line(s, bX + 46, bY + 8, 75.763 * scale, 20, C.teal); text(s, "Hard 75.76%", bX + 420, bY + 7, 150, 20, { size: 14, color: C.navy, bold: true });
    line(s, bX + 46, bY + 42, 360, 20, "#E5ECEF"); line(s, bX + 46, bY + 42, 75.070 * scale, 20, C.orange); text(s, "Soft 75.07%", bX + 420, bY + 41, 150, 20, { size: 14, color: C.navy });
    rect(s, 650, 500, 558, 104, "#FFF5E6", "rounded-lg");
    text(s, "Interpretation", 678, 523, 150, 20, { size: 15, color: C.orange, bold: true });
    text(s, "Hard pseudo-labels outperform weighted-duplication soft training by 0.69 F1 points and 0.89 IoU points in this controlled comparison.", 678, 550, 494, 40, { size: 15, color: C.ink, lineSpacing: 0.94 });
  }

  // 7. K means
  {
    const s = deck.slides.add();
    titleSlide(s, "Part C • unsupervised transfer", "K=6 maximized silhouette and validation fixed the transfer variant", 7,
      "Method source: MacQueen (1967), https://digicoll.lib.berkeley.edu/record/113015. Evidence: kmeans_teacher_selection.csv and kmeans_test_results.csv.");
    await image(s, path.join(FIG, "kmeans_teacher_selection.png"), 72, 145, 575, 354, "KMeans K selection by silhouette", "contain");
    rect(s, 684, 145, 524, 354, C.white, "rounded-xl");
    text(s, "Teacher K selection", 714, 175, 300, 26, { size: 22, color: C.navy, bold: true });
    [["K=6", "Silhouette 0.177", "Selected"], ["K=8", "Silhouette 0.165", "Not selected"], ["K=10", "Silhouette 0.150", "Not selected"]].forEach(([k, sc, st], i) => {
      rect(s, 714, 224 + i * 72, 450, 52, i === 0 ? C.mint : C.light, "rounded-lg");
      text(s, k, 735, 241 + i * 72, 70, 20, { size: 16, color: C.navy, bold: true });
      text(s, sc, 820, 241 + i * 72, 160, 20, { size: 15, color: C.gray });
      text(s, st, 1000, 241 + i * 72, 140, 20, { size: 14, color: i === 0 ? C.teal : C.gray, bold: i === 0, align: "right" });
    });
    text(s, "Selected validation variant", 72, 534, 290, 20, { size: 15, color: C.teal, bold: true });
    text(s, "Refined + re-estimated L mapping", 72, 562, 500, 33, { size: 26, color: C.navy, bold: true });
    metricCard(s, "Final F1", "69.00%", "96.49% of teacher F1 retained", 684, 512, 240, C.teal);
    metricCard(s, "Final IoU", "52.67%", "ARI 0.250 • NMI 0.350", 952, 512, 240, C.teal);
    text(s, "No-teacher K-Means base: 52.63% IoU. Selected transfer: 52.67% IoU, a +0.05-point change. This is not a substantive gain. Frozen variants are not selected post hoc.", 72, 636, 1136, 18, { size: 13, color: C.gray, align: "center" });
  }

  // 8. qualitative
  {
    const s = deck.slides.add();
    titleSlide(s, "Part D • qualitative analysis", "Where the students agree — and where they fail", 8,
      "Evidence: results/figures/qualitative_1_3015.png through qualitative_4_1058.png, produced during the frozen final evaluation.");
    const imgs = [
      ["qualitative_1_3015.png", "3015.jpg  •  large flooded region"],
      ["qualitative_2_3095.png", "3095.jpg  •  difficult scene"],
      ["qualitative_3_3.png", "3.jpg  •  RF student over-segmentation"],
      ["qualitative_4_1058.png", "1058.jpg  •  boundary differences"],
    ];
    for (const [i, [file, label]] of imgs.entries()) {
      const x = 72 + (i % 2) * 575, y = 148 + Math.floor(i / 2) * 248;
      rect(s, x, y, 540, 218, C.white, "rounded-xl");
      await image(s, path.join(FIG, file), x + 14, y + 14, 512, 166, label, "contain");
      text(s, label, x + 18, y + 187, 500, 18, { size: 13, color: C.gray, bold: true, align: "center" });
    }
    rect(s, 72, 655, 1136, 1, C.line);
    text(s, "Panels compare RGB input, ground truth, RF teacher, selected RF student, K-Means teacher and selected K-Means student.", 72, 660, 1136, 15, { size: 12, color: C.gray, align: "center" });
  }

  // 9. close
  {
    const s = deck.slides.add();
    titleSlide(s, "Conclusion", "A reproducible transfer result — with transparent limits", 9,
      "Sources: project checkpoints/tables; Faizal Karim dataset; Breiman (2001) https://doi.org/10.1023/A:1010933404324; Cover and Hart (1967) https://doi.org/10.1109/TIT.1967.1053964; MacQueen (1967); Hinton et al. (2015). Full references are in the report.");
    rect(s, 72, 150, 716, 440, C.white, "rounded-xl");
    text(s, "What this study shows", 106, 184, 420, 30, { size: 23, color: C.navy, bold: true });
    bullet(s, "The validation-selected hard RF student improves compact-RF IoU by +2.89 points on the frozen test.", 110, 237, 625, C.ink, 17);
    bullet(s, "It preserves 92.39% of teacher F1 using six features and 150.28× smaller serialized state.", 110, 310, 625, C.ink, 17);
    bullet(s, "Hard pseudo-labels beat the matched soft alternative. K-Means transfer improves its no-teacher base by only +0.05 IoU points.", 110, 383, 625, C.ink, 17);
    bullet(s, "Do not overclaim: exact cross-split duplicates can inflate generalisation estimates; RGB binary segmentation is not deployment validation.", 110, 456, 625, C.red, 17, C.red);
    rect(s, 824, 150, 384, 440, C.navy, "rounded-xl");
    text(s, "Reproduce", 860, 188, 240, 28, { size: 23, color: C.white, bold: true });
    ["pip install -r requirements.txt", "python scripts/download_data.py", "python -m unittest discover -s tests -v", "python -m src.run --stage audit", "python -m src.run --stage development", "python -m src.run --stage final"].forEach((cmd, i) => text(s, cmd, 860, 240 + i * 42, 312, 24, { size: 13, color: "#D8F5F3", bold: i === 5 }));
    text(s, "Delivery checklist", 860, 498, 220, 20, { size: 14, color: "#9DE8E7", bold: true });
    text(s, "✓ 75 tests passed\n✓ CP04 frozen final passed\n✓ report + notebook + PPTX", 860, 527, 300, 70, { size: 14, color: C.white, lineSpacing: 1.0 });
    text(s, "Thank you", 72, 624, 1136, 28, { size: 21, color: C.teal, bold: true, align: "center" });
  }

  for (const [i, slide] of deck.slides.items.entries()) {
    await writeBlob(path.join(SLIDE_PNGS, `builder-slide-${String(i + 1).padStart(2, "0")}.png`), await deck.export({ slide, format: "png", scale: 1 }));
  }
  await (await PresentationFile.exportPptx(deck)).save(DRAFT);
  const { finalizePresentation } = await import(pathToFileURL(path.join(SKILL, "container_tools", "artifact_tool_utils.mjs")).href);
  const result = await finalizePresentation({
    explicitTotalSlideCount: 9,
    requiredNativeTableOwnerSlides: [],
    requiredNativeChartOwnerSlides: [],
    workspaceDir: ROOT,
    candidatePath: DRAFT,
    finalPath: FINAL,
    pythonExecutable: PYTHON,
    integrityValidatorPath: path.join(SKILL, "container_tools", "inspect_presentation_package_integrity.py"),
    layoutValidatorPath: path.join(SKILL, "container_tools", "inspect_presentation_layout_geometry.py"),
    layoutArgs: ["--expected-slide-size-emu", "12192000,6858000", "--validate-bullet-geometry", "--validate-heading-fit"],
    fontPolicy: { basis: "design", families: [FONT] },
    verifyArtifactToolImport: true,
    receiptPath: path.join(BUILD, "presentation_revised_v2.validation.json"),
  });
  console.log(JSON.stringify({ final: FINAL, draft: DRAFT, receipt: path.join(BUILD, "presentation_revised_v2.validation.json"), result }, null, 2));
}

build().catch((error) => { console.error(error); process.exitCode = 1; });
