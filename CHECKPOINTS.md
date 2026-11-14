# Verified build checkpoints

This log is deliberately append-only. A later experiment may replace a selected model,
but it must not erase the evidence that established an earlier gate.

## C0 - Requirements and experimental contract (passed 2026-11-09)

- The project brief was read before the general guidelines, then both were visually
  checked after text extraction.
- The exact fixed split, leakage restrictions, permitted algorithms, capacity budgets,
  evaluation metrics and submission requirements were transcribed into
  `configs/experiment.json` and the implementation plan.
- Interpretation: masks for `U` are never loaded in training or feature-selection code;
  ground truth for `L` is allowed only where the brief explicitly permits it (teacher
  fitting, no-teacher baselines, and K-Means cluster mapping).
- Verification evidence: source PDFs, their rendered pages, and the configuration review.

## Gates yet to pass

## C1 - Dataset acquisition, fixed split and audit (passed 2026-11-09)

- Downloaded Kaggle dataset `faizalkarim/flood-area-segmentation`, version 1, directly
  from its public API. Archive SHA-256:
  `e1f7ec2a37fbbfbe45a2fccab02470508e711de5cc1bf80ecea482c552f8aee5`.
- Structural audit found 290 images, 290 one-to-one paired masks, and the required
  fixed split sizes (L=40, U=163, validation=29, test=58). There were no corrupt
  source images.
- Post-preprocessing flood fractions are L=36.489%, validation=37.887%, test=38.345%.
  The test figure is descriptive only; no model-selection code receives test labels.
- Passed 28 unit tests covering splits, mask access, preprocessing, features, metrics,
  model budgets and checkpoint binding. Detailed evidence is in the generated
  `results/checkpoints/CP01_dataset_audit.json` manifest.

## Gates yet to pass

1. Development-only training samples, preprocessing and feature timing.
2. Teacher/baseline selection on validation only.
3. Distillation matrix and K-Means variants, with capacity checks.
4. Frozen final test evaluation and error analysis.
5. Reproducibility/package check.

## C2 - Leak-safe feature pipeline and capacity verification (passed 2026-11-09)

- Verified RGB conversion, 256 x 256 resizing, binary-mask thresholding, raster-order
  alignment and finite handcrafted features. The teacher uses 20 features and the
  student uses 6, satisfying the required student-to-teacher feature budget.
- Extracted features from 272 permitted images without loading a `U` mask. On the
  recorded smoke workload, teacher extraction averaged 0.00778 s/image and student
  extraction averaged 0.00123 s/image.
- Verified actual fitted reference/sample counts and model restrictions for KNN, Random
  Forest, GaussianNB, K-Means and frozen-centroid students. Evidence:
  `results/checkpoints/CP02_feature_pipeline_smoke.json`.

## C3 - Development-only selection (passed after correction, 2026-11-09)

- All teacher/baseline choices, the 3 x 3 hard-distillation matrix, confidence study,
  soft-vs-hard comparison and K-Means variants were selected using `L`, `U` images and
  validation only. No test mask was opened during this stage.
- The frozen selections are hard RF -> RF for the supervised track (validation IoU
  56.165%) and refined K-Means with re-estimated `L` mapping for the unsupervised track.
  The K-Means teacher used K=6, selected by validation-independent silhouette on the
  specified image-only sample.
- A reporting-only defect in the confidence-retention denominator was discovered during
  final packaging. It did not change pseudo-labels, fitted models, selections, or any
  final metric. The prior C3/C4 manifests were preserved with
  `_pre_tau_fraction_correction` suffixes; the source and regression test were fixed,
  then C3 was regenerated. The invalidation record is
  `results/checkpoints/INVALIDATED_CP03_CP04_tau_fraction.json`.
- Corrected evidence: `results/checkpoints/CP03_development_selection.json`, frozen
  bundle SHA-256 `de00f5ee403d65a8336b66830ab4aff9e007edfdcc61b54468fd490da53c747d`.

## C4 - Frozen final evaluation (passed after corrected C3, 2026-11-09)

- Ran the corrected frozen final evaluation over all 58 test images (3,801,088 pixels),
  bound to the corrected configuration SHA-256
  `2eef9cdf86eb58b7948cf94a9c2d16b89352479f37964d535200933c74adcd88` and C3 bundle.
- The selected supervised student reached F1=75.763% and IoU=60.983%, with 92.386%
  teacher retention and 150.28x serialized-model compression. The selected
  unsupervised student reached F1=69.000% and IoU=52.672%.
- Machine-checks confirm 9/9 hard teacher-student cells, five K-Means rows, two track
  summaries, the expected test-pixel total, and a passed CP04 manifest. Evidence:
  `results/checkpoints/CP04_final_test.json` and `results/tables/`.
- A prior C3/C4 result set exists because the corrected retention-fraction report required
  regenerating the frozen sequence. The earlier manifests remain retained and are marked
  invalidated in `INVALIDATED_CP03_CP04_tau_fraction.json`. No test result changed model
  selection, thresholds, mappings, K, or retraining; however, this history means the project
  must not claim one historical test-read event.

## C5 - Reproducibility and deliverable packaging (passed 2026-11-09)

- The deterministic unit suite passed: 75 tests, 0 failures. Final artifact invariants
  verify CP04, 58 test images / 3,801,088 pixels, nine hard-distillation cells, five
  K-Means records and two track summaries.
- Generated the original 7-page report and a revised 8-page DOCX/PDF report. Every
  rendered page was visually checked. The revised report adds the all-candidate validation
  inventory, direct no-teacher comparisons, and the corrected result-lifecycle disclosure.
  Both reports are sourced from frozen CSV/JSON artifacts rather than manually retyped values.
- Generated the original and revised 9-slide editable PPTX decks. The finalizer passed package integrity, slide
  count, heading/bullet geometry, declared Arial font policy and first-party import;
  every final slide was rendered and visually checked. A path-reference image defect
  was detected in the initial render, then repaired by embedding image bytes before
  the finalization rerun.
- Generated a 22-cell runnable notebook with download, test, audit, development,
  frozen-final and artifact-verification gates. Its JSON structure and required stage
  references were checked.
- Added an all-candidate development-validation inventory and a final no-teacher-baseline
  comparison table. The selected RF transfer gains +2.89 IoU points over its matched
  compact-RF base. The selected K-Means transfer gains only +0.05 points over its independent
  no-teacher K-Means base, so that K-Means result is not presented as a substantive gain.
