# Suggested 10-slide project presentation

Use generated figures/tables rather than manually retyping results. Replace every
placeholder only after the frozen final evaluation.

1. **Title and objective**
   - Flood Area Segmentation with Classical Teacher-Student Learning
   - Name, roll number, course; one-sentence measurable objective.

2. **Problem and constraints**
   - Pixel-wise binary flood segmentation; output mask.
   - Classical ML only, fixed split, no external data/test tuning, compact student budget.

3. **Dataset audit and leakage controls**
   - Dataset counts/split/class-balance figure.
   - Fixed split protocol and strict L/U/validation/test access diagram.
   - Briefly flag duplicate cross-partition images as an unavoidable fixed-split limitation.

4. **Feature design and pipeline**
   - Generated architecture diagram.
   - phi_T versus phi_S table: 20 rich hand-crafted features versus 6 compact direct features.

5. **Supervised teacher-student design**
   - L-ground-truth teacher, L union U pseudo-labelling, confidence filter, compact student.
   - Teacher/student budgets and matching no-teacher baseline.

6. **Unsupervised K-Means design**
   - Rich K-Means teacher, L-only mapping, grouping/centroid initialization, frozen/refined variants.
   - Explicitly distinguish inherited and L-reestimated mappings.

7. **Supervised results**
   - Teacher/baseline table and 3 x 3 hard-distillation IoU matrix.
   - Tau-retention plot plus the hard-vs-soft result.

8. **Unsupervised results**
   - KT silhouette/inertia choice, no-teacher baseline, four variant table.
   - ARI/NMI/agreement interpretation.

9. **Qualitative examples and failures**
   - At least four test examples: RGB, GT, best S teacher/student, KM-T, best KM-S.
   - Concise evidence-based error categories.

10. **Final comparison, conclusion, limitations**
    - Two-row S/U F1, IoU, retention, CR, SU, Z summary.
    - Four-axis verdict: accuracy, labels, size, speed.
    - Duplicate-split limitation, deployment caveat, reproducibility link/path, Q&A.
