# Flood Area Segmentation with Teacher Student Learning

**Student:** Bojedla Shinesh
**Roll number:** 2401CS75
**Course:** CS3103 Machine Learning

This repository contains a reproducible classical machine-learning project for binary flood-area segmentation. It compares KNN, Random Forest, Gaussian Naive Bayes, and K-Means teachers with compact six-feature students under a fixed labelled-image budget.

## Repository contents

- `src/` - leak-aware data access, feature extraction, model fitting, distillation, evaluation, and audit code.
- `tests/` - deterministic unit tests for splits, feature budgets, mask access, model constraints, and metrics.
- `configs/experiment.json` - the seed-42 experimental contract and search spaces.
- `results/` - saved checkpoint manifests, generated figures, tables, and provenance used by the report.
- `notebooks/Flood_Area_Segmentation_Colab.ipynb` - a linear notebook for Google Colab or local execution.
- `deliverables/` - the final report and poster PDFs.
- `docs/` - report template and slide/poster outline notes.

Raw Kaggle data and fitted model binaries are intentionally excluded from version control. The downloader recreates the raw dataset when needed.

## Run the project

```bash
python3 -m pip install -r requirements.txt
python3 scripts/download_data.py
python3 -m unittest discover -s tests -v
python3 -m src.run --stage audit
python3 -m src.run --stage development
python3 -m src.run --stage final
```

The download step uses the assigned Kaggle dataset `faizalkarim/flood-area-segmentation`. Configure Kaggle credentials before running it. Development selects all thresholds and model variants using L, U-image, and validation evidence only. The final stage requires a frozen development selection before reading test masks.

## Results at a glance

The selected hard Random Forest to Random Forest student reaches 75.76% F1 and 60.98% IoU on the frozen test set. It improves the matched compact Random Forest baseline by 2.89 IoU percentage points. The report discusses the required split's cross-partition duplicate-image limitation.

## GitHub publishing

This folder is ready to publish. Create an empty GitHub repository, then add its remote and push:

```bash
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```
