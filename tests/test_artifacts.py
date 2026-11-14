from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.artifacts import (
    ArtifactError,
    plot_kmeans_selection,
    plot_tau_study,
    write_json,
    write_pipeline_diagram,
    write_rows_csv,
)


class ArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.output = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_json_and_csv_outputs_are_written_with_deterministic_columns(self) -> None:
        json_path = write_json(self.output / "nested" / "record.json", {"b": 2, "a": [1, 2]})
        self.assertEqual(json.loads(json_path.read_text(encoding="utf-8")), {"a": [1, 2], "b": 2})
        csv_path = write_rows_csv(
            self.output / "nested" / "rows.csv",
            [{"beta": {"x": 1}, "alpha": 2}, {"gamma": True, "alpha": 3}],
        )
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(list(rows[0]), ["alpha", "beta", "gamma"])
        self.assertEqual(rows[0]["beta"], '{"x":1}')
        self.assertEqual(rows[1]["gamma"], "True")

    def test_pipeline_and_selection_figures_are_nonempty(self) -> None:
        pipeline = write_pipeline_diagram(self.output / "pipeline.png")
        tau = plot_tau_study(
            [
                {"tau": 0.5, "valid": True, "validation_iou": 0.46, "retained_fraction": 1.0},
                {"tau": 0.6, "valid": True, "validation_iou": 0.49, "retained_fraction": 0.72},
                {"tau": 0.9, "valid": False, "reason": "one class"},
            ],
            self.output / "tau.png",
        )
        kmeans = plot_kmeans_selection(
            [
                {"clusters": 6, "silhouette": 0.22, "inertia": 345.0},
                {"clusters": 8, "silhouette": 0.25, "inertia": 290.0},
            ],
            self.output / "kmeans.png",
        )
        for path in (pipeline, tau, kmeans):
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 1_000)

    def test_bad_inputs_fail_before_writing_misleading_artifacts(self) -> None:
        with self.assertRaises(ArtifactError):
            write_rows_csv(self.output / "empty.csv", [])
        with self.assertRaises(ArtifactError):
            plot_tau_study([{"tau": 0.5, "valid": False}], self.output / "bad.png")
        with self.assertRaises(ArtifactError):
            plot_kmeans_selection([{"clusters": 1, "silhouette": 2, "inertia": -1}], self.output / "bad.png")
        with self.assertRaises(ArtifactError):
            write_pipeline_diagram(self.output / "pipeline.bad")


if __name__ == "__main__":
    unittest.main()
