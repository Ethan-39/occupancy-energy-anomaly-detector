import unittest
from datetime import datetime, timedelta
import math
import tempfile
from pathlib import Path

from occupancy_energy_anomaly_detector import analyze_records, fit_hour_model, read_records, write_example


class OccupancyEnergyTests(unittest.TestCase):
    def test_occupancy_baseline_explains_expected_load(self):
        points = [(float(x), 0.8 + 0.1 * x) for x in range(8)]
        intercept, slope, mad = fit_hour_model(points)
        self.assertAlmostEqual(intercept, 0.8)
        self.assertAlmostEqual(slope, 0.1)
        self.assertAlmostEqual(mad, 0.0)

    def test_holdout_spikes_are_flagged_and_labels_do_not_drive_scores(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.csv"
            write_example(path)
            records = read_records(path)
        report, scored = analyze_records(records)
        flagged = [row["timestamp"] for row in scored if row["partition"] == "test" and row["flagged"]]
        known = [row["timestamp"].isoformat(timespec="minutes")
                 for row in records if row["scenario_truth"] == "synthetic_spike"]
        self.assertEqual(set(known), set(flagged))
        changed_truth = [dict(row, scenario_truth="hidden") for row in records]
        _, rescored = analyze_records(changed_truth)
        self.assertEqual([row["flagged"] for row in scored], [row["flagged"] for row in rescored])
        self.assertEqual(report["synthetic_scenario_evaluation"]["recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
