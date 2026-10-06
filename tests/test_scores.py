import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("summarize_scores", ROOT / "scripts/summarize_scores.py")
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)


class ScoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.gt = self.write("gt.csv", "gt_ml", [(i, 100) for i in range(1, 35)])

    def write(self, name, column, rows):
        path = self.root / name
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["id", column])
            writer.writerows(rows)
        return path

    def test_known_split_means(self):
        pred = self.write("pred.csv", "predicted", [(i, 110 if i <= 10 else 120) for i in range(1, 35)])
        row = scorer.summarize(self.gt, [pred])["results"][0]
        self.assertAlmostEqual(row["public_10"], 0.1)
        self.assertAlmostEqual(row["private_24"], 0.2)
        self.assertAlmostEqual(row["all_34"], (10 * 0.1 + 24 * 0.2) / 34)
        self.assertEqual(len(row["sha256"]), 64)

    def test_missing_id_rejected(self):
        pred = self.write("pred.csv", "predicted", [(i, 100) for i in range(1, 34)])
        with self.assertRaises(ValueError):
            scorer.summarize(self.gt, [pred])

    def test_duplicate_rejected(self):
        pred = self.write("pred.csv", "predicted", [(1, 100), (1, 100)])
        with self.assertRaises(ValueError):
            scorer.summarize(self.gt, [pred])

    def test_nonfinite_and_negative_rejected(self):
        for value in ("nan", "inf", -1):
            with self.subTest(value=value):
                pred = self.write("pred.csv", "predicted", [(i, value if i == 1 else 100) for i in range(1, 35)])
                with self.assertRaises(ValueError):
                    scorer.summarize(self.gt, [pred])

    def test_zero_gt_rejected(self):
        gt = self.write("zero.csv", "gt_ml", [(i, 0 if i == 1 else 100) for i in range(1, 35)])
        pred = self.write("pred.csv", "predicted", [(i, 100) for i in range(1, 35)])
        with self.assertRaises(ValueError):
            scorer.summarize(gt, [pred])

    def test_zero_prediction_allowed(self):
        pred = self.write("pred.csv", "predicted", [(i, 0) for i in range(1, 35)])
        self.assertEqual(scorer.summarize(self.gt, [pred])["results"][0]["all_34"], 1.0)


if __name__ == "__main__":
    unittest.main()
