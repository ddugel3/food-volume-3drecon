import importlib.util
import math
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
HAS_NUMPY = importlib.util.find_spec("numpy") is not None
if HAS_NUMPY:
    import fuse
    import shape_prior
    import make_grid


@unittest.skipUnless(HAS_NUMPY, "Install core dependencies for numeric tests")
class NumericTests(unittest.TestCase):
    def test_geometry_merge(self):
        value, _ = fuse.fuse(100, 400, None, shrink=False)
        self.assertAlmostEqual(value, 200)

    def test_equal_estimates_preserved_without_shrinkage(self):
        value, _ = fuse.fuse(100, 100, 100, w_qwen_scale=0.25, shrink=False)
        self.assertAlmostEqual(value, 100)

    def test_empty_evidence(self):
        self.assertEqual(fuse.fuse(None, None, None), (None, None))

    def test_lower_qwen_weight_reduces_influence(self):
        low, _ = fuse.fuse(100, 100, 400, w_qwen_scale=0.25, shrink=False)
        high, _ = fuse.fuse(100, 100, 400, w_qwen_scale=1, shrink=False)
        self.assertLess(low, high)

    def test_hemisphere_volume(self):
        self.assertAlmostEqual(shape_prior.interior_solid_volume(10, 10), 2 * math.pi * 10 ** 3 / 3)

    def test_shape_factors(self):
        self.assertEqual(shape_prior.fill_factor("sphere"), 0.8)
        self.assertEqual(shape_prior.fill_factor("slab"), 1.0)

    def test_container_override_bypasses_fusion(self):
        estimates = {i: [100.0] for i in range(1, 35)}
        shapes = {15: {"vk": [100.0], "vc": [321.0], "shape": "dome"}}
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(make_grid, "gather", return_value=(estimates, estimates, {}, {})), \
                 patch.object(make_grid, "load_qwen", return_value=({}, {})), \
                 patch.object(make_grid, "load_shape_table", return_value=shapes), \
                 patch.object(make_grid, "SUBMISSIONS", Path(directory)):
                output, _, _, _, missing = make_grid.build(["depthpro"], 0.25, False, "test", True, True)
                import csv
                with output.open() as stream:
                    rows = {int(row["id"]): float(row["predicted"]) for row in csv.DictReader(stream)}
                self.assertEqual(rows[15], 321.0)
                self.assertEqual(rows[1], 100.0)
                self.assertEqual(missing, 0)


if __name__ == "__main__":
    unittest.main()
