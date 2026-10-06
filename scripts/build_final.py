"""Build the historical final setting from explicitly selected local tables."""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import make_grid


def main():
    make_grid.ALIGN_GLOB = "align_depthpro_e5_shard*.csv"
    make_grid.SHAPE_GLOB = "shape_depthpro_e5.0.csv"
    make_grid.CONT_GLOB = "shape_depthpro_e5_rim.csv"
    paths = []
    for pattern in (make_grid.ALIGN_GLOB, make_grid.SHAPE_GLOB,
                    make_grid.CONT_GLOB, "qwen_qc_shard0.csv"):
        found = list(make_grid.OUTPUTS.glob(pattern))
        if not found:
            raise SystemExit(f"Missing final-setting input: outputs/{pattern}")
        paths.extend(found)
    expected = set(range(1, 35))
    available = set()
    for path in paths:
        with path.open(newline="") as stream:
            available.update(int(row["id"]) for row in csv.DictReader(stream))
    if not expected.issubset(available):
        raise SystemExit(f"Missing input IDs: {sorted(expected - available)}")
    hf, mv, _, _ = make_grid.gather(["depthpro"])
    qwen, _ = make_grid.load_qwen()
    usable = set(hf) | set(mv) | set(qwen)
    if not expected.issubset(usable):
        raise SystemExit(f"Missing usable estimates for IDs: {sorted(expected - usable)}")
    path, _, _, _, missing = make_grid.build(
        ["depthpro"], 0.25, False, "final_seeded", True, True
    )
    if missing:
        raise SystemExit(f"Unexpected fallback estimates: {missing}; do not use {path}")
    print(path)


if __name__ == "__main__":
    main()
