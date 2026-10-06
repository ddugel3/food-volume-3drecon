"""Post-hoc evaluation of saved volume predictions, not an inference component."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def read_values(path, column):
    values = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            item = int(row["id"])
            value = float(row[column])
            if item in values or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid or duplicate prediction/target in {path}")
            values[item] = value
    return values


def summarize(gt_path, predictions):
    gt = read_values(gt_path, "gt_ml")
    if set(gt) != set(range(1, 35)) or any(value <= 0 for value in gt.values()):
        raise ValueError("Expected positive ground-truth volumes for IDs 1 through 34")
    results = []
    for path in predictions:
        pred = read_values(path, "predicted")
        if pred.keys() != gt.keys():
            raise ValueError(f"Expected exactly 34 matching prediction IDs: {path}")
        errors = {item: abs(pred[item] - target) / target for item, target in gt.items()}
        result = {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for name, ids in (("all_34", range(1, 35)), ("public_10", range(1, 11)),
                          ("private_24", range(11, 35))):
            result[name] = sum(errors[item] for item in ids) / len(ids)
        results.append(result)
    return {"evaluation": "post-hoc local MAPE; not verified Kaggle scores",
            "gt_sha256": hashlib.sha256(gt_path.read_bytes()).hexdigest(),
            "results": results}


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, default=root / "outputs/_HELD_OUT_gt_volumes.csv")
    parser.add_argument("predictions", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps(summarize(args.gt, args.predictions), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
