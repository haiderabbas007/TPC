#!/usr/bin/env python3

from pathlib import Path
import csv
from statistics import mean

ROOT = Path(__file__).resolve().parent
FEATURES = ROOT / "features" / "train_features.csv"

def load_rows(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in row:
                if key not in ("split", "event_name"):
                    try:
                        if "." in row[key]:
                            row[key] = float(row[key])
                        else:
                            row[key] = int(row[key])
                    except ValueError:
                        pass
            rows.append(row)
    return rows

def summarize(rows, feature):
    vals0 = [r[feature] for r in rows if r["label"] == 0]
    vals1 = [r[feature] for r in rows if r["label"] == 1]

    print(f"\nFeature: {feature}")
    print(f"  label 0: n={len(vals0)}, mean={mean(vals0):.3f}, min={min(vals0)}, max={max(vals0)}")
    print(f"  label 1: n={len(vals1)}, mean={mean(vals1):.3f}, min={min(vals1)}, max={max(vals1)}")

def main():
    rows = load_rows(FEATURES)
    for feature in [
        "total_hits",
        "roi_hits",
        "roi_frac",
        "roi_active_rows",
        "longest_y_run",
        "roi_longest_y_run",
        "longest_row_run",
        "std_y",
    ]:
        summarize(rows, feature)

if __name__ == "__main__":
    main()