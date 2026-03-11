#!/usr/bin/env python3

from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parent
FEATURES = ROOT / "features" / "train_features.csv"


def load_rows(path):
    rows = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "label": int(r["label"]),
                "roi_active_rows": int(r["roi_active_rows"]),
                "longest_row_run": int(r["longest_row_run"]),
                "longest_y_run": int(r["longest_y_run"])
            })
    return rows


def predict(r):

    if r["roi_active_rows"] <= 3 and r["longest_row_run"] >= 8:
        return 1
    else:
        return 0


def evaluate(rows):

    tp = tn = fp = fn = 0

    for r in rows:

        y = r["label"]
        yhat = predict(r)

        if y == 1 and yhat == 1:
            tp += 1
        elif y == 0 and yhat == 0:
            tn += 1
        elif y == 0 and yhat == 1:
            fp += 1
        else:
            fn += 1

    total = tp + tn + fp + fn

    print()
    print("TP", tp)
    print("TN", tn)
    print("FP", fp)
    print("FN", fn)
    print()

    print("accuracy =", (tp + tn) / total)
    print("precision =", tp / (tp + fp) if tp + fp else 0)
    print("recall =", tp / (tp + fn) if tp + fn else 0)


def main():

    rows = load_rows(FEATURES)

    evaluate(rows)


if __name__ == "__main__":
    main()