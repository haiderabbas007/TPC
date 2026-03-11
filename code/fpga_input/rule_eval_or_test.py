#!/usr/bin/env python3

from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parent
FEATURES = ROOT / "features" / "test_features.csv"


def load_rows(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "label": int(r["label"]),
                "roi_active_rows": int(r["roi_active_rows"]),
                "longest_row_run": int(r["longest_row_run"]),
                "roi_longest_y_run": int(r["roi_longest_y_run"]),
                "longest_y_run": int(r["longest_y_run"]),
                "event_name": r["event_name"],
            })
    return rows


def predict(r):
    branch1 = (
        r["roi_active_rows"] <= 2
        and r["longest_row_run"] >= 2
        and r["roi_longest_y_run"] >= 2
    )

    branch2 = (
        r["roi_active_rows"] <= 4
        and r["longest_y_run"] >= 10
    )

    return int(branch1 or branch2)


def evaluate(rows):
    tp = tn = fp = fn = 0

    false_positives = []
    false_negatives = []

    for r in rows:
        y = r["label"]
        yhat = predict(r)

        if y == 1 and yhat == 1:
            tp += 1
        elif y == 0 and yhat == 0:
            tn += 1
        elif y == 0 and yhat == 1:
            fp += 1
            false_positives.append(r["event_name"])
        else:
            fn += 1
            false_negatives.append(r["event_name"])

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print("\nTest rule:\n")
    print("Branch 1:")
    print("  roi_active_rows <= 2")
    print("  longest_row_run >= 2")
    print("  roi_longest_y_run >= 2")
    print("Branch 2:")
    print("  roi_active_rows <= 4")
    print("  longest_y_run >= 10")

    print(f"\nTP {tp}")
    print(f"TN {tn}")
    print(f"FP {fp}")
    print(f"FN {fn}\n")

    print(f"accuracy = {accuracy}")
    print(f"precision = {precision}")
    print(f"recall = {recall}")
    print(f"f1 = {f1}")

    print("\nFirst 10 false positives:")
    for name in false_positives[:10]:
        print(" ", name)

    print("\nFirst 10 false negatives:")
    for name in false_negatives[:10]:
        print(" ", name)


def main():
    rows = load_rows(FEATURES)
    evaluate(rows)


if __name__ == "__main__":
    main()