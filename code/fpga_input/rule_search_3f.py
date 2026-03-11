#!/usr/bin/env python3

from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parent
FEATURES = ROOT / "features" / "train_features.csv"


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
                "total_hits": int(r["total_hits"]),
                "roi_hits": int(r["roi_hits"]),
            })
    return rows


def metrics(rows, a, b, c):
    tp = tn = fp = fn = 0

    for r in rows:
        pred = int(
            r["roi_active_rows"] <= a
            and r["longest_row_run"] >= b
            and r["roi_longest_y_run"] >= c
        )
        y = r["label"]

        if y == 1 and pred == 1:
            tp += 1
        elif y == 0 and pred == 0:
            tn += 1
        elif y == 0 and pred == 1:
            fp += 1
        else:
            fn += 1

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }


def main():
    rows = load_rows(FEATURES)

    best = None
    best_result = None

    for a in range(1, 7):         # roi_active_rows <= a
        for b in range(1, 16):    # longest_row_run >= b
            for c in range(1, 16):# roi_longest_y_run >= c
                m = metrics(rows, a, b, c)

                # Choose what you want to optimize.
                # F1 is a good balance between precision and recall.
                score = m["f1"]

                if best is None or score > best:
                    best = score
                    best_result = (a, b, c, m)

    a, b, c, m = best_result

    print("\nBest 3-feature rule on TRAIN:\n")
    print(f"roi_active_rows <= {a}")
    print(f"longest_row_run >= {b}")
    print(f"roi_longest_y_run >= {c}\n")

    print(f"TP {m['tp']}")
    print(f"TN {m['tn']}")
    print(f"FP {m['fp']}")
    print(f"FN {m['fn']}\n")

    print(f"accuracy = {m['accuracy']}")
    print(f"precision = {m['precision']}")
    print(f"recall = {m['recall']}")
    print(f"f1 = {m['f1']}")


if __name__ == "__main__":
    main()