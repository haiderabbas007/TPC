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
            })
    return rows


def predict(r, a, b, c, d, e):
    branch1 = (
        r["roi_active_rows"] <= a and
        r["longest_row_run"] >= b and
        r["roi_longest_y_run"] >= c
    )

    branch2 = (
        r["roi_active_rows"] <= d and
        r["longest_y_run"] >= e
    )

    return int(branch1 or branch2)


def metrics(rows, a, b, c, d, e):
    tp = tn = fp = fn = 0

    for r in rows:
        y = r["label"]
        yhat = predict(r, a, b, c, d, e)

        if y == 1 and yhat == 1:
            tp += 1
        elif y == 0 and yhat == 0:
            tn += 1
        elif y == 0 and yhat == 1:
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
    best_rule = None
    best_metrics = None

    for a in range(1, 6):
        for b in range(1, 8):
            for c in range(1, 8):
                for d in range(1, 6):
                    for e in range(3, 20):
                        m = metrics(rows, a, b, c, d, e)

                        # prioritize recall a bit, but still reward precision
                        score = m["f1"]

                        if best is None or score > best:
                            best = score
                            best_rule = (a, b, c, d, e)
                            best_metrics = m

    a, b, c, d, e = best_rule
    m = best_metrics

    print("\nBest OR-of-ANDs rule on TRAIN:\n")
    print("Branch 1:")
    print(f"  roi_active_rows <= {a}")
    print(f"  longest_row_run >= {b}")
    print(f"  roi_longest_y_run >= {c}")
    print("Branch 2:")
    print(f"  roi_active_rows <= {d}")
    print(f"  longest_y_run >= {e}\n")

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