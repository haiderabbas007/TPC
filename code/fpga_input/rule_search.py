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
            })
    return rows


def evaluate(rows, row_thresh, run_thresh):

    tp = tn = fp = fn = 0

    for r in rows:

        pred = int(
            r["roi_active_rows"] <= row_thresh
            and r["longest_row_run"] >= run_thresh
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

    accuracy = (tp + tn) / total
    recall = tp / (tp + fn) if tp + fn else 0
    precision = tp / (tp + fp) if tp + fp else 0

    return accuracy, precision, recall, tp, tn, fp, fn


def main():

    rows = load_rows(FEATURES)

    best = None

    for row_thresh in range(1, 7):
        for run_thresh in range(3, 20):

            acc, prec, rec, tp, tn, fp, fn = evaluate(
                rows,
                row_thresh,
                run_thresh
            )

            score = rec * prec   # balanced metric

            if best is None or score > best[0]:

                best = (
                    score,
                    row_thresh,
                    run_thresh,
                    acc,
                    prec,
                    rec,
                    tp,
                    tn,
                    fp,
                    fn,
                )

    print("\nBest rule found:\n")

    (
        score,
        row_thresh,
        run_thresh,
        acc,
        prec,
        rec,
        tp,
        tn,
        fp,
        fn,
    ) = best

    print("roi_active_rows <=", row_thresh)
    print("longest_row_run >=", run_thresh)

    print()
    print("TP", tp)
    print("TN", tn)
    print("FP", fp)
    print("FN", fn)

    print()
    print("accuracy =", acc)
    print("precision =", prec)
    print("recall =", rec)


if __name__ == "__main__":
    main()