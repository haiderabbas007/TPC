#!/usr/bin/env python3
"""
roi_compare.py

Standalone ROI comparison script.
It does NOT modify or depend on your existing feature CSV files.

What it does:
1. Reads event hit text files directly from train/ val/ test
2. Computes the same ROI-dependent features as extract_features.py
3. Searches for the best OR-of-ANDs rule on TRAIN
4. Evaluates that rule on VAL and TEST
5. Repeats for one or more ROIs
6. Prints a clean comparison table

Usage examples:
    python roi_compare.py --roi 120 150 --roi 80 180
    python roi_compare.py --roi 120 150 --roi 80 180 --roi 115 145
"""

from __future__ import annotations

from pathlib import Path
import argparse
import math
import re
from collections import Counter, defaultdict

NAME_RE = re.compile(r"event_(\d+)_label_(\d)\.txt$")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default=".",
        help="Directory containing train/, val/, test/ folders",
    )
    parser.add_argument(
        "--x-min",
        type=int,
        default=0,
        help="ROI x minimum",
    )
    parser.add_argument(
        "--x-max",
        type=int,
        default=19,
        help="ROI x maximum",
    )
    parser.add_argument(
        "--roi",
        type=int,
        nargs=2,
        action="append",
        metavar=("YMIN", "YMAX"),
        required=True,
        help="Add an ROI in y, e.g. --roi 120 150",
    )
    return parser.parse_args()


def parse_event_file(path: Path) -> list[tuple[int, int]]:
    hits: list[tuple[int, int]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) != 2:
                continue
            try:
                x, y = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            hits.append((x, y))
    return hits


def longest_consecutive_run(values: list[int]) -> int:
    if not values:
        return 0

    uniq = sorted(set(values))
    best = 1
    cur = 1

    for i in range(1, len(uniq)):
        if uniq[i] == uniq[i - 1] + 1:
            cur += 1
        else:
            best = max(best, cur)
            cur = 1

    return max(best, cur)


def compute_features(
    hits: list[tuple[int, int]],
    x_min_roi: int,
    x_max_roi: int,
    y_min_roi: int,
    y_max_roi: int,
) -> dict[str, int | float]:
    total_hits = len(hits)

    xs = [x for x, _ in hits]
    ys = [y for _, y in hits]

    roi_hits_list = [
        (x, y)
        for x, y in hits
        if x_min_roi <= x <= x_max_roi and y_min_roi <= y <= y_max_roi
    ]
    roi_hits = len(roi_hits_list)

    active_rows = len(set(xs)) if xs else 0
    roi_active_rows = len({x for x, _ in roi_hits_list}) if roi_hits_list else 0

    row_counts = Counter(xs)
    max_hits_in_one_row = max(row_counts.values()) if row_counts else 0

    longest_y_run = longest_consecutive_run(ys)

    roi_ys = [y for _, y in roi_hits_list]
    roi_longest_y_run = longest_consecutive_run(roi_ys)

    ys_by_row: dict[int, list[int]] = defaultdict(list)
    for x, y in hits:
        ys_by_row[x].append(y)

    longest_row_run = 0
    for row_ys in ys_by_row.values():
        longest_row_run = max(longest_row_run, longest_consecutive_run(row_ys))

    mean_y = sum(ys) / total_hits if total_hits else 0.0
    var_y = sum((y - mean_y) ** 2 for y in ys) / total_hits if total_hits else 0.0
    std_y = math.sqrt(var_y)

    roi_frac = roi_hits / total_hits if total_hits else 0.0

    return {
        "total_hits": total_hits,
        "roi_hits": roi_hits,
        "roi_frac": roi_frac,
        "active_rows": active_rows,
        "roi_active_rows": roi_active_rows,
        "max_hits_in_one_row": max_hits_in_one_row,
        "longest_y_run": longest_y_run,
        "roi_longest_y_run": roi_longest_y_run,
        "longest_row_run": longest_row_run,
        "mean_y": mean_y,
        "std_y": std_y,
    }


def load_split_features(
    split_dir: Path,
    x_min_roi: int,
    x_max_roi: int,
    y_min_roi: int,
    y_max_roi: int,
):
    rows = []

    files = sorted(split_dir.glob("event_*_label_*.txt"))
    for path in files:
        m = NAME_RE.match(path.name)
        if not m:
            continue

        event_index = int(m.group(1))
        label = int(m.group(2))
        hits = parse_event_file(path)
        feats = compute_features(hits, x_min_roi, x_max_roi, y_min_roi, y_max_roi)

        rows.append(
            {
                "event_name": path.name,
                "event_index": event_index,
                "label": label,
                **feats,
            }
        )

    return rows


def predict_or_rule(r, a, b, c, d, e):
    branch1 = (
        r["roi_active_rows"] <= a
        and r["longest_row_run"] >= b
        and r["roi_longest_y_run"] >= c
    )

    branch2 = (
        r["roi_active_rows"] <= d
        and r["longest_y_run"] >= e
    )

    return int(branch1 or branch2)


def compute_metrics(rows, rule):
    a, b, c, d, e = rule
    tp = tn = fp = fn = 0

    false_positives = []
    false_negatives = []

    for r in rows:
        y = r["label"]
        yhat = predict_or_rule(r, a, b, c, d, e)

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

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def search_best_or_rule(train_rows):
    best_score = None
    best_rule = None
    best_metrics = None

    for a in range(1, 6):
        for b in range(1, 8):
            for c in range(1, 8):
                for d in range(1, 6):
                    for e in range(3, 20):
                        rule = (a, b, c, d, e)
                        m = compute_metrics(train_rows, rule)
                        score = m["f1"]

                        if best_score is None or score > best_score:
                            best_score = score
                            best_rule = rule
                            best_metrics = m

    return best_rule, best_metrics


def summarize_roi(train_rows, val_rows, test_rows, rule):
    train_m = compute_metrics(train_rows, rule)
    val_m = compute_metrics(val_rows, rule)
    test_m = compute_metrics(test_rows, rule)
    return train_m, val_m, test_m


def print_rule(rule):
    a, b, c, d, e = rule
    print("Best OR-of-ANDs rule on TRAIN:")
    print("  Branch 1:")
    print(f"    roi_active_rows <= {a}")
    print(f"    longest_row_run >= {b}")
    print(f"    roi_longest_y_run >= {c}")
    print("  Branch 2:")
    print(f"    roi_active_rows <= {d}")
    print(f"    longest_y_run >= {e}")


def print_metrics(name, m):
    print(f"{name}:")
    print(f"  TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']}")
    print(f"  accuracy = {m['accuracy']:.4f}")
    print(f"  precision = {m['precision']:.4f}")
    print(f"  recall = {m['recall']:.4f}")
    print(f"  f1 = {m['f1']:.4f}")


def main():
    args = parse_args()

    root = Path(args.data_dir).resolve()
    train_dir = root / "train"
    val_dir = root / "val"
    test_dir = root / "test"

    if not train_dir.exists() or not val_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            f"Could not find train/, val/, test/ under {root}"
        )

    all_results = []

    print(f"ROOT = {root}")
    print(f"X ROI = [{args.x_min}, {args.x_max}]")
    print()

    for ymin, ymax in args.roi:
        print("=" * 72)
        print(f"Evaluating ROI y = [{ymin}, {ymax}]")
        print("=" * 72)

        train_rows = load_split_features(train_dir, args.x_min, args.x_max, ymin, ymax)
        val_rows = load_split_features(val_dir, args.x_min, args.x_max, ymin, ymax)
        test_rows = load_split_features(test_dir, args.x_min, args.x_max, ymin, ymax)

        rule, train_best = search_best_or_rule(train_rows)
        train_m, val_m, test_m = summarize_roi(train_rows, val_rows, test_rows, rule)

        print_rule(rule)
        print()
        print_metrics("TRAIN", train_m)
        print()
        print_metrics("VAL", val_m)
        print()
        print_metrics("TEST", test_m)
        print()

        all_results.append(
            {
                "roi": (ymin, ymax),
                "rule": rule,
                "train": train_m,
                "val": val_m,
                "test": test_m,
            }
        )

    print("=" * 72)
    print("COMPARISON SUMMARY")
    print("=" * 72)
    print(
        f"{'ROI':>12} | {'VAL F1':>8} | {'TEST F1':>8} | {'VAL Prec':>8} | "
        f"{'VAL Rec':>8} | {'TEST Prec':>9} | {'TEST Rec':>8}"
    )
    print("-" * 72)

    all_results_sorted = sorted(
        all_results,
        key=lambda x: x["test"]["f1"],
        reverse=True,
    )

    for item in all_results_sorted:
        ymin, ymax = item["roi"]
        val_m = item["val"]
        test_m = item["test"]
        print(
            f"[{ymin:3d},{ymax:3d}]"
            f" | {val_m['f1']:8.4f}"
            f" | {test_m['f1']:8.4f}"
            f" | {val_m['precision']:8.4f}"
            f" | {val_m['recall']:8.4f}"
            f" | {test_m['precision']:9.4f}"
            f" | {test_m['recall']:8.4f}"
        )

    print()
    best = all_results_sorted[0]
    ymin, ymax = best["roi"]
    print(f"Best ROI by TEST F1: [{ymin}, {ymax}]")
    print_rule(best["rule"])


if __name__ == "__main__":
    main()