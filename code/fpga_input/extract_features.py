#!/usr/bin/env python3
"""
Extract simple event-level features from exported FPGA hit text files.

Expected folder structure (same directory as this script):
    .
    ├── extract_features.py
    ├── train/
    │   ├── event_00000_label_1.txt
    │   └── ...
    ├── val/
    │   └── ...
    ├── test/
    │   └── ...
    └── features/
        ├── train_features.csv
        ├── val_features.csv
        └── test_features.csv
"""

from __future__ import annotations

from pathlib import Path
import csv
import math
import re
from collections import Counter, defaultdict


# -----------------------------
# User-adjustable ROI settings
# -----------------------------
X_MIN = 0
X_MAX = 19
Y_MIN = 120
Y_MAX = 150

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT
FEATURES_DIR = ROOT / "features"

NAME_RE = re.compile(r"event_(\d+)_label_(\d)\.txt$")


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

    best = max(best, cur)
    return best


def compute_features(hits: list[tuple[int, int]]) -> dict[str, float | int]:
    total_hits = len(hits)

    xs = [x for x, _ in hits]
    ys = [y for _, y in hits]

    roi_hits_list = [
        (x, y)
        for x, y in hits
        if X_MIN <= x <= X_MAX and Y_MIN <= y <= Y_MAX
    ]
    roi_hits = len(roi_hits_list)

    active_rows = len(set(xs)) if xs else 0
    roi_active_rows = len({x for x, _ in roi_hits_list}) if roi_hits_list else 0

    x_min = min(xs) if xs else -1
    x_max = max(xs) if xs else -1
    y_min = min(ys) if ys else -1
    y_max = max(ys) if ys else -1

    x_span = (x_max - x_min) if xs else 0
    y_span = (y_max - y_min) if ys else 0

    row_counts = Counter(xs)
    max_hits_in_one_row = max(row_counts.values()) if row_counts else 0

    # Global longest run in y
    longest_y_run = longest_consecutive_run(ys)

    # Longest run in y within ROI
    roi_ys = [y for _, y in roi_hits_list]
    roi_longest_y_run = longest_consecutive_run(roi_ys)

    # Longest run per row (max over rows)
    ys_by_row: dict[int, list[int]] = defaultdict(list)
    for x, y in hits:
        ys_by_row[x].append(y)

    longest_row_run = 0
    for row_ys in ys_by_row.values():
        run = longest_consecutive_run(row_ys)
        if run > longest_row_run:
            longest_row_run = run

    mean_y = sum(ys) / total_hits if total_hits else 0.0
    var_y = sum((y - mean_y) ** 2 for y in ys) / total_hits if total_hits else 0.0
    std_y = math.sqrt(var_y)

    roi_frac = (roi_hits / total_hits) if total_hits else 0.0

    return {
        "total_hits": total_hits,
        "roi_hits": roi_hits,
        "roi_frac": roi_frac,
        "active_rows": active_rows,
        "roi_active_rows": roi_active_rows,
        "x_min": x_min,
        "x_max": x_max,
        "x_span": x_span,
        "y_min": y_min,
        "y_max": y_max,
        "y_span": y_span,
        "max_hits_in_one_row": max_hits_in_one_row,
        "longest_y_run": longest_y_run,
        "roi_longest_y_run": roi_longest_y_run,
        "longest_row_run": longest_row_run,
        "mean_y": mean_y,
        "std_y": std_y,
    }


def process_split(split: str) -> None:
    in_dir = DATA_DIR / split
    out_path = FEATURES_DIR / f"{split}_features.csv"

    if not in_dir.exists():
        raise FileNotFoundError(f"Missing directory: {in_dir}")

    if not in_dir.is_dir():
        raise NotADirectoryError(f"Expected directory, got: {in_dir}")

    files = sorted(in_dir.glob("event_*_label_*.txt"))

    if not files:
        print(f"[!] No event files found in {in_dir}")

    fieldnames = [
        "split",
        "event_name",
        "event_index",
        "label",
        "total_hits",
        "roi_hits",
        "roi_frac",
        "active_rows",
        "roi_active_rows",
        "x_min",
        "x_max",
        "x_span",
        "y_min",
        "y_max",
        "y_span",
        "max_hits_in_one_row",
        "longest_y_run",
        "roi_longest_y_run",
        "longest_row_run",
        "mean_y",
        "std_y",
    ]

    rows: list[dict[str, int | float | str]] = []

    for path in files:
        m = NAME_RE.match(path.name)
        if not m:
            continue

        event_index = int(m.group(1))
        label = int(m.group(2))
        hits = parse_event_file(path)
        feats = compute_features(hits)

        row = {
            "split": split,
            "event_name": path.name,
            "event_index": event_index,
            "label": label,
            **feats,
        }
        rows.append(row)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[+] Wrote {out_path} with {len(rows)} events")


def main() -> None:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    print("[+] ROOT directory:")
    print(f"    {ROOT}")
    print("[+] Looking for data splits in:")
    print(f"    {DATA_DIR}")
    print("[+] ROI settings:")
    print(f"    X in [{X_MIN}, {X_MAX}]")
    print(f"    Y in [{Y_MIN}, {Y_MAX}]")

    for split in ("train", "val", "test"):
        process_split(split)

    print("[+] Done.")


if __name__ == "__main__":
    main()