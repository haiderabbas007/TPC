from pathlib import Path
from collections import Counter, defaultdict
import re

ROOT = Path(__file__).resolve().parent
FPGA_INPUT = FPGA_INPUT = Path("/workspaces/TPC/code/fpga_input")

NAME_RE = re.compile(r"event_(\d+)_label_(\d)\.txt$")


def read_hits(path: Path):
    hits = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) != 2:
                continue
            x, y = map(int, parts)
            hits.append((x, y))
    return hits


def summarize_split(split_name: str, split_dir: Path):
    files = sorted(split_dir.glob("event_*_label_*.txt"))
    if not files:
        print(f"[!] No files found in {split_dir}")
        return None

    global_y_counter = Counter()
    class_y_counter = {0: Counter(), 1: Counter()}
    all_event_ymins = []
    all_event_ymaxs = []
    class_event_ymins = {0: [], 1: []}
    class_event_ymaxs = {0: [], 1: []}
    total_hits = 0
    total_events = 0

    for path in files:
        m = NAME_RE.match(path.name)
        if not m:
            continue
        label = int(m.group(2))

        hits = read_hits(path)
        if not hits:
            continue

        ys = [y for _, y in hits]
        total_events += 1
        total_hits += len(ys)

        ymin, ymax = min(ys), max(ys)
        all_event_ymins.append(ymin)
        all_event_ymaxs.append(ymax)
        class_event_ymins[label].append(ymin)
        class_event_ymaxs[label].append(ymax)

        global_y_counter.update(ys)
        class_y_counter[label].update(ys)

    result = {
        "split": split_name,
        "events": total_events,
        "hits": total_hits,
        "global_y_counter": global_y_counter,
        "class_y_counter": class_y_counter,
        "all_event_ymins": all_event_ymins,
        "all_event_ymaxs": all_event_ymaxs,
        "class_event_ymins": class_event_ymins,
        "class_event_ymaxs": class_event_ymaxs,
    }
    return result


def print_basic_stats(name, ys):
    if not ys:
        print(f"  {name}: no data")
        return
    ys_sorted = sorted(ys)
    n = len(ys_sorted)

    def pct(p):
        idx = min(n - 1, max(0, int(round((p / 100) * (n - 1)))))
        return ys_sorted[idx]

    print(
        f"  {name}: min={ys_sorted[0]}, p5={pct(5)}, p25={pct(25)}, "
        f"median={pct(50)}, p75={pct(75)}, p95={pct(95)}, max={ys_sorted[-1]}"
    )


def roi_hit_fraction(counter: Counter, y0: int, y1: int):
    total = sum(counter.values())
    if total == 0:
        return 0.0
    inside = sum(v for y, v in counter.items() if y0 <= y <= y1)
    return inside / total


def top_y_values(counter: Counter, k=20):
    return counter.most_common(k)


def merge_counters(counters):
    out = Counter()
    for c in counters:
        out.update(c)
    return out


def main():
    splits = ["train", "val", "test"]
    results = []

    for split in splits:
        res = summarize_split(split, FPGA_INPUT / split)
        if res is not None:
            results.append(res)

    if not results:
        print("No data found.")
        return

    # Per-split stats
    for res in results:
        print(f"\n=== {res['split'].upper()} ===")
        print(f"events = {res['events']}")
        print(f"hits   = {res['hits']}")

        global_counter = res["global_y_counter"]
        all_y_values = list(global_counter.elements())
        print_basic_stats("all y", all_y_values)

        for label in [0, 1]:
            class_counter = res["class_y_counter"][label]
            class_y_values = list(class_counter.elements())
            print_basic_stats(f"class {label} y", class_y_values)

        print("  top 20 y bins overall:", top_y_values(global_counter, 20))
        print("  top 20 y bins class 0:", top_y_values(res["class_y_counter"][0], 20))
        print("  top 20 y bins class 1:", top_y_values(res["class_y_counter"][1], 20))

        for roi in [(80, 100), (100, 120), (120, 150), (80, 150)]:
            y0, y1 = roi
            frac_all = roi_hit_fraction(global_counter, y0, y1)
            frac_0 = roi_hit_fraction(res["class_y_counter"][0], y0, y1)
            frac_1 = roi_hit_fraction(res["class_y_counter"][1], y0, y1)
            print(
                f"  ROI y=[{y0},{y1}] -> hit fraction all={frac_all:.3f}, "
                f"class0={frac_0:.3f}, class1={frac_1:.3f}"
            )

    # Combined stats across all splits
    merged_global = merge_counters([r["global_y_counter"] for r in results])
    merged_0 = merge_counters([r["class_y_counter"][0] for r in results])
    merged_1 = merge_counters([r["class_y_counter"][1] for r in results])

    print("\n=== COMBINED ALL SPLITS ===")
    print_basic_stats("all y", list(merged_global.elements()))
    print_basic_stats("class 0 y", list(merged_0.elements()))
    print_basic_stats("class 1 y", list(merged_1.elements()))

    print("  top 30 y bins overall:", top_y_values(merged_global, 30))
    print("  top 30 y bins class 0:", top_y_values(merged_0, 30))
    print("  top 30 y bins class 1:", top_y_values(merged_1, 30))

    for roi in [(80, 100), (100, 120), (120, 150), (80, 150)]:
        y0, y1 = roi
        frac_all = roi_hit_fraction(merged_global, y0, y1)
        frac_0 = roi_hit_fraction(merged_0, y0, y1)
        frac_1 = roi_hit_fraction(merged_1, y0, y1)
        print(
            f"  ROI y=[{y0},{y1}] -> hit fraction all={frac_all:.3f}, "
            f"class0={frac_0:.3f}, class1={frac_1:.3f}"
        )
print("ROOT =", ROOT)
print("FPGA_INPUT =", FPGA_INPUT)

if __name__ == "__main__":
    main()