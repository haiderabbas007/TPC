#!/usr/bin/env python3

import os
import h5py
import numpy as np

H5_PATH = "out_sub/test.h5"
OUT_DIR = "event_mems"

CROP_START = 80
CROP_END = 180

NORMALIZATION_SCALE = 0.9490740895271301
APPLY_NORMALIZATION = True

INPUT_FRAC_BITS = 5
SCALE = 2 ** INPUT_FRAC_BITS

N_EVENTS = 20

os.makedirs(OUT_DIR, exist_ok=True)

def to_ap_fixed_8_3_hex(x):
    raw = int(np.round(float(x) * SCALE))
    raw = max(-128, min(127, raw))

    if raw < 0:
        raw = (1 << 8) + raw

    return f"{raw:02x}"

with h5py.File(H5_PATH, "r") as h5:
    X = h5["X"][()]
    y = h5["y"][()] if "y" in h5 else None

print("[+] X shape:", X.shape)
if y is not None:
    print("[+] y shape:", y.shape)

summary_path = os.path.join(OUT_DIR, "event_labels.txt")

with open(summary_path, "w") as summary:
    summary.write("event_index label nonzero_count max_value mem_file\n")

    for idx in range(N_EVENTS):
        event = X[idx, :, CROP_START:CROP_END, 0].astype(np.float32)

        if APPLY_NORMALIZATION:
            event = event / NORMALIZATION_SCALE

        flat = event.reshape(-1)

        mem_name = f"event_{idx:03d}.mem"
        mem_path = os.path.join(OUT_DIR, mem_name)

        with open(mem_path, "w") as f:
            for val in flat:
                f.write(to_ap_fixed_8_3_hex(val) + "\n")

        label = int(y[idx]) if y is not None else -1
        nonzero_count = int(np.count_nonzero(flat))
        max_value = float(np.max(flat))

        summary.write(f"{idx} {label} {nonzero_count} {max_value:.8f} {mem_name}\n")

        print(f"[+] Wrote {mem_path}: label={label}, nonzero={nonzero_count}, max={max_value:.6f}")

print(f"\n[+] Summary written to {summary_path}")
