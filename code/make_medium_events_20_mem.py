#!/usr/bin/env python3

import os
import h5py
import numpy as np

H5_PATH = "out_sub/test.h5"
OUT_DIR = "event_mems_medium"
OUT_MEM = os.path.join(OUT_DIR, "events_20_medium.mem")
OUT_LABELS = os.path.join(OUT_DIR, "events_20_medium_labels.txt")

N_EVENTS = 20
CROP_START = 80
CROP_END = 180

NORMALIZATION_SCALE = 0.9490740895271301
APPLY_NORMALIZATION = True

# Medium model input_t is ap_fixed<10,4>.
# Fractional bits = 10 - 4 = 6.
FRAC_BITS = 6
SCALE = 2 ** FRAC_BITS

os.makedirs(OUT_DIR, exist_ok=True)

def to_ap_fixed_10_4_hex16(x):
    raw = int(np.round(float(x) * SCALE))

    # ap_fixed<10,4> signed raw range is [-512, 511]
    raw = max(-512, min(511, raw))

    # Store as 16-bit two's complement for the 16-bit AXI TDATA port.
    if raw < 0:
        raw = (1 << 16) + raw

    return f"{raw:04x}"

with h5py.File(H5_PATH, "r") as h5:
    X = h5["X"][()]
    y = h5["y"][()] if "y" in h5 else None

print("[+] X shape:", X.shape)
if y is not None:
    print("[+] y shape:", y.shape)

with open(OUT_MEM, "w") as mem, open(OUT_LABELS, "w") as lab:
    lab.write("event_index label nonzero_count max_value\n")

    for ev in range(N_EVENTS):
        event = X[ev, :, CROP_START:CROP_END, 0].astype(np.float32)

        if APPLY_NORMALIZATION:
            event = event / NORMALIZATION_SCALE

        flat = event.reshape(-1)

        for val in flat:
            mem.write(to_ap_fixed_10_4_hex16(val) + "\n")

        label = int(y[ev]) if y is not None else -1
        nonzero_count = int(np.count_nonzero(flat))
        max_value = float(np.max(flat))

        lab.write(f"{ev} {label} {nonzero_count} {max_value:.8f}\n")
        print(f"[+] Event {ev:02d}: label={label}, nonzero={nonzero_count}, max={max_value:.6f}")

print(f"\n[+] Wrote {OUT_MEM}")
print(f"[+] Wrote {OUT_LABELS}")
print(f"[+] Total lines expected = {N_EVENTS * 2000}")
