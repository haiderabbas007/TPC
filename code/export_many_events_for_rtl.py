#!/usr/bin/env python3

import h5py
import numpy as np
import os

H5_PATH = "out_sub_80_180/test.h5"
OUT_DIR = "rtl_events"
N_EXPORT = 20

# input_t = nnet::array<ap_fixed<10,4>, 1*1>
W = 10
I = 4
FRAC = W - I
FIXED_SCALE = 2 ** FRAC

NORM_SCALE = 0.9490740895271301

os.makedirs(OUT_DIR, exist_ok=True)

def float_to_ap_fixed_raw(x):
    raw_signed = np.rint(x * FIXED_SCALE).astype(np.int64)
    raw_signed = np.clip(raw_signed, -(2**(W-1)), 2**(W-1)-1)
    raw_unsigned = np.where(raw_signed < 0, raw_signed + 2**W, raw_signed)
    return raw_unsigned.astype(np.int64)

with h5py.File(H5_PATH, "r") as h5:
    X = h5["X"]
    y = h5["y"] if "y" in h5 else h5["labels"]

    exported = []

    for idx in range(min(N_EXPORT, X.shape[0])):
        event = np.array(X[idx], dtype=np.float32)
        event = np.squeeze(event)

        if event.shape != (20, 100):
            raise RuntimeError(f"Bad event shape at index {idx}: {event.shape}")

        label = int(np.array(y[idx]).squeeze())

        event = event / NORM_SCALE
        flat = event.reshape(-1)

        raw = float_to_ap_fixed_raw(flat)

        out_txt = os.path.join(OUT_DIR, f"event_input_{idx:03d}.txt")
        with open(out_txt, "w") as f:
            for v in raw:
                f.write(f"{int(v)}\n")

        exported.append((idx, label, out_txt, int(raw.min()), int(raw.max())))

with open(os.path.join(OUT_DIR, "events_meta.csv"), "w") as f:
    f.write("index,label,file,raw_min,raw_max\n")
    for row in exported:
        f.write(f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]}\n")

print("[+] Exported events:")
for row in exported:
    print(row)

print(f"[+] Wrote {OUT_DIR}/events_meta.csv")
