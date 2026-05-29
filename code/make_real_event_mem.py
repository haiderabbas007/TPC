#!/usr/bin/env python3

import os
import glob
import h5py
import numpy as np

# ============================================================
# Settings
# ============================================================

# Change this if your test HDF5 is elsewhere
CANDIDATE_FILES = [
    "out_sub/test.h5",
    "out_sub_80_180/test.h5",
    "data/test.h5",
    "test.h5",
]

OUT_MEM = "real_event.mem"

# HLS input type is likely ap_fixed<8,3>
# Total bits = 8, integer bits = 3, fractional bits = 5
# Real value -> raw integer = round(value * 2^5)
FRAC_BITS = 5
SCALE = 2 ** FRAC_BITS

# Your training log used normalization scale about 0.949074
# If your HDF5 data is already normalized, set APPLY_NORMALIZATION = False.
APPLY_NORMALIZATION = True
NORMALIZATION_SCALE = 0.9490740895271301

# Pick which event from the test file
EVENT_INDEX = 0


# ============================================================
# Helpers
# ============================================================

def find_h5_file():
    for f in CANDIDATE_FILES:
        if os.path.exists(f):
            return f

    hits = glob.glob("**/*.h5", recursive=True)
    if hits:
        print("[!] Did not find one of the expected files.")
        print("[+] Available HDF5 files:")
        for h in hits[:20]:
            print("   ", h)
        return hits[0]

    raise FileNotFoundError("No .h5 file found. Tell me where your test HDF5 file is.")


def find_dataset(h5):
    """
    Try common dataset names.
    We need something shaped like:
      (N, 20, 100, 1)
      (N, 20, 100)
      (20, 100, 1)
      (20, 100)
    """
    possible_names = [
        "X",
        "x",
        "X_test",
        "test",
        "images",
        "hitmap",
        "hitmaps",
        "data",
        "inputs",
    ]

    for name in possible_names:
        if name in h5:
            arr = h5[name]
            if len(arr.shape) in [2, 3, 4]:
                return name, arr

    # If common names fail, inspect all datasets
    datasets = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            datasets.append((name, obj.shape))

    h5.visititems(visitor)

    print("[+] Available datasets:")
    for name, shape in datasets:
        print(f"    {name}: {shape}")

    for name, shape in datasets:
        if len(shape) in [2, 3, 4] and 20 in shape and 100 in shape:
            return name, h5[name]

    raise RuntimeError("Could not automatically find a 20x100 dataset.")


def to_ap_fixed_8_3_hex(x):
    """
    Convert real x to signed ap_fixed<8,3> raw hex.
    ap_fixed<8,3> has 5 fractional bits.
    Range is approximately [-4, 3.96875].
    """
    raw = int(np.round(float(x) * SCALE))

    # signed 8-bit saturation
    raw = max(-128, min(127, raw))

    # two's complement for negative values
    if raw < 0:
        raw = (1 << 8) + raw

    return f"{raw:02x}"


# ============================================================
# Main
# ============================================================

h5_path = find_h5_file()
print(f"[+] Using HDF5 file: {h5_path}")

with h5py.File(h5_path, "r") as h5:
    dset_name, dset = find_dataset(h5)
    print(f"[+] Using dataset: {dset_name}, shape={dset.shape}")

    arr = dset[()]

# Select one event
if arr.ndim == 4:
    event = arr[EVENT_INDEX, :, :, 0]
elif arr.ndim == 3:
    if arr.shape[0] == 20 and arr.shape[1] == 100:
        event = arr[:, :, 0] if arr.shape[2] == 1 else arr[:, :]
    else:
        event = arr[EVENT_INDEX, :, :]
elif arr.ndim == 2:
    event = arr
else:
    raise RuntimeError(f"Unsupported array shape: {arr.shape}")

event = np.asarray(event, dtype=np.float32)

print("[+] Raw selected event shape:", event.shape)

# If event is 20x256, crop 80:180
if event.shape == (20, 256):
    print("[+] Detected 20x256 event. Cropping columns 80:180.")
    event = event[:, 80:180]

if event.shape != (20, 100):
    raise RuntimeError(f"Expected event shape 20x100 after crop, got {event.shape}")

if APPLY_NORMALIZATION:
    event = event / NORMALIZATION_SCALE

# Flatten in row-major order: pad index first, then time-bin
flat = event.reshape(-1)

if flat.size != 2000:
    raise RuntimeError(f"Expected 2000 samples, got {flat.size}")

with open(OUT_MEM, "w") as f:
    for val in flat:
        f.write(to_ap_fixed_8_3_hex(val) + "\n")

print(f"[+] Wrote {OUT_MEM} with {flat.size} samples.")
print("[+] First 10 real values:", flat[:10])
print("[+] First 10 hex values:")
with open(OUT_MEM, "r") as f:
    for _ in range(10):
        print("   ", f.readline().strip())
