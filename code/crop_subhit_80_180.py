#!/usr/bin/env python3

from pathlib import Path
import h5py
import numpy as np

IN_DIR = Path("out_sub")
OUT_DIR = Path("out_sub_80_180")

X_START = 80
X_END = 180   # Python excludes 180, so this gives 100 columns

OUT_DIR.mkdir(exist_ok=True)

def crop_file(split):
    in_path = IN_DIR / f"{split}.h5"
    out_path = OUT_DIR / f"{split}.h5"

    print(f"[+] Reading {in_path}")

    with h5py.File(in_path, "r") as f:
        X = f["X"][...]
        y = f["y"][...]

    print("    Original X:", X.shape)

    if X.ndim == 4:
        # shape: N, 20, 256, 1
        X_crop = X[:, :, X_START:X_END, :]
    elif X.ndim == 3:
        # shape: N, 20, 256
        X_crop = X[:, :, X_START:X_END]
    else:
        raise ValueError(f"Unexpected X shape: {X.shape}")

    print("    Cropped X :", X_crop.shape)

    with h5py.File(out_path, "w") as f:
        f.create_dataset("X", data=X_crop, compression="gzip")
        f.create_dataset("y", data=y, compression="gzip")

    print(f"[+] Saved {out_path}\n")


for split in ["train", "val", "test"]:
    crop_file(split)

print("[+] Done.")
print(f"[+] Cropped files saved in {OUT_DIR}")