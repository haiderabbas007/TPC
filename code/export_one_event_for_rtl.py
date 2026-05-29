#!/usr/bin/env python3

import argparse
import h5py
import numpy as np

# From hls4ml_medium_stream_cnn_80_180/firmware/defines.h:
# typedef nnet::array<ap_fixed<10,4>, 1*1> input_t;
W = 10
I = 4
SIGNED = True
FRAC = W - I
FIXED_SCALE = 2 ** FRAC

# From your training log
DEFAULT_NORM_SCALE = 0.9490740895271301


def list_datasets(h5):
    out = []
    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            out.append((name, obj.shape, obj.dtype))
    h5.visititems(visit)
    return out


def find_x_dataset(h5):
    candidates = []
    for name, shape, dtype in list_datasets(h5):
        if len(shape) == 4 and shape[1:] == (20, 100, 1):
            candidates.append(name)
        elif len(shape) == 3 and shape[1:] == (20, 100):
            candidates.append(name)
        elif len(shape) == 4 and shape[1:] == (1, 20, 100):
            candidates.append(name)

    if not candidates:
        print("Available datasets:")
        for name, shape, dtype in list_datasets(h5):
            print(f"  {name}: shape={shape}, dtype={dtype}")
        raise RuntimeError("Could not find X dataset shaped like N x 20 x 100 or N x 20 x 100 x 1")

    for preferred in ["X", "x", "data", "hitmaps", "hitmap", "images"]:
        for c in candidates:
            if c.lower().endswith(preferred.lower()):
                return c

    return candidates[0]


def find_y_dataset(h5, n):
    for name, shape, dtype in list_datasets(h5):
        if len(shape) == 1 and shape[0] == n:
            if name.lower().endswith(("y", "label", "labels", "target", "targets")):
                return name

    for name, shape, dtype in list_datasets(h5):
        if len(shape) == 1 and shape[0] == n:
            return name

    return None


def float_to_ap_fixed_raw(x):
    """
    Convert float to raw unsigned integer bits for ap_fixed<10,4>.
    ap_fixed<10,4> has 6 fractional bits.
    raw_signed = round(x * 64)
    negative values are written as 10-bit two's complement.
    """
    raw_signed = np.rint(x * FIXED_SCALE).astype(np.int64)

    min_raw = -(2 ** (W - 1))
    max_raw =  (2 ** (W - 1)) - 1
    raw_signed = np.clip(raw_signed, min_raw, max_raw)

    raw_unsigned = np.where(raw_signed < 0, raw_signed + (2 ** W), raw_signed)
    return raw_unsigned.astype(np.int64), raw_signed.astype(np.int64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--out", default="event_input_0.txt")
    parser.add_argument("--scale", type=float, default=DEFAULT_NORM_SCALE)
    parser.add_argument("--no-normalize", action="store_true")
    args = parser.parse_args()

    with h5py.File(args.h5, "r") as h5:
        xname = find_x_dataset(h5)
        X = h5[xname]
        n = X.shape[0]

        yname = find_y_dataset(h5, n)
        label = None
        if yname is not None:
            label = int(np.array(h5[yname][args.index]).squeeze())

        event = np.array(X[args.index], dtype=np.float32)
        event = np.squeeze(event)

    if event.shape != (20, 100):
        raise RuntimeError(f"Expected event shape (20,100), got {event.shape}")

    print(f"[+] H5 file: {args.h5}")
    print(f"[+] X dataset: {xname}")
    print(f"[+] Event index: {args.index}")
    print(f"[+] Label: {label}")
    print(f"[+] Event shape: {event.shape}")
    print(f"[+] Float range before norm: min={event.min()}, max={event.max()}")

    if not args.no_normalize:
        event = event / args.scale
        print(f"[+] Applied normalization: divide by {args.scale}")
    else:
        print("[!] Normalization disabled")

    print(f"[+] Float range after norm: min={event.min()}, max={event.max()}")

    flat = event.reshape(-1)
    if flat.size != 2000:
        raise RuntimeError(f"Expected 2000 samples, got {flat.size}")

    raw_unsigned, raw_signed = float_to_ap_fixed_raw(flat)

    with open(args.out, "w") as f:
        for v in raw_unsigned:
            # Write into 16-bit AXI TDATA container.
            # Actual meaningful bits are lower 10 bits.
            f.write(f"{int(v)}\n")

    meta = args.out.replace(".txt", "_meta.txt")
    with open(meta, "w") as f:
        f.write(f"h5={args.h5}\n")
        f.write(f"x_dataset={xname}\n")
        f.write(f"index={args.index}\n")
        f.write(f"label={label}\n")
        f.write("input_t=nnet::array<ap_fixed<10,4>, 1*1>\n")
        f.write(f"W={W}\n")
        f.write(f"I={I}\n")
        f.write(f"fractional_bits={FRAC}\n")
        f.write(f"fixed_scale={FIXED_SCALE}\n")
        f.write(f"normalization_scale={args.scale}\n")
        f.write(f"normalized={not args.no_normalize}\n")
        f.write(f"raw_unsigned_min={int(raw_unsigned.min())}\n")
        f.write(f"raw_unsigned_max={int(raw_unsigned.max())}\n")
        f.write(f"raw_signed_min={int(raw_signed.min())}\n")
        f.write(f"raw_signed_max={int(raw_signed.max())}\n")

    print(f"[+] Wrote {args.out}")
    print(f"[+] Wrote {meta}")
    print(f"[+] Number of samples: {len(raw_unsigned)}")
    print(f"[+] Raw unsigned range: {raw_unsigned.min()} to {raw_unsigned.max()}")
    print(f"[+] Raw signed range:   {raw_signed.min()} to {raw_signed.max()}")
    print("[+] First 20 values:")
    print(raw_unsigned[:20])


if __name__ == "__main__":
    main()
