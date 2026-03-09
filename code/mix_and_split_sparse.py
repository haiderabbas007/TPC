"""
Module: sparse_group_to_images
------------------------------
This script converts sparse 3D hitmap groups stored in HDF5 files
into 2D image projections, aggregates them per class (positive/negative),
balances datasets, normalizes images, and outputs train/val/test splits.

Workflow:
1. Resolve file lists from YAML (mix_map.yaml)
2. Parse sparse voxel hit data into (H, W) images
3. Construct class datasets with limits to avoid OOM
4. Normalize, channelize, label and split
5. Save output datasets to compressed HDF5 files

Typical usage:
    python sparse_group_to_images.py
"""

import re
import glob
import yaml
import h5py
import numpy as np
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "out"
MAP_PATH = ROOT / "mix_map.yaml"

# Avoid OOM — raise when stable
PER_CLASS_LIMIT = 200

# Regex to extract x,y size hints from group names
gx = re.compile(r"_x(\d+)")
gy = re.compile(r"_y(\d+)")


def parse_xy_from_group(name: str) -> tuple[int | None, int | None]:
    """
    Extract pixel dimensions encoded in HDF5 group names.

    Example:
        "group5212_x80_y100" → (80, 100)

    Parameters
    ----------
    name : str
        Group name string.

    Returns
    -------
    tuple of int | None
        (x_size, y_size) if detected, else (None, None).
    """
    x = gx.search(name)
    y = gy.search(name)
    return (int(x.group(1)) if x else None,
            int(y.group(1)) if y else None)


def group_to_2d(f: h5py.File, gname: str) -> np.ndarray:
    """
    Convert a sparse 3D hit group into a 2D aggregated image.

    Parameters
    ----------
    f : h5py.File
        Open HDF5 file handle.

    gname : str
        Name of group to convert.

    Returns
    -------
    np.ndarray
        Image (H, W) representing counts summed over depth axis.
    """
    g = f[gname]
    shp = np.array(g["shape"][...]).tolist()   # [Z,Y,X]
    coords = g["coords"][...]                 # (N,3)
    counts = g["counts"][...]                 # (N,)

    # Infer which axes correspond to X and Y spatial dimensions
    want_x, want_y = parse_xy_from_group(gname)
    ix = iy = iz = None

    for a in (0, 1, 2):
        for b in (0, 1, 2):
            if a == b:
                continue
            c = ({0, 1, 2} - {a, b}).pop()
            if (want_x is None or shp[a] == want_x) and \
               (want_y is None or shp[b] == want_y):
                ix, iy, iz = a, b, c
                break
        if ix is not None:
            break

    # Fallback if nothing matched
    if ix is None:
        iz, iy, ix = 0, 1, 2

    H, W = shp[iy], shp[ix]
    img = np.zeros((H, W), dtype=np.float32)

    xs = coords[:, ix]
    ys = coords[:, iy]
    vals = counts.astype(np.float32)

    # Robust bounds check
    mask = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    np.add.at(img, (ys[mask], xs[mask]), vals[mask])

    return img


def files_from_globs(patterns) -> list[str]:
    """
    Expand glob patterns into sorted unique file lists.

    Parameters
    ----------
    patterns : sequence of str
        Relative or absolute glob paths.

    Returns
    -------
    list of str
        Sorted resolved file paths.
    """
    files = []
    for p in patterns:
        full = p if p.startswith("/") else str(ROOT / p)
        files.extend(glob.glob(full))
    return sorted(set(files))


def read_all_groups_as_images(h5path: str, limit: int | None = None) -> list[np.ndarray]:
    """
    Convert every group in an HDF5 file into 2D images.

    Parameters
    ----------
    h5path : str
        Input HDF5 file path.

    limit : int | None
        Optional cap on number of images returned.

    Returns
    -------
    list of np.ndarray
        List of 2D images.
    """
    imgs = []
    with h5py.File(h5path, "r") as f:
        for gname in f.keys():
            if limit is not None and len(imgs) >= limit:
                break
            imgs.append(group_to_2d(f, gname))
    return imgs


def build_class(files: list[str],
                per_file_limit: int | None = None,
                total_limit: int | None = None) -> np.ndarray:
    """
    Build a dataset for a class from multiple sparse HDF5 files.

    Parameters
    ----------
    files : list of str
        HDF5 input files for a class.

    per_file_limit : int | None
        Cap per HDF5 file.

    total_limit : int | None
        Cap per class.

    Returns
    -------
    np.ndarray
        Stacked image array (N, H, W).
    """
    X = []
    for fp in files:
        need = None
        if total_limit is not None:
            remaining = total_limit - len(X)
            if remaining <= 0:
                break
            need = min(per_file_limit, remaining) if per_file_limit else remaining

        imgs = read_all_groups_as_images(fp, limit=need)
        X.extend(imgs)

        if total_limit and len(X) >= total_limit:
            break

    return np.stack(X, axis=0)


def split_idxs(n: int,
               train: float = 0.8,
               val: float = 0.1,
               test: float = 0.1,
               seed: int = 42):
    """
    Shuffle sample indices and return train/val/test splits.

    Returns
    -------
    tuple of np.ndarray
        (train_idx, val_idx, test_idx)
    """
    idx = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)

    ntr = int(n * train)
    nva = int(n * val)
    return idx[:ntr], idx[ntr:ntr + nva], idx[ntr + nva:]


def save_h5(path: Path, X: np.ndarray, y: np.ndarray) -> None:
    """
    Save features + labels to compressed HDF5 file.

    Parameters
    ----------
    path : Path
        Output file path.

    X : np.ndarray
        Feature data.

    y : np.ndarray
        Labels.
    """
    with h5py.File(path, "w") as f:
        f.create_dataset("X", data=X, compression="gzip")
        f.create_dataset("y", data=y, compression="gzip")


def main() -> None:
    """
    Execute dataset generation pipeline:
    - Load map YAML
    - Resolve input file lists
    - Build per-class datasets
    - Normalize, split, and save
    """
    with open(MAP_PATH) as f:
        mp = yaml.safe_load(f)

    pos_files = files_from_globs(mp["positive"])
    neg_files = files_from_globs(mp["negative"])

    if not pos_files or not neg_files:
        raise RuntimeError("mix_map.yaml resolved to empty input lists.")

    print(f"[+] positives: {len(pos_files)} files")
    print(f"[+] negatives: {len(neg_files)} files")

    Xp = build_class(pos_files, per_file_limit=50, total_limit=PER_CLASS_LIMIT)
    Xn = build_class(neg_files, per_file_limit=50, total_limit=PER_CLASS_LIMIT)

    m = min(len(Xp), len(Xn))
    Xp, Xn = Xp[:m], Xn[:m]
    yp = np.ones((m,), np.int8)
    yn = np.zeros((m,), np.int8)

    # Build full dataset
    X = np.concatenate([Xp, Xn], axis=0)
    y = np.concatenate([yp, yn], axis=0)

    # channelize & normalize
    X = X[..., None].astype("float32")
    if (mx := X.max()) > 0:
        X /= mx

    OUTDIR.mkdir(exist_ok=True)
    i_tr, i_va, i_te = split_idxs(len(X))

    save_h5(OUTDIR / "train.h5", X[i_tr], y[i_tr])
    save_h5(OUTDIR / "val.h5", X[i_va], y[i_va])
    save_h5(OUTDIR / "test.h5", X[i_te], y[i_te])

    print("Saved:", OUTDIR / "train.h5", OUTDIR / "val.h5", OUTDIR / "test.h5")


if __name__ == "__main__":
    main()