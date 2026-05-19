"""
Module: subhitmap_dataset_builder
---------------------------------
This script converts sparse 3D hitmaps into cropped / rebinned
sub-hitmaps suitable for ML training (e.g., (20,256) projections).

Processing steps:
1. Read sparsified group hitmaps from HDF5 files.
2. Convert them into dense 2D images.
3. Apply drift-direction rebinning followed by spatial cropping.
4. Balance positive/negative classes with configurable sample caps.
5. Normalize, split into train/val/test sets, and store in compressed HDF5.

Typical usage:
    python subhitmap_dataset_builder.py
"""

import re
import glob
import yaml
import h5py
import numpy as np
from pathlib import Path

from subhitmap_tools import rebin_drift, crop_sub_hitmap  # domain-specific ops


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "out_sub"
MAP_PATH = ROOT / "mix_map.yaml"

# Dataset cap per class — increase once stable
PER_CLASS_LIMIT = 4000

# Patterns used to infer X and Y dimensions from ROOT group names
gx = re.compile(r"_x(\d+)")
gy = re.compile(r"_y(\d+)")


def parse_xy_from_group(name: str) -> tuple[int | None, int | None]:
    """
    Extract X and Y dimensions encoded inside HDF5 group names.

    Example
    -------
    'group5216_x80_y100'  -> (80, 100)

    Parameters
    ----------
    name : str
        Group name string.

    Returns
    -------
    tuple[int | None, int | None]
        Parsed (x, y) sizes or (None, None) if unavailable.
    """
    x = gx.search(name)
    y = gy.search(name)
    return (
        int(x.group(1)) if x else None,
        int(y.group(1)) if y else None,
    )


def group_to_2d(f: h5py.File, gname: str) -> np.ndarray:
    """
    Convert a sparse voxel hitmap group into a dense 2D projection.

    Parameters
    ----------
    f : h5py.File
        Input file handle.
    gname : str
        Name of the group to read.

    Returns
    -------
    np.ndarray
        2D image of shape (H, W).
    """
    g = f[gname]
    shp = np.array(g["shape"][...]).tolist()
    coords = g["coords"][...]
    counts = g["counts"][...]

    want_x, want_y = parse_xy_from_group(gname)

    # Determine which axis mapping matches expected geometry
    ix = iy = iz = None
    for a in (0, 1, 2):
        for b in (0, 1, 2):
            if a == b:
                continue
            c = ({0, 1, 2} - {a, b}).pop()
            if ((want_x is None or shp[a] == want_x) and
                (want_y is None or shp[b] == want_y)):
                ix, iy, iz = a, b, c
                break
        if ix is not None:
            break

    # Default fallback if name parsing failed
    if ix is None:
        iz, iy, ix = 0, 1, 2

    H, W = shp[iy], shp[ix]
    img = np.zeros((H, W), dtype=np.float32)

    xs = coords[:, ix]
    ys = coords[:, iy]
    vals = counts.astype(np.float32)

    mask = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    np.add.at(img, (ys[mask], xs[mask]), vals[mask])

    return img


def files_from_globs(patterns) -> list[str]:
    """
    Resolve glob patterns defined in the YAML into actual file paths.

    Parameters
    ----------
    patterns : list[str]
        Relative or absolute patterns.

    Returns
    -------
    list[str]
        Sorted list of unique file paths.
    """
    files = []
    for p in patterns:
        base = p if p.startswith("/") else str(ROOT / p)
        files.extend(glob.glob(base))
    return sorted(set(files))


def read_all_groups_as_images(h5path: str, limit: int | None = None) -> list[np.ndarray]:
    """
    Convert every group in a ROOT-converted sparse HDF5 into 2D images.

    Parameters
    ----------
    h5path : str
        HDF5 input file.
    limit : int | None
        Optional maximum images to extract.

    Returns
    -------
    list[np.ndarray]
        Collection of dense 2D projections.
    """
    imgs = []
    with h5py.File(h5path, "r") as f:
        for gname in f.keys():
            if limit is not None and len(imgs) >= limit:
                break
            imgs.append(group_to_2d(f, gname))
    return imgs


def build_class(files,
                per_file_limit: int | None = None,
                total_limit: int | None = None) -> np.ndarray:
    """
    Build a class dataset, applying rebin and cropping operations.

    Transformation chain:
        (20, 10000) -> (20, 512) via rebin_drift
                      -> (20, 256) via crop_sub_hitmap

    Parameters
    ----------
    files : list[str]
        Input file paths for a given class.
    per_file_limit : int | None
        Max groups processed per file.
    total_limit : int | None
        Max total samples across files.

    Returns
    -------
    np.ndarray
        Dataset tensor of shape (N, 20, 256).
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

        for img in imgs:
            rebinned = rebin_drift(img, 512)
            sub = crop_sub_hitmap(rebinned, 256)
            X.append(sub)

        if total_limit is not None and len(X) >= total_limit:
            break

    return np.stack(X, axis=0)


def split_idxs(n: int,
               train: float = 0.8,
               val: float = 0.1,
               test: float = 0.1,
               seed: int = 42):
    """
    Shuffle sample indices and return train/val/test split arrays.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    idx = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)

    ntr = int(n * train)
    nva = int(n * val)
    return idx[:ntr], idx[ntr:ntr + nva], idx[ntr + nva:]


def save_h5(path: Path, X: np.ndarray, y: np.ndarray) -> None:
    """
    Save features and labels into a compressed HDF5.

    Parameters
    ----------
    path : Path
        Output filepath.
    X : np.ndarray
        Data tensor.
    y : np.ndarray
        Labels.
    """
    with h5py.File(path, "w") as f:
        f.create_dataset("X", data=X, compression="gzip")
        f.create_dataset("y", data=y, compression="gzip")


def main() -> None:
    """
    Main dataset generation routine.
    Loads file mappings, builds class datasets,
    normalizes, splits and saves to disk.
    """
    with open(MAP_PATH) as f:
        mp = yaml.safe_load(f)

    pos_files = files_from_globs(mp["positive"])
    neg_files = files_from_globs(mp["negative"])

    if not pos_files or not neg_files:
        raise RuntimeError("mix_map.yaml resolved to empty file lists.")

    print(f"[+] positives: {len(pos_files)} files")
    print(f"[+] negatives: {len(neg_files)} files")

    OUTDIR.mkdir(exist_ok=True)

    Xp = build_class(pos_files, per_file_limit=300, total_limit=PER_CLASS_LIMIT)
    Xn = build_class(neg_files, per_file_limit=300, total_limit=PER_CLASS_LIMIT)

    m = min(len(Xp), len(Xn))
    Xp, Xn = Xp[:m], Xn[:m]
    yp = np.ones((m,), np.int8)
    yn = np.zeros((m,), np.int8)

    X = np.concatenate([Xp, Xn], axis=0)
    y = np.concatenate([yp, yn], axis=0)

    # Normalize + add channel dimension
    X = X[..., None].astype("float32")
    mx = X.max()
    if mx > 0:
        X /= mx

    i_tr, i_va, i_te = split_idxs(len(X))

    save_h5(OUTDIR / "train.h5", X[i_tr], y[i_tr])
    save_h5(OUTDIR / "val.h5", X[i_va], y[i_va])
    save_h5(OUTDIR / "test.h5", X[i_te], y[i_te])

    print("[+] Saved sub-hit dataset:",
          OUTDIR / "train.h5",
          OUTDIR / "val.h5",
          OUTDIR / "test.h5")


if __name__ == "__main__":
    main()