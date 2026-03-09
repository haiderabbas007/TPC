"""
Module: dataset_mixer
---------------------
This script loads positive and negative sparse hit-map HDF5 files,
balances class sizes, normalizes data, generates train/val/test splits,
and writes resulting datasets to new HDF5 files.

The file list patterns are read from `mix_map.yaml`, e.g.:

positive:
  - offline_step/out/low_p_sample_*.h5

negative:
  - offline_step/out/sample_*.h5

Typical usage:
    python dataset_mixer.py
"""

import glob
from pathlib import Path
from typing import Iterable, List, Tuple, Sequence

import yaml
import h5py
import numpy as np


DATASET_KEY = "hitmap"  # Change if your dataset name differs


def load_list(patterns: Sequence[str]) -> List[str]:
    """
    Expand a list of filename patterns into a unique sorted file list.

    Parameters
    ----------
    patterns : sequence of str
        Glob patterns to expand.

    Returns
    -------
    list of str
        Sorted unique file paths.
    """
    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    return sorted(set(files))


def load_stack(files: Iterable[str], key: str) -> np.ndarray:
    """
    Load `key` datasets from multiple HDF5 files and stack along axis 0.

    Parameters
    ----------
    files : iterable of str
        Paths to input HDF5 files.

    key : str
        Dataset key within the files.

    Returns
    -------
    np.ndarray
        Concatenated dataset.

    Raises
    ------
    KeyError
        If a file does not contain the specified dataset key.
    """
    arrays = []
    for fp in files:
        with h5py.File(fp, "r") as f:
            if key not in f:
                raise KeyError(f"{key} not found in {fp}; keys = {list(f.keys())}")
            arrays.append(f[key][...])

    return np.concatenate(arrays, axis=0)


def split_idxs(
    n: int, train: float = 0.8, val: float = 0.1, test: float = 0.1, rng: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Randomly split indices into train/val/test groups.

    Parameters
    ----------
    n : int
        Total number of samples.
    train : float
        Fraction for training set.
    val : float
        Fraction for validation set.
    test : float
        Fraction for test set.
    rng : int
        Random seed.

    Returns
    -------
    tuple of np.ndarray
        (train indices, val indices, test indices)
    """
    idx = np.arange(n)
    rs = np.random.RandomState(rng)
    rs.shuffle(idx)

    n_tr = int(train * n)
    n_va = int(val * n)
    return idx[:n_tr], idx[n_tr : n_tr + n_va], idx[n_tr + n_va :]


def save_h5(path: Path, X: np.ndarray, y: np.ndarray) -> None:
    """
    Write feature and label arrays to a compressed HDF5 file.

    Parameters
    ----------
    path : pathlib.Path
        Output HDF5 file path.

    X : np.ndarray
        Feature array.

    y : np.ndarray
        Label array.
    """
    with h5py.File(path, "w") as f:
        f.create_dataset("X", data=X, compression="gzip")
        f.create_dataset("y", data=y, compression="gzip")


def main() -> None:
    """
    Execute dataset mixing workflow:
    - Read YAML
    - Resolve file lists
    - Load stacks
    - Balance classes
    - Normalize
    - Split train/val/test
    - Save datasets
    """
    with open("mix_map.yaml") as f:
        cfg = yaml.safe_load(f)

    pos_files = load_list(cfg["positive"])
    neg_files = load_list(cfg["negative"])

    if not pos_files or not neg_files:
        raise RuntimeError(
            "mix_map.yaml resolved to empty file lists. "
            "Edit file paths under 'positive' or 'negative'."
        )

    print(f"[+] Found positive files: {len(pos_files)}")
    print(f"[+] Found negative files: {len(neg_files)}")

    Xp = load_stack(pos_files, DATASET_KEY)
    Xn = load_stack(neg_files, DATASET_KEY)

    # Balance class sizes
    m = min(len(Xp), len(Xn))
    Xp, Xn = Xp[:m], Xn[:m]
    yp = np.ones((len(Xp),), np.int64)
    yn = np.zeros((len(Xn),), np.int64)

    X = np.concatenate([Xp, Xn], axis=0)
    y = np.concatenate([yp, yn], axis=0)

    # Promote channel if missing (H, W) -> (H, W, 1)
    if X.ndim == 3:
        X = X[..., None]

    # Normalize to [0, 1]
    X = X.astype("float32")
    xmax = X.max()
    if xmax > 1.0:
        X /= xmax

    idx_tr, idx_va, idx_te = split_idxs(len(X))

    outdir = Path("out")
    outdir.mkdir(exist_ok=True)

    save_h5(outdir / "train.h5", X[idx_tr], y[idx_tr])
    save_h5(outdir / "val.h5", X[idx_va], y[idx_va])
    save_h5(outdir / "test.h5", X[idx_te], y[idx_te])

    print("Saved:", outdir / "train.h5", outdir / "val.h5", outdir / "test.h5")


if __name__ == "__main__":
    main()
