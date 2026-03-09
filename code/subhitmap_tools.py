"""
Utility functions for preprocessing sparse TPC hitmaps.

Functions:
    rebin_drift(img, target_drift_bins):
        Downsamples the drift dimension by summing consecutive bins.

    crop_sub_hitmap(img_rebinned, window_width):
        Extracts a window centered near the region of maximum activity.
"""

import numpy as np


def rebin_drift(img: np.ndarray, target_drift_bins: int = 512) -> np.ndarray:
    """
    Rebin a TPC hitmap along the drift axis (second dimension).

    Converts an array of shape (pads, drift) into (pads, target_drift_bins)
    by summing non-overlapping drift bins.

    Parameters
    ----------
    img : np.ndarray
        Input hitmap array, shape (N_pads, N_drift_bins).

    target_drift_bins : int
        Desired drift dimension size after rebinning.

    Returns
    -------
    np.ndarray
        Rebinned image of shape (N_pads, target_drift_bins).

    Notes
    -----
    The rebinned size must divide evenly; excess columns are trimmed.
    """
    pads, drift = img.shape
    factor = drift // target_drift_bins

    # Trim to ensure exact reshape compatibility
    trimmed = img[:, : factor * target_drift_bins]

    # Reshape (pads, target, factor) and sum collapse last dimension
    rebinned = trimmed.reshape(pads, target_drift_bins, factor).sum(axis=-1)
    return rebinned


def crop_sub_hitmap(img_rebinned: np.ndarray, window_width: int = 256) -> np.ndarray:
    """
    Extract a centered window along the drift axis.

    The crop window is centered at the drift location with maximum
    integrated charge (column sum peak).

    Parameters
    ----------
    img_rebinned : np.ndarray
        Rebinned image array, shape (N_pads, N_drift_bins).

    window_width : int
        Width of the extraction window.

    Returns
    -------
    np.ndarray
        Cropped image of shape (N_pads, window_width).

    Notes
    -----
    If the ideal window extends beyond bounds, it is clipped to remain valid.
    """
    pads, drift_len = img_rebinned.shape

    # Find location with highest total activity
    col_sums = img_rebinned.sum(axis=0)
    center = int(np.argmax(col_sums))

    # Compute start/end safely
    start = max(center - window_width // 2, 0)
    end = start + window_width

    # Clamp window if at right boundary
    if end > drift_len:
        end = drift_len
        start = drift_len - window_width

    return img_rebinned[:, start:end]
