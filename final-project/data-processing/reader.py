import uproot
import numpy as np
import pandas as pd
from tqdm import tqdm
import h5py
from collections import defaultdict
import matplotlib.pyplot as plt

# length unit = mm
def build_hitmap(df, pixel_size_xy=5, voxel_size_z=0.1, smear_sigma=0.0, radius=50, length_z=100):
    """
    Build a cylindrical 3D voxel map from hit positions in a TPC.

    Parameters:
    - df: DataFrame with columns: prePos_x, prePos_y, prePos_z
    - pixel_size_xy: bin size in x and y (mm)
    - voxel_size_z: bin size in z (mm)
    - smear_sigma: Gaussian smearing applied to x, y, z (mm)
    - radius: radius of the TPC cylinder (mm)
    - length_z: full length of the TPC along Z (mm)

    Returns:
    - hist: 3D voxel map as NumPy array
    - x_edges, y_edges, z_edges: bin edges
    """
    required_columns = {"prePos_x", "prePos_y", "prePos_z"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Input DataFrame must contain columns: {required_columns}")

    pos = df[["prePos_x", "prePos_y", "prePos_z"]].to_numpy()
    if smear_sigma > 0:
        pos += np.random.normal(0, smear_sigma, size=pos.shape)


    r2 = pos[:, 0]**2 + pos[:, 1]**2
    mask = (r2 <= radius**2) & (np.abs(pos[:, 2]) <= length_z / 2)
    pos = pos[mask]

    if len(pos) == 0:
        return None, None, None, None, None, None

    x_bins = np.arange(-radius, radius + pixel_size_xy, pixel_size_xy)
    y_bins = np.arange(-radius, radius + pixel_size_xy, pixel_size_xy)
    z_bins = np.arange(-length_z / 2, length_z / 2 + voxel_size_z, voxel_size_z)

    x_idx = np.digitize(pos[:, 0], x_bins) - 1
    y_idx = np.digitize(pos[:, 1], y_bins) - 1
    z_idx = np.digitize(pos[:, 2], z_bins) - 1

    coords = np.stack((x_idx, y_idx, z_idx), axis=1)
    uniq, counts = np.unique(coords, axis=0, return_counts=True)

    shape = (len(x_bins) - 1, len(y_bins) - 1, len(z_bins) - 1)
    return uniq, counts, shape, x_bins, y_bins, z_bins

def save_sparse_h5(h5file, event_id, coords, counts, shape):
    grp = h5file.create_group(str(event_id))
    grp.create_dataset("coords", data=coords, compression="gzip")
    grp.create_dataset("counts", data=counts, compression="gzip")
    grp.create_dataset("shape", data=shape)


def save_sparse_h5_submap(h5file, event_tag, xID, yID, coords, counts, shape):
    name = f"{event_tag}_x{xID}_y{yID}"
    grp = h5file.create_group(name)
    grp.create_dataset("coords", data=coords, compression="gzip")
    grp.create_dataset("counts", data=counts, compression="gzip")
    grp.create_dataset("shape", data=shape)


def convert_root_to_sparse_h5(filename, treename, output_h5="test.h5",
                              pixel_size_xy=0.1, voxel_size_z=100,
                              smear_sigma=0.0, radius=50, length_z=100,
                              submap_size_xy=100, events_per_group=4):
    
    """
    Parameters:
    - pixel_size_xy : bin size in x and y (mm)
    - voxel_size_z : bin size in z (mm)
    - smear_sigma : smear sigma in x,y,z (mm)
    - radius: radius of the TPC, hits outside the radius will be masked 
    - length_z: length of the TPC
    - submap_size_xy: size of the subhitmap (mm) 
    - events_per_group: how many events in this (sub)hitmap
    """
    tree = uproot.open(filename)[treename]
    df = tree.arrays(library="pd")

    df = df[df["charge"] > 0]  # Apply a cut

    grouped = df.groupby("EventID")
    all_event_ids = list(grouped.groups.keys())

    with h5py.File(output_h5, "w") as h5f:
        for i in tqdm(range(0, len(all_event_ids), events_per_group), desc="Processing event groups"):
            selected_ids = all_event_ids[i:i+events_per_group]
            df_group = df[df["EventID"].isin(selected_ids)]

            tag = f"group{i}" if events_per_group > 1 else f"evt{selected_ids[0]}"

            result = build_hitmap(
                df_group,
                pixel_size_xy=pixel_size_xy,
                voxel_size_z=voxel_size_z,
                smear_sigma=smear_sigma,
                radius=radius,
                length_z=length_z,
            )
            if result is None:
                continue

            coords, counts, shape, x_bins, y_bins, z_bins = result
            # Split into submaps
            for xID in range(0, shape[0], submap_size_xy // pixel_size_xy):
                for yID in range(0, shape[1], submap_size_xy // pixel_size_xy):
                    # Filter coords in sub region
                    sub_mask = (
                        (coords[:, 0] >= xID) & (coords[:, 0] < xID + submap_size_xy // pixel_size_xy) &
                        (coords[:, 1] >= yID) & (coords[:, 1] < yID + submap_size_xy // pixel_size_xy)
                    )
                    sub_coords = coords[sub_mask].copy()
                    sub_counts = counts[sub_mask].copy()

                    if len(sub_coords) == 0:
                        continue

                    # Re-index local coords
                    sub_coords[:, 0] -= xID
                    sub_coords[:, 1] -= yID

                    sub_shape = (
                        min(submap_size_xy // pixel_size_xy, shape[0] - xID),
                        min(submap_size_xy // pixel_size_xy, shape[1] - yID),
                        shape[2]
                    )

                    save_sparse_h5_submap(h5f, tag, xID, yID, sub_coords, sub_counts, sub_shape)




# load the h5 sample 


def load_sparse_event(file_path, event_key):
    with h5py.File(file_path, 'r') as f:
      
        grp = f[str(event_key)]
        coords = grp["coords"][:]
        counts = grp["counts"][:]
        shape = tuple(grp["shape"][:])
    return coords, counts, shape

def build_dense_hitmap(coords, counts, shape):
    hitmap = np.zeros(shape, dtype=np.float32)
    for i in range(len(coords)):
        x, y, z = coords[i]
        hitmap[int(x), int(y), int(z)] += counts[i]
    return hitmap

def plot_projection(hitmap, axis='z', cmap='viridis'):
    if axis == 'z':
        proj = np.sum(hitmap, axis=2)
        xlabel, ylabel = 'X', 'Y'
    elif axis == 'x':
        proj = np.sum(hitmap, axis=0)
        xlabel, ylabel = 'Y', 'Z'
    elif axis == 'y':
        proj = np.sum(hitmap, axis=1)
        xlabel, ylabel = 'X', 'Z'
    else:
        raise ValueError("Axis must be one of 'x', 'y', 'z'.")

    plt.figure(figsize=(6,5))
    plt.imshow(proj.T, origin='lower', cmap=cmap, aspect='auto')
    plt.colorbar(label='Hit counts')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'{axis.upper()} projection of hitmap')
    plt.show()


