import h5py
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path("example_images")
OUT_DIR.mkdir(exist_ok=True)

EVENT_INDEX = 0
Y1, Y2 = 80, 180

CANDIDATES_FULL = [
    "out_full/train.h5",
    "out_hitmaps/train.h5",
    "out_full_hitmap/train.h5",
    "out/train.h5",
]

CANDIDATES_SUB = [
    "out_sub/train.h5",
    "out_sub_80_180/train.h5",
]

def find_existing(candidates):
    for p in candidates:
        if Path(p).exists():
            return p
    return None

def save_img(img, title, outfile, figsize=(10, 3)):
    plt.figure(figsize=figsize)
    plt.imshow(img, aspect="auto", origin="lower")
    plt.colorbar(label="Charge")
    plt.xlabel("Column / time sample")
    plt.ylabel("Pad row")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()
    print("Saved:", outfile)

def load_first_image(path):
    with h5py.File(path, "r") as f:
        print(f"\nFile: {path}")
        print("Keys:", list(f.keys()))

        X = f["X"]
        y = f["y"][:] if "y" in f else None

        img = X[EVENT_INDEX, :, :, 0]
        label = y[EVENT_INDEX] if y is not None else "unknown"

    return img, label

full_file = find_existing(CANDIDATES_FULL)
sub_file = find_existing(CANDIDATES_SUB)

if full_file is None:
    print("[!] No full 20x10000 file found.")
else:
    img_full, y_full = load_first_image(full_file)
    save_img(
        img_full,
        f"Full hitmap {img_full.shape}, label={y_full}",
        OUT_DIR / "full_hitmap.png",
        figsize=(14, 3),
    )

if sub_file is None:
    print("[!] No sub-hitmap 20x256 file found.")
else:
    img_sub, y_sub = load_first_image(sub_file)
    save_img(
        img_sub,
        f"Sub-hitmap {img_sub.shape}, label={y_sub}",
        OUT_DIR / "sub_hitmap_20x256.png",
        figsize=(10, 3),
    )

    if img_sub.shape[1] >= Y2:
        img_crop = img_sub[:, Y1:Y2]
        save_img(
            img_crop,
            f"Cropped ROI {img_crop.shape}, columns {Y1}:{Y2}, label={y_sub}",
            OUT_DIR / "cropped_roi_20x100.png",
            figsize=(8, 3),
        )
    else:
        print(f"[!] Cannot crop {Y1}:{Y2}; image shape is {img_sub.shape}")

print("\nDone. Check folder: example_images/")