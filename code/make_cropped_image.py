import h5py
import matplotlib.pyplot as plt
from pathlib import Path

INFILE = "out_sub/train.h5"
OUTPNG = "cropped_example.png"

Y1, Y2 = 80, 180
EVENT_INDEX = 0

with h5py.File(INFILE, "r") as f:
    X = f["X"][:]
    y = f["y"][:]

img_full = X[EVENT_INDEX, :, :, 0]
img_crop = X[EVENT_INDEX, :, Y1:Y2, 0]

plt.figure(figsize=(8, 3))
plt.imshow(img_crop, aspect="auto", origin="lower")
plt.colorbar(label="Charge")
plt.xlabel("Cropped time/sample column")
plt.ylabel("Pad row")
plt.title(f"Cropped ROI 20x100, label={y[EVENT_INDEX]}")
plt.tight_layout()
plt.savefig(OUTPNG, dpi=200)
plt.close()

print(f"Saved {OUTPNG}")
print("Full shape:", img_full.shape)
print("Cropped shape:", img_crop.shape)