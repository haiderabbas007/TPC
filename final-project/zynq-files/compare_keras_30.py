import h5py
import numpy as np
import tensorflow as tf
from pathlib import Path

# Search automatically from /workspaces/TPC
root = Path("/workspaces/TPC")

model_matches = list(root.rglob("keras_model.keras"))
test_matches = list(root.rglob("test.h5"))

print("Found keras_model.keras files:")
for p in model_matches:
    print("  ", p)

print("\nFound test.h5 files:")
for p in test_matches:
    print("  ", p)

if not model_matches:
    raise FileNotFoundError("No keras_model.keras found under /workspaces/TPC")

if not test_matches:
    raise FileNotFoundError("No test.h5 found under /workspaces/TPC")

# Prefer the medium streaming project if available
model_candidates = [p for p in model_matches if "medium" in str(p).lower() and "80_180" in str(p)]
test_candidates = [p for p in test_matches if "medium" in str(p).lower() and "80_180" in str(p)]

MODEL_PATH = model_candidates[0] if model_candidates else model_matches[0]
H5_PATH = test_candidates[0] if test_candidates else test_matches[0]

print("\nUsing model:", MODEL_PATH)
print("Using test file:", H5_PATH)

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
model.summary()

with h5py.File(H5_PATH, "r") as f:
    X = f["X"][:30]
    y = f["y"][:30]

pred = model.predict(X, verbose=0).reshape(-1)

print("\nidx  true_y  keras_score  keras_pred")
print("------------------------------------")
correct = 0

for i, (yi, pi) in enumerate(zip(y, pred)):
    keras_pred = int(pi >= 0.5)
    correct += int(keras_pred == int(yi))
    print(f"{i:3d}  {int(yi):6d}  {float(pi):.8f}  {keras_pred:10d}")

print("------------------------------------")
print(f"Keras accuracy on first 30 = {correct}/30 = {correct/30:.3f}")
