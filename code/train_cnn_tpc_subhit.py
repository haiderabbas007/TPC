#!/usr/bin/env python3
"""
Module: cnn_subhitmap_trainer
-----------------------------
Train a CNN classifier on cropped / rebinned TPC sub-hitmaps of shape (20, 256, 1).

Pipeline:
1. Load train/val/test HDF5 datasets from `out_sub/`.
2. Build compact CNN architecture tailored to sub-hitmap geometry.
3. Train using early stopping and best checkpointing.
4. Plot accuracy/loss curves.
5. Evaluate on held-out test set and compute confusion matrix & report.
6. Save trained models and plots.

Outputs:
    models/cnn_subhit_best.h5
    models/cnn_subhit_final.h5
    subhit_acc_curve.png
    subhit_loss_curve.png
    subhit_confusion_matrix.png
"""

import os
import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report


# -------------------------------------------------------------------
# 1) Data Loading
# -------------------------------------------------------------------

def load_split(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load features and labels from an HDF5 file.

    Parameters
    ----------
    path : str
        Path to an HDF5 containing datasets 'X' and 'y'.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Feature tensor (float32) and integer labels.
    """
    with h5py.File(path, "r") as f:
        X = f["X"][...].astype(np.float32)
        y = f["y"][...].astype(np.int32)
    return X, y


print("[+] Loading sub-hitmap dataset ...")

X_train, y_train = load_split("out_sub/train.h5")
X_val,   y_val   = load_split("out_sub/val.h5")
X_test,  y_test  = load_split("out_sub/test.h5")

print(" train:", X_train.shape, y_train.shape)
print(" val  :", X_val.shape,   y_val.shape)
print(" test :", X_test.shape,  y_test.shape)


# -------------------------------------------------------------------
# 2) Model Construction
# -------------------------------------------------------------------

input_shape = (20, 256, 1)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(
        16, (3, 7), activation="relu", padding="same", input_shape=input_shape
    ),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv2D(
        32, (5, 15), activation="relu", padding="same"
    ),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)


# -------------------------------------------------------------------
# 3) Training Setup
# -------------------------------------------------------------------

os.makedirs("models", exist_ok=True)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "models/cnn_subhit_best.h5",
    monitor="val_accuracy",
    mode="max",
    save_best_only=True
)

earlystop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
)


# -------------------------------------------------------------------
# 4) Train Network
# -------------------------------------------------------------------

print("\n[+] Training model...\n")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=64,
    epochs=50,
    callbacks=[checkpoint, earlystop],
    verbose=1
)

model.save("models/cnn_subhit_final.h5")


# -------------------------------------------------------------------
# 5) Plot Learning Curves
# -------------------------------------------------------------------

plt.figure()
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("subhit_acc_curve.png")
plt.close()

plt.figure()
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("subhit_loss_curve.png")
plt.close()


# -------------------------------------------------------------------
# 6) Evaluation
# -------------------------------------------------------------------

print("\n[+] Evaluating on test set...\n")

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
print(f"\nTest loss      = {test_loss:.4f}")
print(f"Test accuracy  = {test_acc:.4f}")


# -------------------------------------------------------------------
# 7) Confusion Matrix + Report
# -------------------------------------------------------------------

y_pred = (model.predict(X_test) > 0.5).astype(int)

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion matrix:\n", cm)

print("\nClassification Report:\n",
      classification_report(y_test, y_pred, digits=3))


# -------------------------------------------------------------------
# 8) Confusion Matrix Figure
# -------------------------------------------------------------------

plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap="Blues")
plt.colorbar()
plt.xticks([0, 1], ["Pred 0", "Pred 1"])
plt.yticks([0, 1], ["True 0", "True 1"])
plt.title("Sub-Hitmap CNN Confusion Matrix")

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="black")

plt.tight_layout()
plt.savefig("subhit_confusion_matrix.png")
plt.close()

print("\n[+] Training complete. Saved models and plots.\n")
