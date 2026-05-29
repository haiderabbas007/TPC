#!/usr/bin/env python3
"""
Train a tiny FPGA-friendly CNN for TPC sub-hitmaps.

Input:
    out_sub/train.h5
    out_sub/val.h5
    out_sub/test.h5

Each HDF5 file must contain:
    X : shape (N, 20, 256, 1)
    y : shape (N,)

Outputs:
    models/tiny_cnn_subhit_best.h5
    models/tiny_cnn_subhit_final.h5
    models/tiny_cnn_weights.txt
    models/tiny_cnn_summary.txt

    plots/tiny_cnn_accuracy.png
    plots/tiny_cnn_loss.png
    plots/tiny_cnn_confusion_matrix.png
    plots/tiny_cnn_roc_curve.png
    plots/tiny_cnn_score_distribution.png
"""

import os
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
)


# ============================================================
# 0. Reproducibility
# ============================================================

SEED = 12345
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# 1. Paths
# ============================================================

DATA_DIR = Path("out_sub")
MODEL_DIR = Path("models")
PLOT_DIR = Path("plots")

MODEL_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. Load Data
# ============================================================

def load_split(path):
    with h5py.File(path, "r") as f:
        X = f["X"][...].astype(np.float32)
        y = f["y"][...].astype(np.int32)

    # Ensure CNN shape: (N, 20, 256, 1)
    if X.ndim == 3:
        X = X[..., np.newaxis]

    return X, y


print("[+] Loading data...")

X_train, y_train = load_split(DATA_DIR / "train.h5")
X_val, y_val = load_split(DATA_DIR / "val.h5")
X_test, y_test = load_split(DATA_DIR / "test.h5")

print("Train:", X_train.shape, y_train.shape)
print("Val  :", X_val.shape, y_val.shape)
print("Test :", X_test.shape, y_test.shape)


# ============================================================
# 3. Normalize Data
# ============================================================

# Use train maximum only to avoid test-data leakage.
max_val = np.max(X_train)

if max_val > 0:
    X_train = X_train / max_val
    X_val = X_val / max_val
    X_test = X_test / max_val

print(f"[+] Normalization scale = {max_val}")


# ============================================================
# 4. Build Tiny FPGA-Friendly CNN
# ============================================================

input_shape = (20, 256, 1)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=input_shape),

    tf.keras.layers.Conv2D(
        filters=4,
        kernel_size=(3, 7),
        padding="same",
        activation="relu",
        name="conv1",
    ),
    tf.keras.layers.MaxPooling2D(
        pool_size=(2, 4),
        name="pool1",
    ),

    tf.keras.layers.Conv2D(
        filters=8,
        kernel_size=(3, 7),
        padding="same",
        activation="relu",
        name="conv2",
    ),
    tf.keras.layers.MaxPooling2D(
        pool_size=(2, 4),
        name="pool2",
    ),

    # Much better for FPGA than Flatten -> huge Dense layer
    tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool"),

    tf.keras.layers.Dense(
        8,
        activation="relu",
        name="dense1",
    ),

    tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="output",
    ),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()

with open(MODEL_DIR / "tiny_cnn_summary.txt", "w") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))


# ============================================================
# 5. Train
# ============================================================

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    MODEL_DIR / "tiny_cnn_subhit_best.h5",
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1,
)

earlystop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True,
    verbose=1,
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-5,
    verbose=1,
)

print("\n[+] Training tiny CNN...\n")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=80,
    batch_size=64,
    callbacks=[checkpoint, earlystop, reduce_lr],
    verbose=1,
)

model.save(MODEL_DIR / "tiny_cnn_subhit_final.h5")


# ============================================================
# 6. Plot Accuracy and Loss
# ============================================================

plt.figure()
plt.plot(history.history["accuracy"], label="Train accuracy")
plt.plot(history.history["val_accuracy"], label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Tiny CNN Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / "tiny_cnn_accuracy.png", dpi=200)
plt.close()

plt.figure()
plt.plot(history.history["loss"], label="Train loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Tiny CNN Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / "tiny_cnn_loss.png", dpi=200)
plt.close()


# ============================================================
# 7. Evaluation
# ============================================================

print("\n[+] Evaluating on test set...\n")

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

y_score = model.predict(X_test, verbose=0).reshape(-1)
y_pred = (y_score >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

cm = confusion_matrix(y_test, y_pred)

print("Test loss     =", test_loss)
print("Test accuracy =", acc)
print("Precision     =", prec)
print("Recall        =", rec)
print("F1 score      =", f1)
print("\nConfusion matrix:")
print(cm)

print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=4))


# ============================================================
# 8. Save Metrics to Text File
# ============================================================

with open(MODEL_DIR / "tiny_cnn_metrics.txt", "w") as f:
    f.write("Tiny CNN Test Metrics\n")
    f.write("=====================\n\n")
    f.write(f"Normalization scale = {max_val}\n\n")
    f.write(f"Test loss     = {test_loss:.6f}\n")
    f.write(f"Test accuracy = {acc:.6f}\n")
    f.write(f"Precision     = {prec:.6f}\n")
    f.write(f"Recall        = {rec:.6f}\n")
    f.write(f"F1 score      = {f1:.6f}\n\n")
    f.write("Confusion matrix:\n")
    f.write(str(cm))
    f.write("\n\nClassification report:\n")
    f.write(classification_report(y_test, y_pred, digits=4))


# ============================================================
# 9. Confusion Matrix Plot
# ============================================================

plt.figure(figsize=(5, 4))
plt.imshow(cm)
plt.title("Tiny CNN Confusion Matrix")
plt.colorbar()
plt.xticks([0, 1], ["Pred 0", "Pred 1"])
plt.yticks([0, 1], ["True 0", "True 1"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")

plt.tight_layout()
plt.savefig(PLOT_DIR / "tiny_cnn_confusion_matrix.png", dpi=200)
plt.close()


# ============================================================
# 10. ROC Curve
# ============================================================

fpr, tpr, thresholds = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Tiny CNN ROC Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / "tiny_cnn_roc_curve.png", dpi=200)
plt.close()

with open(MODEL_DIR / "tiny_cnn_metrics.txt", "a") as f:
    f.write(f"\nROC AUC = {roc_auc:.6f}\n")


# ============================================================
# 11. Score Distribution Plot
# ============================================================

scores_0 = y_score[y_test == 0]
scores_1 = y_score[y_test == 1]

plt.figure()
plt.hist(scores_0, bins=40, alpha=0.6, label="True label 0")
plt.hist(scores_1, bins=40, alpha=0.6, label="True label 1")
plt.xlabel("CNN output score")
plt.ylabel("Events")
plt.title("Tiny CNN Output Score Distribution")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / "tiny_cnn_score_distribution.png", dpi=200)
plt.close()


# ============================================================
# 12. Save Weights to Text File
# ============================================================

weights_path = MODEL_DIR / "tiny_cnn_weights.txt"

with open(weights_path, "w") as f:
    f.write("Tiny CNN Weights\n")
    f.write("================\n\n")

    for layer in model.layers:
        weights = layer.get_weights()

        if not weights:
            continue

        f.write(f"Layer: {layer.name}\n")
        f.write(f"Type : {layer.__class__.__name__}\n")

        for idx, arr in enumerate(weights):
            f.write(f"\n  Weight array {idx}\n")
            f.write(f"  Shape: {arr.shape}\n")
            f.write("  Values:\n")

            flat = arr.flatten()
            for i, val in enumerate(flat):
                f.write(f"{val:.8e}")

                if (i + 1) % 8 == 0:
                    f.write("\n")
                else:
                    f.write(" ")

            f.write("\n\n")

        f.write("-" * 80 + "\n\n")

print(f"\n[+] Saved model files in: {MODEL_DIR}")
print(f"[+] Saved plots in: {PLOT_DIR}")
print(f"[+] Saved weights to: {weights_path}")
print("\n[+] Done.")