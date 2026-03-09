#!/usr/bin/env python3
"""
Module: resnet_tpc_classifier
-----------------------------
Train and evaluate a lightweight ResNet-style CNN on TPC sub-hitmap data.

Pipeline:
1. Load train/val/test datasets from HDF5.
2. Build a compact residual CNN architecture tuned for (H,W,1) inputs.
3. Train with adaptive LR, early stopping, and best-checkpoint saving.
4. Plot loss/accuracy trends.
5. Evaluate on held-out test data and print confusion matrix + classification report.

Outputs:
    resnet_tpc_best.h5
    resnet_tpc_final.h5
    resnet_acc_curve.png
    resnet_loss_curve.png
"""

import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation, Add,
    MaxPooling2D, GlobalAveragePooling2D, Dense
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from sklearn.metrics import confusion_matrix, classification_report


# ---------------------------------------------------------------------
# Residual Block
# ---------------------------------------------------------------------

def residual_block(x: tf.Tensor, filters: int) -> tf.Tensor:
    """
    Basic residual block: Conv → BN → ReLU → Conv → BN + identity shortcut.

    Parameters
    ----------
    x : tf.Tensor
        Input feature tensor.

    filters : int
        Number of convolution output channels.

    Returns
    -------
    tf.Tensor
        Output tensor after residual addition and activation.
    """
    shortcut = x

    # First conv
    x = Conv2D(filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # Second conv
    x = Conv2D(filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)

    # Channel-match shortcut if needed
    if shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), padding="same")(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    return Activation("relu")(x)


# ---------------------------------------------------------------------
# ResNet Model Builder
# ---------------------------------------------------------------------

def build_resnet_tpc(input_shape: tuple[int, int, int]) -> Model:
    """
    Build compact ResNet variant suited for TPC hitmaps.

    Parameters
    ----------
    input_shape : tuple
        Shape of input tensors (H, W, C).

    Returns
    -------
    keras.Model
    """
    inp = Input(shape=input_shape)

    x = Conv2D(16, (3, 3), padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = residual_block(x, 16)
    x = MaxPooling2D((1, 2))(x)

    x = residual_block(x, 32)
    x = MaxPooling2D((1, 2))(x)

    x = residual_block(x, 64)

    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation="relu")(x)
    out = Dense(1, activation="sigmoid")(x)

    return Model(inp, out)


# ---------------------------------------------------------------------
# Dataset Loader
# ---------------------------------------------------------------------

def load_h5(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load feature and label arrays from an HDF5 file.

    Parameters
    ----------
    path : str
        Source file containing datasets 'X' and 'y'.

    Returns
    -------
    (X, y) : tuple[np.ndarray, np.ndarray]
        Input tensor and labels.
    """
    with h5py.File(path, "r") as f:
        X = f["X"][...]
        y = f["y"][...]
    return X, y


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

print("[+] Loading dataset…")

X_train, y_train = load_h5("out_sub/train.h5")
X_val,   y_val   = load_h5("out_sub/val.h5")
X_test,  y_test  = load_h5("out_sub/test.h5")

print("Train:", X_train.shape, y_train.shape)
print("Val  :", X_val.shape,   y_val.shape)
print("Test :", X_test.shape,  y_test.shape)

input_shape = X_train.shape[1:]


# ---------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------

model = build_resnet_tpc(input_shape)
model.summary()

model.compile(
    optimizer=Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# Callbacks
lr_sched = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-5,
    verbose=1
)

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        "resnet_tpc_best.h5",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    lr_sched
]


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

print("[+] Training network…")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

model.save("resnet_tpc_final.h5")


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------

plt.figure()
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("ResNet Training Accuracy")
plt.savefig("resnet_acc_curve.png", dpi=300)

plt.figure()
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("ResNet Training Loss")
plt.savefig("resnet_loss_curve.png", dpi=300)


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

print("\n[+] Evaluating on test set…")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
print(f"Test loss      = {test_loss:.4f}")
print(f"Test accuracy  = {test_acc:.4f}")


# ---------------------------------------------------------------------
# Confusion matrix + report
# ---------------------------------------------------------------------

y_pred = (model.predict(X_test) > 0.5).astype(int)

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=3))

print("\n[+] Training complete. Models and plots saved.")
