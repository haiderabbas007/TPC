#!/usr/bin/env python3
"""
Module: tpc_resnet_light_detector
---------------------------------
Train a lightweight residual/se–enhanced CNN classifier on sub-hitmap TPC data.

Pipeline:
1. Load train/val/test datasets from HDF5.
2. Construct a compact CNN with residual and squeeze–excitation (SE) blocks.
3. Apply imbalance-aware training (class weighting).
4. Train with adaptive LR + early stopping + checkpoint saving.
5. Plot accuracy/loss trends and evaluate on test set.
6. Print confusion matrix and precision/recall/F1 report.

Outputs:
    best_detector_light.h5
    resnet_light_final.h5
    resnet_light_acc.png
    resnet_light_loss.png
"""

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, ReLU, Add,
    GlobalAveragePooling2D, Dense, Reshape, Multiply,
    MaxPooling2D, Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Dataset Loading
# ----------------------------------------------------------------------

print("[+] Loading dataset…")

with h5py.File("out_sub/train.h5", "r") as f:
    Xtr = f["X"][:]
    ytr = f["y"][:]
with h5py.File("out_sub/val.h5", "r") as f:
    Xva = f["X"][:]
    yva = f["y"][:]
with h5py.File("out_sub/test.h5", "r") as f:
    Xte = f["X"][:]
    yte = f["y"][:]

print("Train:", Xtr.shape, ytr.shape)
print("Val  :", Xva.shape, yva.shape)
print("Test :", Xte.shape, yte.shape)

input_shape = Xtr.shape[1:]


# ----------------------------------------------------------------------
# Network Components
# ----------------------------------------------------------------------

def res_block_light(x: tf.Tensor, filters: int) -> tf.Tensor:
    """
    Lightweight residual block: two 3×3 convs + identity shortcut correction.

    Parameters
    ----------
    x : tf.Tensor
        Input feature map.

    filters : int
        Output channel dimension.

    Returns
    -------
    tf.Tensor
        Output feature map after skip connection + activation.
    """
    shortcut = x

    x = Conv2D(filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)

    if shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), padding="same")(shortcut)

    return ReLU()(Add()([shortcut, x]))


def squeeze_excite(x: tf.Tensor, reduction: int = 8) -> tf.Tensor:
    """
    Squeeze–Excitation module for channel-wise reweighting.

    Parameters
    ----------
    x : tf.Tensor
        Input feature tensor.

    reduction : int
        Channel reduction factor in bottleneck layer.

    Returns
    -------
    tf.Tensor
        Channel–scaled output tensor.
    """
    filters = x.shape[-1]

    se = GlobalAveragePooling2D()(x)
    se = Dense(filters // reduction, activation="relu")(se)
    se = Dense(filters, activation="sigmoid")(se)
    se = Reshape((1, 1, filters))(se)

    return Multiply()([x, se])


def build_best_tpc_light(input_shape: tuple[int, int, int]) -> Model:
    """
    Construct lightweight ResNet-style model with SE attention.

    Parameters
    ----------
    input_shape : tuple[int,int,int]
        Input dimensions (H, W, C).

    Returns
    -------
    keras.Model
    """
    inp = Input(shape=input_shape)

    x = Conv2D(16, (5, 5), padding="same")(inp)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D((1, 2))(x)

    x = res_block_light(x, 32)
    x = res_block_light(x, 32)
    x = squeeze_excite(x)

    x = res_block_light(x, 48)
    x = MaxPooling2D((1, 2))(x)
    x = res_block_light(x, 48)

    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.4)(x)

    out = Dense(1, activation="sigmoid")(x)

    return Model(inp, out)


# ----------------------------------------------------------------------
# Compile Model
# ----------------------------------------------------------------------

model = build_best_tpc_light(input_shape)
model.summary()

loss_fn = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=loss_fn,
    metrics=["accuracy"]
)

# Class imbalance handling
n0, n1 = np.sum(ytr == 0), np.sum(ytr == 1)
class_weight = {0: 1.0, 1: float(n0 / n1)}

print("\nUsing class weights:", class_weight)

callbacks = [
    EarlyStopping(
        monitor="val_accuracy",
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        "best_detector_light.h5",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-5,
        verbose=1
    )
]


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

history = model.fit(
    Xtr, ytr,
    validation_data=(Xva, yva),
    epochs=25,
    batch_size=64,
    callbacks=callbacks,
    class_weight=class_weight,
    verbose=1
)

model.save("resnet_light_final.h5")


# ----------------------------------------------------------------------
# Plot Curves
# ----------------------------------------------------------------------

plt.figure()
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig("resnet_light_acc.png")

plt.figure()
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("resnet_light_loss.png")


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------

print("\n[+] Evaluating on test set…")
test_loss, test_acc = model.evaluate(Xte, yte, verbose=1)
print(f"Test loss     = {test_loss:.4f}")
print(f"Test accuracy = {test_acc:.4f}")

pred = (model.predict(Xte) > 0.5).astype(int).flatten()

cm = confusion_matrix(yte, pred)
print("\nConfusion Matrix:\n", cm)

print("\nClassification Report:\n",
      classification_report(yte, pred, digits=3))

print("\n[+] Training complete. Saved models and plots.")
