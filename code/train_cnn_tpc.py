"""
Module: cnn_tpc_trainer
-----------------------
Train and evaluate a lightweight 2D CNN classifier on processed TPC hitmap
datasets stored in HDF5 format.

Workflow:
1. Load train/val/test datasets from disk.
2. Wrap them in tf.data pipelines to mitigate memory usage.
3. Build a compact CNN with aggressive pooling suited for elongated inputs.
4. Train with early stopping & model checkpointing.
5. Save training curves and final/best models.

Expected input directory structure:
    out/
      train.h5
      val.h5
      test.h5
"""

import h5py
import numpy as np
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


BASE = Path(__file__).resolve().parent / "out"


def load_h5(name: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load feature and label datasets from compressed HDF5.

    Parameters
    ----------
    name : str
        Base file name (without .h5 extension) in `BASE`.

    Returns
    -------
    (X, y) : tuple of np.ndarray
        Feature tensor in float16 and labels.
    """
    path = BASE / f"{name}.h5"
    with h5py.File(path, "r") as f:
        X = f["X"][...]
        y = f["y"][...]
    return X.astype("float16"), y


def make_dataset(X: np.ndarray, y: np.ndarray,
                 batch_size: int = 2,
                 shuffle: bool = True) -> tf.data.Dataset:
    """
    Wrap arrays into a TensorFlow dataset with batching and prefetch.

    Parameters
    ----------
    X : np.ndarray
        Feature tensor.

    y : np.ndarray
        Label tensor.

    batch_size : int
        Batch size (kept small to avoid OOM).

    shuffle : bool
        Whether to shuffle each epoch.

    Returns
    -------
    tf.data.Dataset
    """
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(len(X), reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_model(input_shape: tuple[int, int, int]) -> tf.keras.Model:
    """
    Construct and compile a compact CNN for binary classification.

    Parameters
    ----------
    input_shape : tuple[int, int, int]
        Input tensor shape e.g. (20, 10000, 1).

    Returns
    -------
    tf.keras.Model
    """
    model = Sequential([
        Conv2D(8, (3, 9), activation="relu", padding="same",
               input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 10)),
        Dropout(0.2),

        Conv2D(16, (3, 9), activation="relu", padding="same"),
        MaxPooling2D(pool_size=(2, 10)),
        Dropout(0.3),

        Flatten(),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def main() -> None:
    """
    Execute full train/validation/test cycle, generate plots,
    and save final/best models.
    """
    print("[+] Loading data …")
    X_tr, y_tr = load_h5("train")
    X_va, y_va = load_h5("val")
    X_te, y_te = load_h5("test")

    print("train:", X_tr.shape, y_tr.shape, X_tr.dtype)
    print("val  :", X_va.shape, y_va.shape, X_va.dtype)
    print("test :", X_te.shape, y_te.shape, X_te.dtype)

    ds_train = make_dataset(X_tr, y_tr, batch_size=2, shuffle=True)
    ds_val   = make_dataset(X_va, y_va, batch_size=2, shuffle=False)
    ds_test  = make_dataset(X_te, y_te, batch_size=2, shuffle=False)

    input_shape = X_tr.shape[1:]  # (20, 10000, 1)
    model = build_model(input_shape)
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            "cnn_tpc_best.h5",
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False
        ),
    ]

    print("[+] Training …")
    history = model.fit(
        ds_train,
        epochs=30,
        validation_data=ds_val,
        callbacks=callbacks,
        verbose=1,
    )

    hist = history.history

    # Learning curves
    plt.figure()
    plt.plot(hist["loss"], label="train_loss")
    plt.plot(hist["val_loss"], label="val_loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("loss_curve.png", dpi=150)

    plt.figure()
    plt.plot(hist["accuracy"], label="train_acc")
    plt.plot(hist["val_accuracy"], label="val_acc")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig("acc_curve.png", dpi=150)

    print("[+] Evaluating on test set …")
    test_loss, test_acc = model.evaluate(ds_test, verbose=1)
    print(f"Test loss = {test_loss:.4f}, acc = {test_acc:.4f}")

    model.save("cnn_tpc_final.h5")
    print("[+] Saved: cnn_tpc_best.h5, cnn_tpc_final.h5")


if __name__ == "__main__":
    main()