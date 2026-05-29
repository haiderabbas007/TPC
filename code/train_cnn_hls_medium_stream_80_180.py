#!/usr/bin/env python3
"""
train_cnn_hls_medium_stream_80_180.py

Train a less-collapsed FPGA/HLS-friendly CNN for the 80:180 crop.

This replaces the ultra-tiny 229-parameter model that synthesized successfully
but collapsed to nearly constant hls4ml fixed-point scores.

Architecture:
    Input 20x100x1
    Conv2D(4, 3x5) -> ReLU -> MaxPool(2x2)
    Conv2D(8, 3x7) -> ReLU -> MaxPool(2x4)
    GlobalAveragePooling2D
    Dense(8) -> ReLU
    Dense(1 linear logit)

No QKeras, no manual pruning in this first pass.
We want score separation first.
"""

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
    roc_auc_score,
)

# ============================================================
# Reproducibility
# ============================================================

SEED = 12345
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
# Paths
# ============================================================

DATA_DIR = Path("out_sub_80_180")

MODEL_DIR = Path("models_hls_medium_stream_cnn_80_180")
PLOT_DIR = Path("plots_hls_medium_stream_cnn_80_180")
WEIGHT_DIR = Path("weights_hls_medium_stream_cnn_80_180")

MODEL_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)
WEIGHT_DIR.mkdir(exist_ok=True)

# ============================================================
# User knobs
# ============================================================

INPUT_SHAPE = (20, 100, 1)

EPOCHS = 120
BATCH_SIZE = 64
LEARNING_RATE = 5e-4

# ============================================================
# Data loading
# ============================================================

def load_split(path):
    with h5py.File(path, "r") as f:
        X = f["X"][...].astype(np.float32)
        y = f["y"][...].astype(np.int32)

    if X.ndim == 3:
        X = X[..., np.newaxis]

    return X, y


print("[+] Loading cropped 20x100 data...")

X_train, y_train = load_split(DATA_DIR / "train.h5")
X_val, y_val = load_split(DATA_DIR / "val.h5")
X_test, y_test = load_split(DATA_DIR / "test.h5")

print("Train:", X_train.shape, y_train.shape)
print("Val  :", X_val.shape, y_val.shape)
print("Test :", X_test.shape, y_test.shape)

print("\nClass balance:")
print("Train:", np.bincount(y_train))
print("Val  :", np.bincount(y_val))
print("Test :", np.bincount(y_test))

if X_train.shape[1:] != INPUT_SHAPE:
    raise ValueError(
        f"Expected input shape {INPUT_SHAPE}, got {X_train.shape[1:]}. "
        "Check out_sub_80_180 files."
    )

scale = np.max(X_train)

if scale > 0:
    X_train = X_train / scale
    X_val = X_val / scale
    X_test = X_test / scale

print(f"\n[+] Normalization scale = {scale}")

with open(MODEL_DIR / "normalization.txt", "w") as f:
    f.write(f"scale = {scale:.10f}\n")
    f.write("Input was divided by this scale during training/testing.\n")
    f.write("Input crop: original time bins 80:180, shape 20x100.\n")

# ============================================================
# Build model
# ============================================================

def build_medium_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=INPUT_SHAPE, name="input_layer"),

            tf.keras.layers.Conv2D(
                4,
                (3, 5),
                padding="same",
                use_bias=True,
                activation=None,
                name="conv1",
            ),
            tf.keras.layers.ReLU(name="relu1"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool1"),

            tf.keras.layers.Conv2D(
                8,
                (3, 7),
                padding="same",
                use_bias=True,
                activation=None,
                name="conv2",
            ),
            tf.keras.layers.ReLU(name="relu2"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 4), name="pool2"),

            tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool"),

            tf.keras.layers.Dense(8, use_bias=True, activation=None, name="dense1"),
            tf.keras.layers.ReLU(name="relu_dense1"),

            tf.keras.layers.Dense(1, use_bias=True, activation="linear", name="output"),
        ],
        name="hls_medium_stream_cnn_80_180",
    )

    return model


model = build_medium_model()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="acc_at_logit_0", threshold=0.0),
        tf.keras.metrics.AUC(name="auc", from_logits=True),
        tf.keras.metrics.Precision(name="precision_at_logit_0", thresholds=0.0),
        tf.keras.metrics.Recall(name="recall_at_logit_0", thresholds=0.0),
    ],
)

print("\n[+] Medium model summary:")
model.summary()

with open(MODEL_DIR / "model_summary.txt", "w") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))

# ============================================================
# Train
# ============================================================

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=str(MODEL_DIR / "best.weights.h5"),
    monitor="val_auc",
    mode="max",
    save_best_only=True,
    save_weights_only=True,
    verbose=1,
)

earlystop = tf.keras.callbacks.EarlyStopping(
    monitor="val_auc",
    mode="max",
    patience=20,
    restore_best_weights=True,
    verbose=1,
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_auc",
    mode="max",
    factor=0.5,
    patience=7,
    min_lr=1e-6,
    verbose=1,
)

print("\n[+] Training medium stream CNN...\n")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint, earlystop, reduce_lr],
    verbose=1,
)

best_weights = MODEL_DIR / "best.weights.h5"
if best_weights.exists():
    print(f"\n[+] Loading best weights from {best_weights}")
    model.load_weights(str(best_weights))

# ============================================================
# Save model
# ============================================================

h5_path = MODEL_DIR / "hls_medium_stream_cnn_80_180.h5"
keras_path = MODEL_DIR / "hls_medium_stream_cnn_80_180.keras"

model.save(h5_path)
model.save(keras_path)

print(f"\n[+] Saved model:")
print(f"    {h5_path}")
print(f"    {keras_path}")

# ============================================================
# Plot training curves
# ============================================================

def plot_metric(key, ylabel, filename):
    if key not in history.history or f"val_{key}" not in history.history:
        print(f"[WARN] metric {key} not found; skipping.")
        return

    plt.figure()
    plt.plot(history.history[key], label=f"train_{key}")
    plt.plot(history.history[f"val_{key}"], label=f"val_{key}")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"Medium Stream CNN 80-180: {ylabel}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=200)
    plt.close()


plot_metric("loss", "Loss", "loss_curve.png")
plot_metric("auc", "AUC", "auc_curve.png")
plot_metric("acc_at_logit_0", "Accuracy at logit 0", "accuracy_at_logit_0.png")
plot_metric("precision_at_logit_0", "Precision at logit 0", "precision_at_logit_0.png")
plot_metric("recall_at_logit_0", "Recall at logit 0", "recall_at_logit_0.png")

# ============================================================
# Evaluation
# ============================================================

def sigmoid(x):
    x = np.asarray(x)
    return 1.0 / (1.0 + np.exp(-x))


def threshold_scan(y_true, scores):
    rows = []

    for thr in np.linspace(0.01, 0.99, 99):
        pred = (scores >= thr).astype(int)

        acc = accuracy_score(y_true, pred)
        prec = precision_score(y_true, pred, zero_division=0)
        rec = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)

        rows.append(
            {
                "threshold": float(thr),
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
            }
        )

    best_f1 = max(rows, key=lambda r: r["f1"])

    recall_candidates = [r for r in rows if r["recall"] >= 0.90]
    best_high_recall = max(recall_candidates, key=lambda r: r["f1"]) if recall_candidates else None

    return rows, best_f1, best_high_recall


def save_threshold_scan(rows, path):
    with open(path, "w") as f:
        f.write("threshold,accuracy,precision,recall,f1\n")
        for r in rows:
            f.write(
                f"{r['threshold']:.6f},"
                f"{r['accuracy']:.6f},"
                f"{r['precision']:.6f},"
                f"{r['recall']:.6f},"
                f"{r['f1']:.6f}\n"
            )


def evaluate_at_threshold(y_true, scores, threshold, tag):
    pred = (scores >= threshold).astype(int)

    acc = accuracy_score(y_true, pred)
    prec = precision_score(y_true, pred, zero_division=0)
    rec = recall_score(y_true, pred, zero_division=0)
    f1 = f1_score(y_true, pred, zero_division=0)
    cm = confusion_matrix(y_true, pred)

    print(f"\n[{tag}] threshold = {threshold:.4f}")
    print("accuracy  =", acc)
    print("precision =", prec)
    print("recall    =", rec)
    print("f1        =", f1)
    print("confusion matrix:")
    print(cm)

    report = classification_report(y_true, pred, digits=4, zero_division=0)

    with open(MODEL_DIR / f"metrics_{tag}.txt", "w") as f:
        f.write(f"Metrics: {tag}\n")
        f.write("====================\n\n")
        f.write(f"threshold = {threshold:.6f}\n")
        f.write(f"accuracy  = {acc:.6f}\n")
        f.write(f"precision = {prec:.6f}\n")
        f.write(f"recall    = {rec:.6f}\n")
        f.write(f"f1        = {f1:.6f}\n\n")
        f.write("confusion matrix:\n")
        f.write(str(cm))
        f.write("\n\nclassification report:\n")
        f.write(report)

    return acc, prec, rec, f1, cm


def plot_roc(y_true, scores, filename, title):
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=200)
    plt.close()

    return roc_auc


def plot_score_distribution(y_true, scores, threshold, filename, title):
    scores_0 = scores[y_true == 0]
    scores_1 = scores[y_true == 1]

    plt.figure()
    plt.hist(scores_0, bins=40, alpha=0.6, label="True 0")
    plt.hist(scores_1, bins=40, alpha=0.6, label="True 1")
    plt.axvline(threshold, linestyle="--", label=f"threshold = {threshold:.2f}")
    plt.xlabel("CNN output score after sigmoid")
    plt.ylabel("Events")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=200)
    plt.close()


print("\n[+] Evaluating validation set...")
val_logits = model.predict(X_val, verbose=0).reshape(-1)
val_scores = sigmoid(val_logits)
val_auc = roc_auc_score(y_val, val_scores)
val_rows, val_best_f1, val_best_high_recall = threshold_scan(y_val, val_scores)

print("Validation AUC =", val_auc)
print("Validation best F1 threshold:", val_best_f1)
print("Validation best high-recall threshold:", val_best_high_recall)

save_threshold_scan(val_rows, MODEL_DIR / "validation_threshold_scan.csv")
chosen_threshold = val_best_f1["threshold"]

print("\n[+] Evaluating test set...")
test_logits = model.predict(X_test, verbose=0).reshape(-1)
test_scores = sigmoid(test_logits)
test_auc = roc_auc_score(y_test, test_scores)
test_rows, test_best_f1, test_best_high_recall = threshold_scan(y_test, test_scores)

print("Test AUC =", test_auc)
print("Test best F1 threshold diagnostic only:", test_best_f1)
print("Test best high-recall threshold diagnostic only:", test_best_high_recall)

save_threshold_scan(test_rows, MODEL_DIR / "test_threshold_scan.csv")

evaluate_at_threshold(y_test, test_scores, 0.50, "test_threshold_0p50")
evaluate_at_threshold(y_test, test_scores, chosen_threshold, "test_val_chosen_threshold")
evaluate_at_threshold(y_test, test_scores, test_best_f1["threshold"], "test_oracle_best_f1")

plot_roc(y_val, val_scores, "validation_roc_curve.png", "Validation ROC Curve")
plot_roc(y_test, test_scores, "test_roc_curve.png", "Test ROC Curve")
plot_score_distribution(
    y_val,
    val_scores,
    chosen_threshold,
    "validation_score_distribution_with_threshold.png",
    "Validation Score Distribution",
)
plot_score_distribution(
    y_test,
    test_scores,
    chosen_threshold,
    "test_score_distribution_with_val_threshold.png",
    "Test Score Distribution",
)

# ============================================================
# Save predictions / weights / summary
# ============================================================

with open(MODEL_DIR / "test_predictions.csv", "w") as f:
    f.write("event_index,true_label,logit,score,pred_0p50,pred_val_chosen,pred_test_best_f1\n")

    test_best_thr = test_best_f1["threshold"]

    for i, (yt, logit, score) in enumerate(zip(y_test, test_logits, test_scores)):
        pred_05 = int(score >= 0.5)
        pred_val = int(score >= chosen_threshold)
        pred_test_best = int(score >= test_best_thr)
        f.write(
            f"{i},{int(yt)},{logit:.8f},{score:.8f},"
            f"{pred_05},{pred_val},{pred_test_best}\n"
        )

all_weights_path = WEIGHT_DIR / "hls_medium_stream_cnn_80_180_all_weights.txt"

with open(all_weights_path, "w") as f_all:
    f_all.write("HLS Medium Stream CNN 80-180 Weights\n")
    f_all.write("====================================\n\n")

    for layer in model.layers:
        weights = layer.get_weights()

        if not weights:
            continue

        f_all.write(f"Layer: {layer.name}\n")
        f_all.write(f"Type : {layer.__class__.__name__}\n")

        for idx, arr in enumerate(weights):
            f_all.write(f"\nArray {idx}, shape {arr.shape}\n")
            flat = arr.flatten()

            for j, val in enumerate(flat):
                f_all.write(f"{val:.8e}")
                if (j + 1) % 8 == 0:
                    f_all.write("\n")
                else:
                    f_all.write(" ")

            f_all.write("\n\n")

            layer_file = WEIGHT_DIR / f"{layer.name}_array{idx}_shape_{'_'.join(map(str, arr.shape))}.txt"
            np.savetxt(layer_file, flat, fmt="%.8e")

        f_all.write("-" * 80 + "\n\n")

with open(MODEL_DIR / "chosen_threshold.txt", "w") as f:
    f.write(f"chosen_threshold_from_validation_best_f1 = {chosen_threshold:.6f}\n")
    f.write(f"validation_auc = {val_auc:.6f}\n")
    f.write(str(val_best_f1) + "\n")

with open(MODEL_DIR / "final_summary.txt", "w") as f:
    f.write("Final Summary\n")
    f.write("=============\n\n")
    f.write("Model: HLS medium stream CNN 80-180 crop\n")
    f.write("Input shape: 20x100x1\n")
    f.write("Crop: original columns 80:180\n\n")
    f.write("Architecture:\n")
    f.write("Conv2D(4, 3x5) -> ReLU -> MaxPool(2x2)\n")
    f.write("Conv2D(8, 3x7) -> ReLU -> MaxPool(2x4)\n")
    f.write("GlobalAveragePooling2D -> Dense(8) -> ReLU -> Dense(1 linear logit)\n\n")
    f.write("No manual pruning. No QKeras in this pass.\n\n")
    f.write(f"Normalization scale = {scale:.10f}\n")
    f.write(f"Validation AUC = {val_auc:.6f}\n")
    f.write(f"Test AUC       = {test_auc:.6f}\n\n")
    f.write("Validation best F1 threshold:\n")
    f.write(str(val_best_f1) + "\n\n")
    f.write("Validation best high-recall threshold:\n")
    f.write(str(val_best_high_recall) + "\n\n")
    f.write("Test best F1 threshold diagnostic only:\n")
    f.write(str(test_best_f1) + "\n\n")
    f.write("Test best high-recall threshold diagnostic only:\n")
    f.write(str(test_best_high_recall) + "\n\n")
    f.write(f"Chosen threshold from validation = {chosen_threshold:.6f}\n")

print("\n[+] Done.")
print(f"[+] hls4ml-ready model:")
print(f"    {h5_path}")
print(f"[+] Models saved in:  {MODEL_DIR}")
print(f"[+] Plots saved in:   {PLOT_DIR}")
print(f"[+] Weights saved in: {WEIGHT_DIR}")
print(f"[+] Chosen threshold from validation best F1 = {chosen_threshold:.4f}")
