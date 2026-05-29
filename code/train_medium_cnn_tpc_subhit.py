#!/usr/bin/env python3
"""
Medium FPGA-friendly CNN for TPC sub-hitmaps.

Input:
    out_sub/train.h5
    out_sub/val.h5
    out_sub/test.h5

Outputs:
    models_medium_cnn/
    plots_medium_cnn/
    weights_medium_cnn/
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
)

# ============================================================
# 0. Setup
# ============================================================

SEED = 12345
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_DIR = Path("out_sub")

MODEL_DIR = Path("models_medium_cnn")
PLOT_DIR = Path("plots_medium_cnn")
WEIGHT_DIR = Path("weights_medium_cnn")

MODEL_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)
WEIGHT_DIR.mkdir(exist_ok=True)


# ============================================================
# 1. Load Data
# ============================================================

def load_split(path):
    with h5py.File(path, "r") as f:
        X = f["X"][...].astype(np.float32)
        y = f["y"][...].astype(np.int32)

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


print("\nClass balance:")
print("Train:", np.bincount(y_train))
print("Val  :", np.bincount(y_val))
print("Test :", np.bincount(y_test))

print("\nSample labels:")
print("Train y[:20]:", y_train[:20])
print("Val   y[:20]:", y_val[:20])

print("\nValue range:")
print("Train:", X_train.min(), X_train.max())
print("Val  :", X_val.min(), X_val.max())

# ============================================================
# 2. Normalize
# ============================================================

scale = np.max(X_train)

if scale > 0:
    X_train = X_train / scale
    X_val = X_val / scale
    X_test = X_test / scale

print(f"[+] Normalization scale = {scale}")


# ============================================================
# 3. Build Medium CNN
# ============================================================

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(20, 256, 1), name="input"),

    tf.keras.layers.Conv2D(
        filters=8,
        kernel_size=(3, 7),
        padding="same",
        activation="relu",
        name="conv1",
    ),
    tf.keras.layers.BatchNormalization(name="bn1"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool1"),

    tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=(3, 11),
        padding="same",
        activation="relu",
        name="conv2",
    ),
    tf.keras.layers.BatchNormalization(name="bn2"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 4), name="pool2"),

    tf.keras.layers.Conv2D(
        filters=24,
        kernel_size=(3, 11),
        padding="same",
        activation="relu",
        name="conv3",
    ),
    tf.keras.layers.BatchNormalization(name="bn3"),
    tf.keras.layers.MaxPooling2D(pool_size=(1, 4), name="pool3"),

    tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool"),

    tf.keras.layers.Dense(16, activation="relu", name="dense1"),
    tf.keras.layers.Dense(1, activation="sigmoid", name="output"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ],
)

model.summary()

with open(MODEL_DIR / "medium_cnn_summary.txt", "w") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))


# ============================================================
# 4. Train
# ============================================================

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    MODEL_DIR / "medium_cnn_subhit_best.h5",
    monitor="val_auc",
    mode="max",
    save_best_only=True,
    verbose=1,
)

earlystop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=12,
    restore_best_weights=True,
    verbose=1,
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=4,
    min_lr=1e-6,
    verbose=1,
)

print("\n[+] Training medium CNN...\n")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=120,
    batch_size=64,
    callbacks=[checkpoint, earlystop, reduce_lr],
    verbose=1,
)

model.save(MODEL_DIR / "medium_cnn_subhit_final.h5")


# ============================================================
# 5. Plot Training Curves
# ============================================================

def plot_curve(key, ylabel, filename):
    plt.figure()
    plt.plot(history.history[key], label=f"train_{key}")
    plt.plot(history.history[f"val_{key}"], label=f"val_{key}")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"Medium CNN {ylabel}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=200)
    plt.close()


plot_curve("accuracy", "Accuracy", "medium_cnn_accuracy.png")
plot_curve("loss", "Loss", "medium_cnn_loss.png")
plot_curve("precision", "Precision", "medium_cnn_precision.png")
plot_curve("recall", "Recall", "medium_cnn_recall.png")
plot_curve("auc", "AUC", "medium_cnn_auc.png")


# ============================================================
# 6. Test Evaluation
# ============================================================

print("\n[+] Evaluating on test set...\n")

test_results = model.evaluate(X_test, y_test, verbose=0)
metric_names = model.metrics_names

print("Raw Keras test results:")
for name, value in zip(metric_names, test_results):
    print(f"{name}: {value}")

y_score = model.predict(X_test, verbose=0).reshape(-1)


# ============================================================
# 7. Threshold Scan
# ============================================================

threshold_rows = []

best_f1 = None
best_high_recall = None

for thr in np.linspace(0.05, 0.95, 91):
    y_pred = (y_score >= thr).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    row = {
        "threshold": float(thr),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }
    threshold_rows.append(row)

    if best_f1 is None or f1 > best_f1["f1"]:
        best_f1 = row

    if rec >= 0.90:
        if best_high_recall is None or f1 > best_high_recall["f1"]:
            best_high_recall = row


print("\nBest threshold by F1:")
print(best_f1)

print("\nBest threshold with recall >= 0.90:")
print(best_high_recall)


# ============================================================
# 8. Final Metrics at Thresholds
# ============================================================

def evaluate_at_threshold(thr, tag):
    y_pred = (y_score >= thr).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n[{tag}] threshold = {thr:.3f}")
    print("accuracy =", acc)
    print("precision =", prec)
    print("recall =", rec)
    print("f1 =", f1)
    print("confusion matrix:")
    print(cm)

    report = classification_report(y_test, y_pred, digits=4)

    with open(MODEL_DIR / f"medium_cnn_metrics_{tag}.txt", "w") as f:
        f.write(f"Medium CNN Metrics: {tag}\n")
        f.write("============================\n\n")
        f.write(f"Normalization scale = {scale}\n")
        f.write(f"Threshold = {thr:.6f}\n\n")
        f.write(f"Accuracy  = {acc:.6f}\n")
        f.write(f"Precision = {prec:.6f}\n")
        f.write(f"Recall    = {rec:.6f}\n")
        f.write(f"F1        = {f1:.6f}\n\n")
        f.write("Confusion matrix:\n")
        f.write(str(cm))
        f.write("\n\nClassification report:\n")
        f.write(report)

    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title(f"Medium CNN Confusion Matrix ({tag})")
    plt.colorbar()
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"medium_cnn_confusion_matrix_{tag}.png", dpi=200)
    plt.close()

    return acc, prec, rec, f1, cm


evaluate_at_threshold(0.5, "threshold_0p50")
evaluate_at_threshold(best_f1["threshold"], "best_f1")

if best_high_recall is not None:
    evaluate_at_threshold(best_high_recall["threshold"], "high_recall")


# ============================================================
# 9. Save Threshold Scan
# ============================================================

with open(MODEL_DIR / "medium_cnn_threshold_scan.csv", "w") as f:
    f.write("threshold,accuracy,precision,recall,f1\n")
    for r in threshold_rows:
        f.write(
            f"{r['threshold']:.6f},"
            f"{r['accuracy']:.6f},"
            f"{r['precision']:.6f},"
            f"{r['recall']:.6f},"
            f"{r['f1']:.6f}\n"
        )


# ============================================================
# 10. ROC Curve
# ============================================================

fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Medium CNN ROC Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / "medium_cnn_roc_curve.png", dpi=200)
plt.close()

with open(MODEL_DIR / "medium_cnn_auc.txt", "w") as f:
    f.write(f"ROC AUC = {roc_auc:.6f}\n")


# ============================================================
# 11. Score Distribution
# ============================================================

scores_0 = y_score[y_test == 0]
scores_1 = y_score[y_test == 1]

plt.figure()
plt.hist(scores_0, bins=40, alpha=0.6, label="True 0")
plt.hist(scores_1, bins=40, alpha=0.6, label="True 1")
plt.xlabel("CNN output score")
plt.ylabel("Events")
plt.title("Medium CNN Score Distribution")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / "medium_cnn_score_distribution.png", dpi=200)
plt.close()


# ============================================================
# 12. Save Predictions
# ============================================================

with open(MODEL_DIR / "medium_cnn_test_predictions.csv", "w") as f:
    f.write("event_index,true_label,score,pred_0p50,pred_best_f1\n")
    best_thr = best_f1["threshold"]

    for i, (yt, score) in enumerate(zip(y_test, y_score)):
        pred_05 = int(score >= 0.5)
        pred_best = int(score >= best_thr)
        f.write(f"{i},{int(yt)},{score:.8f},{pred_05},{pred_best}\n")


# ============================================================
# 13. Save Weights to Text Files
# ============================================================

all_weights_path = WEIGHT_DIR / "medium_cnn_all_weights.txt"

with open(all_weights_path, "w") as f_all:
    f_all.write("Medium CNN Weights\n")
    f_all.write("==================\n\n")

    for layer in model.layers:
        weights = layer.get_weights()

        if not weights:
            continue

        f_all.write(f"Layer: {layer.name}\n")
        f_all.write(f"Type : {layer.__class__.__name__}\n")

        for idx, arr in enumerate(weights):
            f_all.write(f"\nArray {idx}, shape {arr.shape}\n")
            flat = arr.flatten()

            for i, val in enumerate(flat):
                f_all.write(f"{val:.8e}")
                if (i + 1) % 8 == 0:
                    f_all.write("\n")
                else:
                    f_all.write(" ")

            f_all.write("\n\n")

            layer_file = WEIGHT_DIR / f"{layer.name}_array{idx}_shape_{'_'.join(map(str, arr.shape))}.txt"
            np.savetxt(layer_file, flat, fmt="%.8e")

        f_all.write("-" * 80 + "\n\n")


# ============================================================
# 14. Save Normalization Info
# ============================================================

with open(MODEL_DIR / "medium_cnn_normalization.txt", "w") as f:
    f.write(f"scale = {scale:.10f}\n")
    f.write("Input was divided by this scale during training/testing.\n")


print("\n[+] Done.")
print(f"[+] Models saved in:  {MODEL_DIR}")
print(f"[+] Plots saved in:   {PLOT_DIR}")
print(f"[+] Weights saved in: {WEIGHT_DIR}")