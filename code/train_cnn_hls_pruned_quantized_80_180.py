#!/usr/bin/env python3
"""
train_cnn_hls_pruned_quantized_80_180.py

FPGA-friendlier replacement for train_cnn_hls_light_80_180.py.

Main changes relative to the previous model:
  1. Smaller CNN: Conv filters 2 -> 4 instead of 4 -> 8.
  2. Smaller dense layer: 4 units instead of 8.
  3. Quantized QKeras layers: low-bit weights, biases, and activations.
  4. Magnitude pruning during training.
  5. Pruning wrappers are stripped before saving the final hls4ml-ready model.

Expected input:
  out_sub_80_180/train.h5
  out_sub_80_180/val.h5
  out_sub_80_180/test.h5

Each file should contain:
  X : event images, expected shape (N, 20, 100) or (N, 20, 100, 1)
  y : labels, expected shape (N,)
"""

from pathlib import Path
import sys
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

try:
    import tensorflow_model_optimization as tfmot
except ImportError:
    print("\n[ERROR] tensorflow_model_optimization is not installed.")
    print("Install it with:")
    print("    pip install tensorflow-model-optimization")
    sys.exit(1)

try:
    from qkeras import QConv2D, QDense, QActivation
except ImportError:
    print("\n[ERROR] qkeras is not installed.")
    print("Install it with:")
    print("    pip install qkeras")
    sys.exit(1)


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

MODEL_DIR = Path("models_hls_pruned_quantized_cnn_80_180")
PLOT_DIR = Path("plots_hls_pruned_quantized_cnn_80_180")
WEIGHT_DIR = Path("weights_hls_pruned_quantized_cnn_80_180")

MODEL_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)
WEIGHT_DIR.mkdir(exist_ok=True)


# ============================================================
# User-adjustable knobs
# ============================================================

INPUT_SHAPE = (20, 100, 1)

EPOCHS = 120
BATCH_SIZE = 64
LEARNING_RATE = 5e-4

# Strong pruning target. If accuracy collapses, reduce FINAL_SPARSITY to 0.50.
INITIAL_SPARSITY = 0.00
FINAL_SPARSITY = 0.70
PRUNING_BEGIN_EPOCH = 5
PRUNING_END_EPOCH = 70

# QKeras quantization.
# 6-bit is a good first target. If accuracy is bad, try 8-bit.
WEIGHT_QUANTIZER = "quantized_bits(6,1,alpha=1)"
BIAS_QUANTIZER = "quantized_bits(6,1,alpha=1)"
ACTIVATION_QUANTIZER = "quantized_relu(6,2)"


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
        f"Expected input shape {INPUT_SHAPE}, but got {X_train.shape[1:]}. "
        "Check whether the crop is really 80:180, giving 20x100."
    )

scale = np.max(X_train)

if scale > 0:
    X_train = X_train / scale
    X_val = X_val / scale
    X_test = X_test / scale

print(f"\n[+] Normalization scale = {scale}")


# ============================================================
# Build smaller quantized model
# ============================================================

def build_quantized_light_model():
    """
    Much smaller than the previous model:

    Old:
      Conv2D 4 filters -> Conv2D 8 filters -> Dense 8

    New:
      QConv2D 2 filters -> QConv2D 4 filters -> QDense 4

    This reduces intermediate feature-map channels and final dense cost.
    """

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=INPUT_SHAPE, name="input_layer"),

            QConv2D(
                2,
                (3, 5),
                padding="same",
                use_bias=True,
                kernel_quantizer=WEIGHT_QUANTIZER,
                bias_quantizer=BIAS_QUANTIZER,
                name="qconv1",
            ),
            QActivation(ACTIVATION_QUANTIZER, name="qrelu1"),
            tf.keras.layers.MaxPooling2D((2, 2), name="pool1"),

            QConv2D(
                4,
                (3, 7),
                padding="same",
                use_bias=True,
                kernel_quantizer=WEIGHT_QUANTIZER,
                bias_quantizer=BIAS_QUANTIZER,
                name="qconv2",
            ),
            QActivation(ACTIVATION_QUANTIZER, name="qrelu2"),
            tf.keras.layers.MaxPooling2D((2, 4), name="pool2"),

            tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool"),

            QDense(
                4,
                use_bias=True,
                kernel_quantizer=WEIGHT_QUANTIZER,
                bias_quantizer=BIAS_QUANTIZER,
                name="qdense1",
            ),
            QActivation(ACTIVATION_QUANTIZER, name="qrelu_dense1"),

            # Linear logit output for HLS.
            # BinaryCrossentropy uses from_logits=True.
            QDense(
                1,
                use_bias=True,
                kernel_quantizer=WEIGHT_QUANTIZER,
                bias_quantizer=BIAS_QUANTIZER,
                name="output",
            ),
        ],
        name="hls_pruned_quantized_light_cnn_80_180",
    )

    return model


base_model = build_quantized_light_model()

print("\n[+] Base quantized model summary:")
base_model.summary()

with open(MODEL_DIR / "base_quantized_model_summary.txt", "w") as f:
    base_model.summary(print_fn=lambda x: f.write(x + "\n"))


# ============================================================
# Apply pruning
# ============================================================

steps_per_epoch = int(np.ceil(len(X_train) / BATCH_SIZE))
begin_step = PRUNING_BEGIN_EPOCH * steps_per_epoch
end_step = PRUNING_END_EPOCH * steps_per_epoch

pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
    initial_sparsity=INITIAL_SPARSITY,
    final_sparsity=FINAL_SPARSITY,
    begin_step=begin_step,
    end_step=end_step,
)

def prune_layer(layer):
    # Prune only layers with weights. Do not prune pooling/activation/input layers.
    if isinstance(layer, (QConv2D, QDense)):
        return tfmot.sparsity.keras.prune_low_magnitude(
            layer,
            pruning_schedule=pruning_schedule,
        )
    return layer


pruned_model = tf.keras.models.clone_model(
    base_model,
    clone_function=prune_layer,
)

pruned_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="acc_at_logit_0", threshold=0.0),
        tf.keras.metrics.AUC(name="auc", from_logits=True),
        tf.keras.metrics.Precision(name="precision_at_logit_0", thresholds=0.0),
        tf.keras.metrics.Recall(name="recall_at_logit_0", thresholds=0.0),
    ],
)

print("\n[+] Pruned + quantized model summary:")
pruned_model.summary()

with open(MODEL_DIR / "pruned_quantized_model_summary.txt", "w") as f:
    pruned_model.summary(print_fn=lambda x: f.write(x + "\n"))


# ============================================================
# Training
# ============================================================

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=str(MODEL_DIR / "best_pruned_quantized_weights.h5"),
    monitor="val_auc",
    mode="max",
    save_best_only=True,
    save_weights_only=True,
    verbose=1,
)

earlystop = tf.keras.callbacks.EarlyStopping(
    monitor="val_auc",
    mode="max",
    patience=18,
    restore_best_weights=True,
    verbose=1,
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_auc",
    mode="max",
    factor=0.5,
    patience=6,
    min_lr=1e-6,
    verbose=1,
)

pruning_callback = tfmot.sparsity.keras.UpdatePruningStep()

print("\n[+] Training pruned + quantized lightweight CNN on 20x100 cropped input...\n")

history = pruned_model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[
        pruning_callback,
        checkpoint,
        earlystop,
        reduce_lr,
    ],
    verbose=1,
)

# Load the best weights according to validation AUC, if saved.
best_weights = MODEL_DIR / "best_pruned_quantized_weights.h5"
if best_weights.exists():
    print(f"\n[+] Loading best weights from {best_weights}")
    pruned_model.load_weights(str(best_weights))


# ============================================================
# Strip pruning wrappers for hls4ml
# ============================================================

print("\n[+] Stripping pruning wrappers for hls4ml conversion...")

stripped_model = tfmot.sparsity.keras.strip_pruning(pruned_model)

stripped_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="acc_at_logit_0", threshold=0.0),
        tf.keras.metrics.AUC(name="auc", from_logits=True),
        tf.keras.metrics.Precision(name="precision_at_logit_0", thresholds=0.0),
        tf.keras.metrics.Recall(name="recall_at_logit_0", thresholds=0.0),
    ],
)

print("\n[+] Stripped hls4ml-ready model summary:")
stripped_model.summary()

with open(MODEL_DIR / "stripped_hls_ready_model_summary.txt", "w") as f:
    stripped_model.summary(print_fn=lambda x: f.write(x + "\n"))

# Save both formats. hls4ml usually works well with .h5 Keras models.
stripped_model.save(MODEL_DIR / "hls_pruned_quantized_light_cnn_80_180_stripped.h5")
stripped_model.save(MODEL_DIR / "hls_pruned_quantized_light_cnn_80_180_stripped.keras")


# ============================================================
# Sparsity report
# ============================================================

def layer_sparsity_report(model, path):
    lines = []
    total_weights = 0
    total_zeros = 0

    for layer in model.layers:
        weights = layer.get_weights()
        if not weights:
            continue

        layer_total = 0
        layer_zeros = 0

        for arr in weights:
            layer_total += arr.size
            layer_zeros += np.sum(arr == 0)

        total_weights += layer_total
        total_zeros += layer_zeros

        sparsity = layer_zeros / layer_total if layer_total else 0.0
        lines.append(
            f"{layer.name:20s} {layer.__class__.__name__:20s} "
            f"zeros={layer_zeros:8d} total={layer_total:8d} sparsity={sparsity:.4f}"
        )

    global_sparsity = total_zeros / total_weights if total_weights else 0.0

    with open(path, "w") as f:
        f.write("Layer sparsity report after strip_pruning\n")
        f.write("=========================================\n\n")
        for line in lines:
            f.write(line + "\n")
        f.write("\n")
        f.write(f"global_zeros = {total_zeros}\n")
        f.write(f"global_total = {total_weights}\n")
        f.write(f"global_sparsity = {global_sparsity:.6f}\n")

    print("\n[+] Sparsity report:")
    for line in lines:
        print(line)
    print(f"Global sparsity = {global_sparsity:.4f}")


layer_sparsity_report(
    stripped_model,
    MODEL_DIR / "sparsity_report.txt",
)


# ============================================================
# Plot training curves
# ============================================================

def plot_history_metric(key, ylabel, filename):
    if key not in history.history:
        print(f"[WARN] metric {key} not found in history; skipping plot.")
        return

    val_key = f"val_{key}"
    if val_key not in history.history:
        print(f"[WARN] metric {val_key} not found in history; skipping plot.")
        return

    plt.figure()
    plt.plot(history.history[key], label=f"train_{key}")
    plt.plot(history.history[val_key], label=f"val_{key}")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"Pruned Quantized HLS CNN 80-180 {ylabel}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=200)
    plt.close()


plot_history_metric("loss", "Loss", "loss_curve.png")
plot_history_metric("auc", "AUC", "auc_curve.png")
plot_history_metric("acc_at_logit_0", "Accuracy at logit threshold 0", "accuracy_at_logit_0.png")
plot_history_metric("precision_at_logit_0", "Precision at logit threshold 0", "precision_at_logit_0.png")
plot_history_metric("recall_at_logit_0", "Recall at logit threshold 0", "recall_at_logit_0.png")


# ============================================================
# Evaluation helpers
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


def plot_threshold_scan(rows, filename, title):
    thrs = [r["threshold"] for r in rows]
    accs = [r["accuracy"] for r in rows]
    precs = [r["precision"] for r in rows]
    recs = [r["recall"] for r in rows]
    f1s = [r["f1"] for r in rows]

    plt.figure()
    plt.plot(thrs, accs, label="accuracy")
    plt.plot(thrs, precs, label="precision")
    plt.plot(thrs, recs, label="recall")
    plt.plot(thrs, f1s, label="f1")
    plt.xlabel("Decision threshold after sigmoid")
    plt.ylabel("Metric")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=200)
    plt.close()


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

    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title(f"Confusion Matrix: {tag}")
    plt.colorbar()
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"confusion_matrix_{tag}.png", dpi=200)
    plt.close()

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


# ============================================================
# Evaluate stripped model
# ============================================================

print("\n[+] Predicting validation scores using stripped hls4ml-ready model...")

val_logits = stripped_model.predict(X_val, verbose=0).reshape(-1)
val_scores = sigmoid(val_logits)
val_auc = roc_auc_score(y_val, val_scores)

val_rows, val_best_f1, val_best_high_recall = threshold_scan(y_val, val_scores)

print("\nValidation ROC AUC =", val_auc)
print("Validation best F1 threshold:", val_best_f1)
print("Validation best high-recall threshold:", val_best_high_recall)

save_threshold_scan(val_rows, MODEL_DIR / "validation_threshold_scan.csv")
plot_threshold_scan(
    val_rows,
    "validation_threshold_scan.png",
    "Validation Metrics vs Threshold",
)

chosen_threshold = val_best_f1["threshold"]

with open(MODEL_DIR / "chosen_threshold.txt", "w") as f:
    f.write(f"chosen_threshold_from_validation_best_f1 = {chosen_threshold:.6f}\n")
    f.write(f"validation_auc = {val_auc:.6f}\n")
    f.write(str(val_best_f1) + "\n")


print("\n[+] Predicting test scores using stripped hls4ml-ready model...")

test_logits = stripped_model.predict(X_test, verbose=0).reshape(-1)
test_scores = sigmoid(test_logits)
test_auc = roc_auc_score(y_test, test_scores)

test_rows, test_best_f1, test_best_high_recall = threshold_scan(y_test, test_scores)

print("\nTest ROC AUC =", test_auc)
print("Test best F1 threshold:", test_best_f1)
print("Test best high-recall threshold:", test_best_high_recall)

save_threshold_scan(test_rows, MODEL_DIR / "test_threshold_scan.csv")
plot_threshold_scan(
    test_rows,
    "test_threshold_scan_diagnostic_only.png",
    "Test Metrics vs Threshold",
)

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
# Save predictions and weights
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


all_weights_path = WEIGHT_DIR / "hls_pruned_quantized_light_cnn_80_180_all_weights.txt"

with open(all_weights_path, "w") as f_all:
    f_all.write("HLS Pruned Quantized Light CNN 80-180 Weights\n")
    f_all.write("============================================\n\n")

    for layer in stripped_model.layers:
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


with open(MODEL_DIR / "normalization.txt", "w") as f:
    f.write(f"scale = {scale:.10f}\n")
    f.write("Input was divided by this scale during training/testing.\n")
    f.write("Input crop: original time bins 80:180, shape 20x100.\n")


with open(MODEL_DIR / "final_summary.txt", "w") as f:
    f.write("Final Summary\n")
    f.write("=============\n\n")
    f.write("Model: HLS pruned quantized light CNN 80-180 crop\n")
    f.write("Input shape: 20x100x1\n")
    f.write("Crop: original columns 80:180\n\n")
    f.write("Architecture:\n")
    f.write("QConv2D(2, 3x5) -> QActivation -> MaxPool(2x2)\n")
    f.write("QConv2D(4, 3x7) -> QActivation -> MaxPool(2x4)\n")
    f.write("GlobalAveragePooling2D -> QDense(4) -> QActivation -> QDense(1 linear logit)\n\n")
    f.write(f"Weight quantizer = {WEIGHT_QUANTIZER}\n")
    f.write(f"Bias quantizer = {BIAS_QUANTIZER}\n")
    f.write(f"Activation quantizer = {ACTIVATION_QUANTIZER}\n")
    f.write(f"Initial sparsity = {INITIAL_SPARSITY:.3f}\n")
    f.write(f"Final target sparsity = {FINAL_SPARSITY:.3f}\n\n")
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
print(f"[+] hls4ml-ready stripped model:")
print(f"    {MODEL_DIR / 'hls_pruned_quantized_light_cnn_80_180_stripped.h5'}")
print(f"[+] Models saved in:  {MODEL_DIR}")
print(f"[+] Plots saved in:   {PLOT_DIR}")
print(f"[+] Weights saved in: {WEIGHT_DIR}")
print(f"[+] Chosen threshold from validation best F1 = {chosen_threshold:.4f}")
