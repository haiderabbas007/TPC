#!/usr/bin/env python3

from pathlib import Path
import h5py
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score
)

MODEL_PATH = Path("models_medium_cnn/medium_cnn_subhit_best.h5")
TEST_PATH = Path("out_sub/test.h5")
NORM_PATH = Path("models_medium_cnn/medium_cnn_normalization.txt")

def load_split(path):
    with h5py.File(path, "r") as f:
        X = f["X"][...].astype(np.float32)
        y = f["y"][...].astype(np.int32)
    if X.ndim == 3:
        X = X[..., np.newaxis]
    return X, y

def read_scale(path):
    # expects line like: scale = 0.9490740895
    with open(path, "r") as f:
        line = f.readline()
    return float(line.split("=")[1].strip())

print("[+] Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("[+] Loading test data...")
X_test, y_test = load_split(TEST_PATH)

scale = read_scale(NORM_PATH)
X_test = X_test / scale

print("Test:", X_test.shape, y_test.shape)
print("Class balance:", np.bincount(y_test))

print("[+] Predicting...")
scores = model.predict(X_test, verbose=1).reshape(-1)

auc = roc_auc_score(y_test, scores)
print(f"\nROC AUC = {auc:.6f}")

best_f1 = None
best_recall90 = None

print("\nThreshold scan:")
print("thr, acc, prec, rec, f1")

for thr in np.linspace(0.05, 0.95, 91):
    pred = (scores >= thr).astype(int)

    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred, zero_division=0)
    rec = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)

    print(f"{thr:.2f}, {acc:.4f}, {prec:.4f}, {rec:.4f}, {f1:.4f}")

    row = (thr, acc, prec, rec, f1)

    if best_f1 is None or f1 > best_f1[4]:
        best_f1 = row

    if rec >= 0.90:
        if best_recall90 is None or f1 > best_recall90[4]:
            best_recall90 = row

def report_at(thr, name):
    pred = (scores >= thr).astype(int)
    cm = confusion_matrix(y_test, pred)

    print(f"\n==============================")
    print(f"{name}")
    print(f"threshold = {thr:.4f}")
    print("==============================")
    print("Confusion matrix:")
    print(cm)
    print()
    print(classification_report(y_test, pred, digits=4))

print("\nBest by F1:")
print("thr, acc, prec, rec, f1 =", best_f1)

if best_recall90:
    print("\nBest with recall >= 0.90:")
    print("thr, acc, prec, rec, f1 =", best_recall90)

report_at(0.50, "Default threshold 0.50")
report_at(best_f1[0], "Best F1 threshold")

if best_recall90:
    report_at(best_recall90[0], "Best high-recall threshold")