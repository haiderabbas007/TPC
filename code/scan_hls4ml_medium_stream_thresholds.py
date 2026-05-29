#!/usr/bin/env python3
"""
scan_hls4ml_medium_stream_thresholds.py

Check whether the medium stream model has useful hls4ml fixed-point score separation.

Do this BEFORE Vitis/Vivado synthesis.

If scores still collapse to a constant, do not waste time in Vivado.
"""

import numpy as np
import h5py
import hls4ml
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

MODEL_PATH = "models_hls_medium_stream_cnn_80_180/hls_medium_stream_cnn_80_180.h5"
H5_PATH = "out_sub/test.h5"

OUTPUT_DIR = "hls4ml_medium_stream_cnn_80_180_scan_check"

N_EVENTS = 300
CROP_START = 80
CROP_END = 180

NORMALIZATION_SCALE = 0.9490740895271301

INPUT_SHAPE = (20, 100, 1)
PART = "xcu250-figd2104-2L-e"

# input_t for this new model conversion is ap_fixed<10,4>
# fractional bits = 10 - 4 = 6
INPUT_FRAC_BITS = 6
INPUT_SCALE = 2 ** INPUT_FRAC_BITS

# output is ap_fixed<12,5>
# fractional bits = 12 - 5 = 7
OUTPUT_FRAC_BITS = 7
OUTPUT_SCALE = 2 ** OUTPUT_FRAC_BITS


def quantize_input_ap_fixed_10_4(x):
    raw = np.round(x * INPUT_SCALE)
    raw = np.clip(raw, -512, 511)
    return raw / INPUT_SCALE


print("[+] Loading data...")

with h5py.File(H5_PATH, "r") as h5:
    X = h5["X"][:N_EVENTS, :, CROP_START:CROP_END, 0].astype(np.float32)
    y = h5["y"][:N_EVENTS].astype(int)

X = X / NORMALIZATION_SCALE
Xq = quantize_input_ap_fixed_10_4(X)
Xq = Xq.reshape(-1, 20, 100, 1).astype(np.float32)

print("[+] Xq shape:", Xq.shape)
print("[+] y balance:", np.bincount(y))
print("[+] Xq stats:")
print("    min =", Xq.min())
print("    max =", Xq.max())
print("    mean =", Xq.mean())
print("    nonzero =", np.count_nonzero(Xq))

print("[+] Loading Keras model...")

old_model = tf.keras.models.load_model(MODEL_PATH, compile=False)

new_input = tf.keras.Input(shape=INPUT_SHAPE, name="input_layer")
z = new_input

for layer in old_model.layers:
    if isinstance(layer, tf.keras.layers.InputLayer):
        continue
    z = layer(z)

model = tf.keras.Model(inputs=new_input, outputs=z)

print("[+] Creating hls4ml config...")

config = hls4ml.utils.config_from_keras_model(
    model,
    granularity="name",
    default_precision="ap_fixed<12,5>",
)

config["Model"]["ReuseFactor"] = 32
config["Model"]["Strategy"] = "Resource"

for layer_name in config["LayerName"]:
    lname = layer_name.lower()

    config["LayerName"][layer_name]["ReuseFactor"] = 32
    config["LayerName"][layer_name]["Precision"] = {
        "result": "ap_fixed<12,5>",
        "weight": "ap_fixed<10,3>",
        "bias": "ap_fixed<10,3>",
    }

    if "input" in lname:
        config["LayerName"][layer_name]["Precision"] = {
            "result": "ap_fixed<10,4>"
        }

    if "output" in lname:
        config["LayerName"][layer_name]["ReuseFactor"] = 8
        config["LayerName"][layer_name]["Precision"] = {
            "result": "ap_fixed<12,5>",
            "weight": "ap_fixed<10,3>",
            "bias": "ap_fixed<10,3>",
        }

print("[+] Converting/compiling hls4ml model...")

hls_model = hls4ml.converters.convert_from_keras_model(
    model,
    hls_config=config,
    output_dir=OUTPUT_DIR,
    backend="Vivado",
    part=PART,
    io_type="io_stream",
)

hls_model.compile()

print("[+] Predicting with Keras and hls4ml C model...")

keras_logits = model.predict(Xq, verbose=0).reshape(-1)
hls_scores = hls_model.predict(Xq).reshape(-1)

raw_scores = np.round(hls_scores * OUTPUT_SCALE).astype(int)

print("\nFirst 30 hls4ml scores:")
for i in range(min(30, len(hls_scores))):
    print(
        f"{i:3d} label={y[i]} "
        f"keras={keras_logits[i]: .6f} "
        f"hls={hls_scores[i]: .6f} "
        f"raw={raw_scores[i]:5d}"
    )

print("\nScore diagnostics:")
print("hls min =", float(np.min(hls_scores)))
print("hls max =", float(np.max(hls_scores)))
print("hls mean =", float(np.mean(hls_scores)))
print("hls std =", float(np.std(hls_scores)))
print("unique raw scores =", len(np.unique(raw_scores)))
print("first 30 unique raw scores =", np.unique(raw_scores)[:30])

try:
    auc = roc_auc_score(y, hls_scores)
    print("hls ROC AUC =", auc)
except Exception as e:
    print("[WARN] Could not compute AUC:", e)

print("\nThreshold scan over raw fixed-point thresholds:")
best = None

raw_min = int(np.min(raw_scores)) - 8
raw_max = int(np.max(raw_scores)) + 8

for raw_thr in range(raw_min, raw_max + 1):
    thr = raw_thr / OUTPUT_SCALE
    pred = (hls_scores > thr).astype(int)

    acc = accuracy_score(y, pred)
    prec = precision_score(y, pred, zero_division=0)
    rec = recall_score(y, pred, zero_division=0)
    f1 = f1_score(y, pred, zero_division=0)

    if best is None or f1 > best["f1"]:
        best = {
            "raw_thr": raw_thr,
            "thr": thr,
            "acc": acc,
            "prec": prec,
            "rec": rec,
            "f1": f1,
            "n_trigger": int(pred.sum()),
        }

print("\nBest threshold by F1:")
print(best)

print("\nUseful candidate thresholds:")
candidate_raws = sorted(set([
    raw_min,
    raw_min + 8,
    0,
    int(np.percentile(raw_scores, 25)),
    int(np.percentile(raw_scores, 50)),
    int(np.percentile(raw_scores, 75)),
    int(np.percentile(raw_scores, 90)),
    raw_max - 8,
    raw_max,
]))

for raw_thr in candidate_raws:
    thr = raw_thr / OUTPUT_SCALE
    pred = (hls_scores > thr).astype(int)

    print(
        f"raw_thr={raw_thr:5d}, thr={thr: .6f}, "
        f"acc={accuracy_score(y,pred):.3f}, "
        f"prec={precision_score(y,pred,zero_division=0):.3f}, "
        f"rec={recall_score(y,pred,zero_division=0):.3f}, "
        f"f1={f1_score(y,pred,zero_division=0):.3f}, "
        f"n_trigger={pred.sum()}"
    )

with open("medium_stream_hls_threshold_scan_summary.txt", "w") as f:
    f.write("Medium Stream hls4ml Threshold Scan Summary\n")
    f.write("==========================================\n\n")
    f.write(f"N_EVENTS = {N_EVENTS}\n")
    f.write(f"y balance = {np.bincount(y)}\n")
    f.write(f"hls min = {float(np.min(hls_scores))}\n")
    f.write(f"hls max = {float(np.max(hls_scores))}\n")
    f.write(f"hls mean = {float(np.mean(hls_scores))}\n")
    f.write(f"hls std = {float(np.std(hls_scores))}\n")
    f.write(f"unique raw scores = {len(np.unique(raw_scores))}\n")
    f.write(f"best = {best}\n\n")
    f.write("First 30 scores:\n")
    for i in range(min(30, len(hls_scores))):
        f.write(
            f"{i:3d} label={y[i]} "
            f"keras={keras_logits[i]: .6f} "
            f"hls={hls_scores[i]: .6f} "
            f"raw={raw_scores[i]:5d}\n"
        )

print("\n[+] Wrote medium_stream_hls_threshold_scan_summary.txt")
print("[+] Done.")
