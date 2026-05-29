#!/usr/bin/env python3

import numpy as np
import h5py
import hls4ml
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

MODEL_PATH = "models_hls_tiny_manual_prune_cnn_80_180/hls_tiny_manual_prune_cnn_80_180.h5"
H5_PATH = "out_sub/test.h5"
OUTPUT_DIR = "hls4ml_tiny_manual_prune_cnn_80_180_stream_check"

N_EVENTS = 300
CROP_START = 80
CROP_END = 180
NORMALIZATION_SCALE = 0.9490740895271301

INPUT_SHAPE = (20, 100, 1)
PART = "xcu250-figd2104-2L-e"

def quantize_input_ap_fixed_8_3(x):
    raw = np.round(x * 32.0)
    raw = np.clip(raw, -128, 127)
    return raw / 32.0

print("[+] Loading data...")
with h5py.File(H5_PATH, "r") as h5:
    X = h5["X"][:N_EVENTS, :, CROP_START:CROP_END, 0].astype(np.float32)
    y = h5["y"][:N_EVENTS].astype(int)

X = X / NORMALIZATION_SCALE
Xq = quantize_input_ap_fixed_8_3(X)
Xq = Xq.reshape(-1, 20, 100, 1).astype(np.float32)

print("[+] Xq shape:", Xq.shape)
print("[+] y balance:", np.bincount(y))

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
    default_precision="ap_fixed<8,3>",
)

config["Model"]["ReuseFactor"] = 16
config["Model"]["Strategy"] = "Resource"

for layer_name in config["LayerName"]:
    config["LayerName"][layer_name]["Precision"] = {
        "result": "ap_fixed<8,3>",
        "weight": "ap_fixed<8,2>",
        "bias": "ap_fixed<8,2>",
    }
    config["LayerName"][layer_name]["ReuseFactor"] = 16

    if layer_name == "conv1":
        config["LayerName"][layer_name]["ReuseFactor"] = 30
        config["LayerName"][layer_name]["Precision"] = {
            "result": "ap_fixed<10,4>",
            "weight": "ap_fixed<8,2>",
            "bias": "ap_fixed<8,2>",
        }
    elif layer_name == "conv2":
        config["LayerName"][layer_name]["ReuseFactor"] = 42
        config["LayerName"][layer_name]["Precision"] = {
            "result": "ap_fixed<10,4>",
            "weight": "ap_fixed<8,2>",
            "bias": "ap_fixed<8,2>",
        }
    elif layer_name == "dense1":
        config["LayerName"][layer_name]["ReuseFactor"] = 16
        config["LayerName"][layer_name]["Precision"] = {
            "result": "ap_fixed<10,4>",
            "weight": "ap_fixed<8,2>",
            "bias": "ap_fixed<8,2>",
        }
    elif layer_name == "output":
        config["LayerName"][layer_name]["ReuseFactor"] = 4
        config["LayerName"][layer_name]["Precision"] = {
            "result": "ap_fixed<10,4>",
            "weight": "ap_fixed<8,2>",
            "bias": "ap_fixed<8,2>",
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

print("[+] Predicting with hls4ml C model...")
scores = hls_model.predict(Xq).reshape(-1)

# hls output is ap_fixed<10,4>, step is 1/64.
raw_scores = np.round(scores * 64).astype(int)

print("\nFirst 20 hls4ml scores:")
for i in range(20):
    print(f"{i:3d} label={y[i]} score={scores[i]: .6f} raw={raw_scores[i]:4d}")

print("\nThreshold scan:")
best = None
for raw_thr in range(-128, 129):
    thr = raw_thr / 64.0
    pred = (scores > thr).astype(int)

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
        }

print("Best threshold by F1:")
print(best)

print("\nUseful candidate thresholds:")
for raw_thr in [-57, 0, 16, 24, 32, 40, 44, 48, 56, 64]:
    thr = raw_thr / 64.0
    pred = (scores > thr).astype(int)
    print(
        f"raw_thr={raw_thr:4d}, thr={thr: .4f}, "
        f"acc={accuracy_score(y,pred):.3f}, "
        f"prec={precision_score(y,pred,zero_division=0):.3f}, "
        f"rec={recall_score(y,pred,zero_division=0):.3f}, "
        f"f1={f1_score(y,pred,zero_division=0):.3f}, "
        f"n_trigger={pred.sum()}"
    )
