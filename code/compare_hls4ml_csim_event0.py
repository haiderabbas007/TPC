#!/usr/bin/env python3

import numpy as np
import hls4ml
import tensorflow as tf

MODEL_PATH = "models_hls_tiny_manual_prune_cnn_80_180/hls_tiny_manual_prune_cnn_80_180.h5"
MEM_PATH = "real_event.mem"
OUTPUT_DIR = "hls4ml_tiny_manual_prune_cnn_80_180_stream_check"

INPUT_SHAPE = (20, 100, 1)
PART = "xcu250-figd2104-2L-e"

print("[+] Loading Keras model...")
old_model = tf.keras.models.load_model(MODEL_PATH, compile=False)

new_input = tf.keras.Input(shape=INPUT_SHAPE, name="input_layer")
x = new_input
for layer in old_model.layers:
    if isinstance(layer, tf.keras.layers.InputLayer):
        continue
    x = layer(x)

model = tf.keras.Model(inputs=new_input, outputs=x)

print("[+] Loading real_event.mem...")
raw = []
with open(MEM_PATH, "r") as f:
    for line in f:
        h = line.strip()
        if not h:
            continue
        v = int(h, 16)
        if v >= 128:
            v -= 256
        raw.append(v)

raw = np.array(raw, dtype=np.float32)
x_quant = raw / 32.0   # ap_fixed<8,3> input, 5 fractional bits
x_quant = x_quant.reshape(1, 20, 100, 1).astype(np.float32)

print("[+] Input stats reconstructed from real_event.mem:")
print("min =", x_quant.min())
print("max =", x_quant.max())
print("mean =", x_quant.mean())
print("nonzero =", np.count_nonzero(x_quant))

print("[+] Keras prediction on reconstructed input...")
y_keras = model.predict(x_quant)
print("Keras output =", y_keras.reshape(-1)[0])

print("[+] Building hls4ml config...")
config = hls4ml.utils.config_from_keras_model(
    model,
    granularity="name",
    default_precision="ap_fixed<8,3>",
)

config["Model"]["ReuseFactor"] = 16
config["Model"]["Strategy"] = "Resource"

for layer_name in config["LayerName"]:
    lname = layer_name.lower()

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

print("[+] Converting to hls4ml...")
hls_model = hls4ml.converters.convert_from_keras_model(
    model,
    hls_config=config,
    output_dir=OUTPUT_DIR,
    backend="Vivado",
    part=PART,
    io_type="io_stream",
)

print("[+] Compiling hls4ml C simulation model...")
hls_model.compile()

print("[+] Running hls4ml prediction...")
y_hls = hls_model.predict(x_quant)

print("\n==============================")
print("Event 0 comparison")
print("==============================")
print("Keras output        =", float(y_keras.reshape(-1)[0]))
print("hls4ml C output     =", float(y_hls.reshape(-1)[0]))
print("Vivado RTL raw      =", 44)
print("Vivado RTL /64      =", 44 / 64.0)
print("Vivado RTL /1024    =", 44 / 1024.0)
