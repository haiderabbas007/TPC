#!/usr/bin/env python3

import hls4ml
import tensorflow as tf

MODEL_PATH = "models_hls_tiny_manual_prune_cnn_80_180/hls_tiny_manual_prune_cnn_80_180.h5"
OUTPUT_DIR = "hls4ml_tiny_manual_prune_cnn_80_180"

DO_COMPILE = False

# New cropped input shape: 20 x 100 x 1
INPUT_SHAPE = (20, 100, 1)

print("[+] Loading Keras model...")
old_model = tf.keras.models.load_model(MODEL_PATH, compile=False)

print("[+] Rebuilding model with safe input name...")

new_input = tf.keras.Input(shape=INPUT_SHAPE, name="input_layer")
x = new_input

for layer in old_model.layers:
    if isinstance(layer, tf.keras.layers.InputLayer):
        continue
    x = layer(x)

model = tf.keras.Model(inputs=new_input, outputs=x, name="hls_tiny_manual_prune_safe")

print("[+] New model summary:")
model.summary()

print("[+] Creating hls4ml config...")

config = hls4ml.utils.config_from_keras_model(
    model,
    granularity="name",
    default_precision="ap_fixed<8,3>",
)

# Conservative FPGA settings.
# ReuseFactor = 16 reduces parallel hardware, usually making synthesis easier.
config["Model"]["ReuseFactor"] = 16
config["Model"]["Strategy"] = "Resource"

for layer_name in config["LayerName"]:
    config["LayerName"][layer_name]["ReuseFactor"] = 16

    # Default layer precision
    config["LayerName"][layer_name]["Precision"] = {
        "result": "ap_fixed<8,3>",
        "weight": "ap_fixed<8,2>",
        "bias": "ap_fixed<8,2>",
    }

    lname = layer_name.lower()

    # Give dense/output layers slightly safer precision
    if "dense" in lname or "output" in lname:
        config["LayerName"][layer_name]["Precision"] = {
            "result": "ap_fixed<10,4>",
            "weight": "ap_fixed<8,2>",
            "bias": "ap_fixed<8,2>",
        }

print("[+] hls4ml config:")
print(config)

print("[+] Converting to hls4ml...")

hls_model = hls4ml.converters.convert_from_keras_model(
    model,
    hls_config=config,
    output_dir=OUTPUT_DIR,
    backend="Vivado",
    part="xcu250-figd2104-2L-e",
    io_type="io_parallel",
)

print("[+] Writing HLS project...")
hls_model.write()

if DO_COMPILE:
    import numpy as np

    print("[+] Compiling hls model...")
    hls_model.compile()

    X = np.random.rand(10, *INPUT_SHAPE).astype("float32")

    print("[+] Running Keras prediction...")
    y_keras = model.predict(X)

    print("[+] Running HLS prediction...")
    y_hls = hls_model.predict(X)

    print("Keras:", y_keras[:5].flatten())
    print("HLS  :", y_hls[:5].flatten())

else:
    print("[!] Skipping hls_model.compile() because Codespaces may run out of memory.")
    print("[+] HLS C++ project generation still completed.")

print("\n[+] Done.")
print(f"[+] HLS project created at: {OUTPUT_DIR}")
print(f"[+] Check generated files under: {OUTPUT_DIR}/firmware/")