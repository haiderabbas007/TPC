#!/usr/bin/env python3
"""
convert_medium_stream_cnn_80_180_to_hls4ml.py

Convert the medium 20x100 CNN to hls4ml using io_stream.

Important:
    We use io_stream because io_parallel previously failed in Vitis HLS 2021.1.
"""

import hls4ml
import tensorflow as tf

MODEL_PATH = "models_hls_medium_stream_cnn_80_180/hls_medium_stream_cnn_80_180.h5"
OUTPUT_DIR = "hls4ml_medium_stream_cnn_80_180"

DO_COMPILE = False

INPUT_SHAPE = (20, 100, 1)
PART = "xcu250-figd2104-2L-e"

print("[+] Loading Keras model...")

old_model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
)

print("[+] Original model summary:")
old_model.summary()

print("[+] Rebuilding with safe input name...")

new_input = tf.keras.Input(shape=INPUT_SHAPE, name="input_layer")
x = new_input

for layer in old_model.layers:
    if isinstance(layer, tf.keras.layers.InputLayer):
        continue
    x = layer(x)

model = tf.keras.Model(
    inputs=new_input,
    outputs=x,
    name="hls_medium_stream_cnn_80_180_safe",
)

print("[+] New model summary:")
model.summary()

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

    if "dense" in lname:
        config["LayerName"][layer_name]["Precision"] = {
            "result": "ap_fixed<12,5>",
            "weight": "ap_fixed<10,3>",
            "bias": "ap_fixed<10,3>",
        }

    if "conv" in lname:
        config["LayerName"][layer_name]["Precision"] = {
            "result": "ap_fixed<12,5>",
            "weight": "ap_fixed<10,3>",
            "bias": "ap_fixed<10,3>",
        }

print("[+] hls4ml config:")
print(config)

print("[+] Converting with io_stream...")

hls_model = hls4ml.converters.convert_from_keras_model(
    model,
    hls_config=config,
    output_dir=OUTPUT_DIR,
    backend="Vivado",
    part=PART,
    io_type="io_stream",
)

print("[+] Writing HLS project...")
hls_model.write()

if DO_COMPILE:
    import numpy as np

    print("[+] Compiling hls model...")
    hls_model.compile()

    X = np.random.rand(4, *INPUT_SHAPE).astype("float32")

    print("[+] Running Keras prediction...")
    y_keras = model.predict(X)

    print("[+] Running hls4ml prediction...")
    y_hls = hls_model.predict(X)

    print("Keras:", y_keras.reshape(-1))
    print("HLS  :", y_hls.reshape(-1))

else:
    print("[!] Skipping hls_model.compile() for now.")
    print("[+] HLS C++ project generation completed.")

print("\n[+] Done.")
print(f"[+] HLS project created at: {OUTPUT_DIR}")
print(f"[+] Check generated files under: {OUTPUT_DIR}/firmware/")
