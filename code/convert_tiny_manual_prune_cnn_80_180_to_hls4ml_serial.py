#!/usr/bin/env python3

import hls4ml
import tensorflow as tf

# ============================================================
# Paths
# ============================================================

MODEL_PATH = "models_hls_tiny_manual_prune_cnn_80_180/hls_tiny_manual_prune_cnn_80_180.h5"
OUTPUT_DIR = "hls4ml_tiny_manual_prune_cnn_80_180_serial"

DO_COMPILE = False

# Your trained tiny model uses the 80:180 crop:
# 20 pads × 100 time bins × 1 channel
INPUT_SHAPE = (20, 100, 1)

PART = "xcu250-figd2104-2L-e"


# ============================================================
# Load Keras model
# ============================================================

print("[+] Loading Keras model...")

old_model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
)

print("[+] Original model summary:")
old_model.summary()


# ============================================================
# Rebuild model with safe input name
# ============================================================

print("[+] Rebuilding model with safe input name...")

new_input = tf.keras.Input(shape=INPUT_SHAPE, name="input_layer")
x = new_input

for layer in old_model.layers:
    if isinstance(layer, tf.keras.layers.InputLayer):
        continue
    x = layer(x)

model = tf.keras.Model(
    inputs=new_input,
    outputs=x,
    name="hls_tiny_manual_prune_serial_safe",
)

print("[+] New model summary:")
model.summary()


# ============================================================
# hls4ml config
# ============================================================

print("[+] Creating hls4ml config...")

config = hls4ml.utils.config_from_keras_model(
    model,
    granularity="name",
    default_precision="ap_fixed<8,3>",
)

# More conservative for Vitis HLS 2021.1
config["Model"]["ReuseFactor"] = 32
config["Model"]["Strategy"] = "Resource"

# Layer-specific settings
for layer_name in config["LayerName"]:
    lname = layer_name.lower()

    config["LayerName"][layer_name]["ReuseFactor"] = 32

    # Conservative default
    config["LayerName"][layer_name]["Precision"] = {
        "result": "ap_fixed<8,3>",
        "weight": "ap_fixed<8,2>",
        "bias": "ap_fixed<8,2>",
    }

    # Slightly safer accumulator/output precision for dense/output layers
    if "dense" in lname or "output" in lname:
        config["LayerName"][layer_name]["Precision"] = {
            "result": "ap_fixed<10,4>",
            "weight": "ap_fixed<8,2>",
            "bias": "ap_fixed<8,2>",
        }

    # Slightly safer for convolution output accumulation
    if "conv" in lname:
        config["LayerName"][layer_name]["Precision"] = {
            "result": "ap_fixed<10,4>",
            "weight": "ap_fixed<8,2>",
            "bias": "ap_fixed<8,2>",
        }

print("[+] hls4ml config:")
print(config)


# ============================================================
# Convert to hls4ml
# ============================================================

print("[+] Converting to hls4ml with io_serial...")

hls_model = hls4ml.converters.convert_from_keras_model(
    model,
    hls_config=config,
    output_dir=OUTPUT_DIR,
    backend="Vivado",
    part=PART,

    # Important change:
    # io_serial is safer than io_parallel for Vitis HLS 2021.1.
    # It should reduce top-level parallel interface/elaboration pressure.
    io_type="io_serial",
)

print("[+] Writing HLS project...")
hls_model.write()


# ============================================================
# Optional compile/test inside Python
# ============================================================

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
print("[+] Next: zip this folder and move it to the UH Windows computer.")