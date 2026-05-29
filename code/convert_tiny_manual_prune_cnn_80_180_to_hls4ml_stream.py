#!/usr/bin/env python3

import hls4ml
import tensorflow as tf

# ============================================================
# Paths
# ============================================================

MODEL_PATH = "models_hls_tiny_manual_prune_cnn_80_180/hls_tiny_manual_prune_cnn_80_180.h5"
OUTPUT_DIR = "hls4ml_tiny_manual_prune_cnn_80_180_stream"

DO_COMPILE = False

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
    name="hls_tiny_manual_prune_stream_safe",
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

# Conservative resource-oriented strategy
config["Model"]["ReuseFactor"] = 16
config["Model"]["Strategy"] = "Resource"

# Layer-specific reuse factors chosen from the valid values hls4ml printed:
# conv1 valid:   1,3,5,15,30
# conv2 valid:   1,2,3,6,7,14,21,42,84,168
# dense1 valid:  1,2,4,8,16
# output valid:  1,2,4

for layer_name in config["LayerName"]:
    lname = layer_name.lower()

    # Default conservative precision
    config["LayerName"][layer_name]["Precision"] = {
        "result": "ap_fixed<8,3>",
        "weight": "ap_fixed<8,2>",
        "bias": "ap_fixed<8,2>",
    }

    # Default reuse factor
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

    elif "relu" in lname:
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

print("[+] Converting to hls4ml with io_stream...")

hls_model = hls4ml.converters.convert_from_keras_model(
    model,
    hls_config=config,
    output_dir=OUTPUT_DIR,
    backend="Vivado",
    part=PART,

    # Use io_stream, not io_serial.
    # io_serial caused an hls4ml writer bug in this environment.
    io_type="io_stream",
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