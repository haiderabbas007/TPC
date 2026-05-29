#!/usr/bin/env python3

import numpy as np
import h5py
import tensorflow as tf

# ============================================================
# Settings
# ============================================================

H5_PATH = "out_sub/test.h5"
MODEL_PATH = "models_hls_tiny_manual_prune_cnn_80_180/hls_tiny_manual_prune_cnn_80_180.h5"

EVENT_INDEX = 0

# Same normalization scale used when making real_event.mem
NORMALIZATION_SCALE = 0.9490740895271301
APPLY_NORMALIZATION = True

# Vivado RTL result from your simulation
RTL_RAW_SIGNED = 44

# If output is ap_fixed<10,4>, fractional bits = 6
RTL_FRAC_BITS = 6
RTL_SCORE_APPROX = RTL_RAW_SIGNED / (2 ** RTL_FRAC_BITS)

print("[+] Loading HDF5 event...")
with h5py.File(H5_PATH, "r") as h5:
    X = h5["X"][()]

event_20x256 = X[EVENT_INDEX, :, :, 0].astype(np.float32)
event_20x100 = event_20x256[:, 80:180]

print("[+] Original event shape:", event_20x256.shape)
print("[+] Cropped event shape :", event_20x100.shape)

if APPLY_NORMALIZATION:
    event_20x100 = event_20x100 / NORMALIZATION_SCALE

x = event_20x100.reshape(1, 20, 100, 1).astype(np.float32)

print("[+] Input stats after crop/normalization:")
print("    min =", x.min())
print("    max =", x.max())
print("    mean =", x.mean())
print("    nonzero count =", np.count_nonzero(x))

print("[+] Loading Keras model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

print("[+] Running Keras prediction...")
y = model.predict(x)

keras_raw = float(y.reshape(-1)[0])

print("\n==============================")
print("Comparison for EVENT_INDEX =", EVENT_INDEX)
print("==============================")
print("Keras model output      =", keras_raw)
print("RTL raw signed integer  =", RTL_RAW_SIGNED)
print("RTL approx score /64    =", RTL_SCORE_APPROX)
print("Difference Keras - RTL  =", keras_raw - RTL_SCORE_APPROX)

# If model output is a logit, sigmoid probability:
prob = 1.0 / (1.0 + np.exp(-keras_raw))
rtl_prob = 1.0 / (1.0 + np.exp(-RTL_SCORE_APPROX))

print("\nIf interpreted as logits:")
print("Keras sigmoid probability =", prob)
print("RTL sigmoid probability   =", rtl_prob)

# Validation-selected threshold from your training summary
threshold_prob = 0.29
threshold_logit = np.log(threshold_prob / (1.0 - threshold_prob))

print("\nThresholds:")
print("Probability threshold =", threshold_prob)
print("Equivalent logit threshold =", threshold_logit)

print("\nDecisions:")
print("Keras decision =", int(keras_raw > threshold_logit))
print("RTL decision   =", int(RTL_SCORE_APPROX > threshold_logit))
