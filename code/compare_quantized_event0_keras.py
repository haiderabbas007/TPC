#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

MODEL_PATH = "models_hls_tiny_manual_prune_cnn_80_180/hls_tiny_manual_prune_cnn_80_180.h5"
MEM_PATH = "real_event.mem"

# HLS input assumed ap_fixed<8,3>, so fractional bits = 5
INPUT_FRAC_BITS = 5

# RTL result from Vivado
RTL_RAW_SIGNED = 44

# Try common output fractional-bit interpretations
possible_output_frac_bits = [4, 5, 6, 7, 8]

print("[+] Loading real_event.mem...")
raw = []
with open(MEM_PATH, "r") as f:
    for line in f:
        h = line.strip()
        if not h:
            continue
        val = int(h, 16)
        if val >= 128:
            val -= 256
        raw.append(val)

raw = np.array(raw, dtype=np.float32)

if raw.size != 2000:
    raise RuntimeError(f"Expected 2000 samples, got {raw.size}")

# Convert back from ap_fixed<8,3> raw integer to real value
x_quantized_real = raw / (2 ** INPUT_FRAC_BITS)
x_quantized_real = x_quantized_real.reshape(1, 20, 100, 1).astype(np.float32)

print("[+] Quantized input stats reconstructed from mem:")
print("    min =", x_quantized_real.min())
print("    max =", x_quantized_real.max())
print("    mean =", x_quantized_real.mean())
print("    nonzero count =", np.count_nonzero(x_quantized_real))

print("[+] Loading Keras model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

print("[+] Running Keras on reconstructed quantized input...")
y = model.predict(x_quantized_real)
keras_quantized_input_output = float(y.reshape(-1)[0])

print("\n==============================")
print("Keras on quantized real_event.mem")
print("==============================")
print("Keras output on quantized input =", keras_quantized_input_output)

print("\nRTL output interpretations:")
for fb in possible_output_frac_bits:
    rtl_score = RTL_RAW_SIGNED / (2 ** fb)
    print(f"  if output frac bits = {fb}: RTL score = {rtl_score}, diff = {keras_quantized_input_output - rtl_score}")

threshold_prob = 0.29
threshold_logit = np.log(threshold_prob / (1.0 - threshold_prob))

print("\nThreshold:")
print("threshold logit =", threshold_logit)
print("Keras decision on quantized input =", int(keras_quantized_input_output > threshold_logit))

for fb in possible_output_frac_bits:
    rtl_score = RTL_RAW_SIGNED / (2 ** fb)
    print(f"RTL decision if frac bits={fb} =", int(rtl_score > threshold_logit))
