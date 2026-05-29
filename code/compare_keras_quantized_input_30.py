import h5py
import numpy as np
import tensorflow as tf

MODEL_PATH = "/workspaces/TPC/code/hls4ml_medium_stream_cnn_80_180/keras_model.keras"
H5_PATH    = "/workspaces/TPC/code/out_sub_80_180/test.h5"

SCALE = 64

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

with h5py.File(H5_PATH, "r") as f:
    X = f["X"][:30]
    y = f["y"][:30]

Xq = np.round(X * SCALE) / SCALE

scores = model.predict(Xq, verbose=0).reshape(-1)

print("\nidx  true_y  quant_keras_score  quant_keras_pred")
print("------------------------------------------------")
correct = 0

for i, (yi, s) in enumerate(zip(y, scores)):
    yi = int(yi)
    pred = int(s >= 0.0)
    correct += int(pred == yi)
    print(f"{i:3d}  {yi:6d}  {float(s):17.8f}  {pred:16d}")

print("------------------------------------------------")
print(f"Keras accuracy with ap_fixed<10,4>-like input quantization = {correct}/30 = {correct/30:.3f}")
