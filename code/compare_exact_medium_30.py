import h5py
import numpy as np
import tensorflow as tf

MODEL_PATH = "/workspaces/TPC/code/hls4ml_medium_stream_cnn_80_180/keras_model.keras"
H5_PATH    = "/workspaces/TPC/code/out_sub_80_180/test.h5"

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
model.summary()

with h5py.File(H5_PATH, "r") as f:
    X = f["X"][:30]
    y = f["y"][:30]

scores = model.predict(X, verbose=0).reshape(-1)

print("\nidx  true_y  keras_score  pred_thresh_0  pred_thresh_0p5")
print("---------------------------------------------------------")

correct0 = 0
correct05 = 0

for i, (yi, s) in enumerate(zip(y, scores)):
    yi = int(yi)

    # Since the scores include negative and positive values,
    # this model likely outputs a logit, not a sigmoid probability.
    pred0 = int(s >= 0.0)

    # Also print 0.5 threshold for comparison.
    pred05 = int(s >= 0.5)

    correct0 += int(pred0 == yi)
    correct05 += int(pred05 == yi)

    print(f"{i:3d}  {yi:6d}  {float(s):11.8f}  {pred0:13d}  {pred05:15d}")

print("---------------------------------------------------------")
print(f"Keras accuracy with threshold 0.0 = {correct0}/30 = {correct0/30:.3f}")
print(f"Keras accuracy with threshold 0.5 = {correct05}/30 = {correct05/30:.3f}")
