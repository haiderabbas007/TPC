import h5py
import numpy as np
import tensorflow as tf

MODEL_PATH = "/workspaces/TPC/code/hls4ml_medium_stream_cnn_80_180/keras_model.keras"
H5_PATH    = "/workspaces/TPC/code/out_sub_80_180/test.h5"

SCALE = 64

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

with h5py.File(H5_PATH, "r") as f:
    X = f["X"][:300]
    y = f["y"][:300].astype(int)

# FPGA-like input quantization for ap_fixed<10,4>
Xq = np.round(X * SCALE) / SCALE

scores = model.predict(Xq, verbose=0).reshape(-1)

# Linear logit threshold
pred = (scores >= 0.0).astype(int)

correct = (pred == y)
acc = correct.mean()

TP = int(np.sum((y == 1) & (pred == 1)))
TN = int(np.sum((y == 0) & (pred == 0)))
FP = int(np.sum((y == 0) & (pred == 1)))
FN = int(np.sum((y == 1) & (pred == 0)))

print("Keras quantized-input full 300 result")
print("====================================")
print("Correct:", int(correct.sum()), "/", len(y))
print("Accuracy:", acc)
print()
print("Confusion matrix with rows=true labels, cols=predicted labels:")
print("[[TN, FP],")
print(" [FN, TP]]")
print(np.array([[TN, FP], [FN, TP]]))
print()
print("TN:", TN)
print("FP:", FP)
print("FN:", FN)
print("TP:", TP)

precision = TP / (TP + FP) if (TP + FP) else 0
recall = TP / (TP + FN) if (TP + FN) else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

print()
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)

print()
print("First 30 check:")
print("Correct first 30:", int(correct[:30].sum()), "/ 30")
