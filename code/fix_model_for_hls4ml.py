import tensorflow as tf
from tensorflow.keras import layers, Model

OLD_MODEL_PATH = "models_medium_cnn_corrected/medium_cnn_best_by_val_auc.h5"
NEW_MODEL_PATH = "models_medium_cnn_corrected/medium_cnn_hls_ready.h5"

print("[+] Loading old model...")
old_model = tf.keras.models.load_model(OLD_MODEL_PATH)

# ============================================================
# Rebuild model with Functional API
# ============================================================

inputs = layers.Input(shape=(20,256,1), name="input_layer")

x = layers.Conv2D(8, (3,7), padding="same", activation="relu", name="conv1")(inputs)
x = layers.MaxPooling2D((2,2), name="pool1")(x)

x = layers.Conv2D(16, (3,11), padding="same", activation="relu", name="conv2")(x)
x = layers.MaxPooling2D((2,4), name="pool2")(x)

x = layers.Conv2D(24, (3,11), padding="same", activation="relu", name="conv3")(x)
x = layers.MaxPooling2D((1,4), name="pool3")(x)

x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

x = layers.Dense(16, activation="relu", name="dense1")(x)
outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

new_model = Model(inputs=inputs, outputs=outputs)

# ============================================================
# Copy weights
# ============================================================

print("[+] Copying weights...")

for layer in new_model.layers:
    try:
        old_layer = old_model.get_layer(layer.name)
        layer.set_weights(old_layer.get_weights())
        print(f"✓ Copied: {layer.name}")
    except:
        print(f"Skipping: {layer.name}")

# ============================================================
# Save fixed model
# ============================================================

new_model.save(NEW_MODEL_PATH)

print(f"\n[+] Saved fixed model to: {NEW_MODEL_PATH}")
