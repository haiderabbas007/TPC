import tensorflow as tf
from tensorflow.keras.utils import plot_model


# ============================================================
# Original larger CNN
# ============================================================

large_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(20, 256, 1), name="input_20x256"),

    tf.keras.layers.Conv2D(16, (3, 7), activation="relu", padding="same", name="conv1_16_3x7"),
    tf.keras.layers.MaxPooling2D((2, 2), name="pool1_2x2"),
    tf.keras.layers.Dropout(0.2, name="dropout1"),

    tf.keras.layers.Conv2D(32, (5, 15), activation="relu", padding="same", name="conv2_32_5x15"),
    tf.keras.layers.MaxPooling2D((2, 2), name="pool2_2x2"),
    tf.keras.layers.Dropout(0.2, name="dropout2"),

    tf.keras.layers.Flatten(name="flatten"),
    tf.keras.layers.Dense(128, activation="relu", name="dense_128"),
    tf.keras.layers.Dropout(0.4, name="dropout3"),
    tf.keras.layers.Dense(1, activation="sigmoid", name="output_sigmoid"),
], name="large_subhitmap_cnn")

large_model.summary()

plot_model(
    large_model,
    to_file="keras_large_cnn_architecture.png",
    show_shapes=True,
    show_layer_names=True,
    expand_nested=True,
    dpi=200,
)


# ============================================================
# Compressed 229-parameter CNN
# ============================================================

small_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(20, 100, 1), name="input_20x100"),

    tf.keras.layers.Conv2D(4, (3, 5), padding="same", use_bias=True, activation=None, name="conv1_4_3x5"),
    tf.keras.layers.ReLU(name="relu1"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool1_2x2"),

    tf.keras.layers.Conv2D(8, (3, 7), padding="same", use_bias=True, activation=None, name="conv2_8_3x7"),
    tf.keras.layers.ReLU(name="relu2"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 4), name="pool2_2x4"),

    tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool"),

    tf.keras.layers.Dense(8, use_bias=True, activation=None, name="dense_8"),
    tf.keras.layers.ReLU(name="relu_dense"),

    tf.keras.layers.Dense(1, use_bias=True, activation="linear", name="output_logit"),
], name="compressed_229_parameter_cnn")

small_model.summary()

plot_model(
    small_model,
    to_file="keras_229_cnn_architecture.png",
    show_shapes=True,
    show_layer_names=True,
    expand_nested=True,
    dpi=200,
)

print("[+] Saved keras_large_cnn_architecture.png")
print("[+] Saved keras_229_cnn_architecture.png")