import numpy as np
from tensorflow.keras.models import load_model
import os

MODEL_PATH = "models/stock_rnn.keras"
OUTPUT_PATH = "models/rnn_weights.npz"

print("Loading trained RNN model...")

model = load_model(MODEL_PATH)

print("\nModel layers:")

for i, layer in enumerate(model.layers):
    print(i, layer.name, layer.__class__.__name__)

# --------------------------------------------------
# Find the RNN layers
# --------------------------------------------------

rnn_layers = []

for layer in model.layers:
    if layer.__class__.__name__ == "SimpleRNN":
        rnn_layers.append(layer)

if len(rnn_layers) != 2:
    raise ValueError(
        f"Expected 2 SimpleRNN layers, found {len(rnn_layers)}"
    )

rnn1 = rnn_layers[0]
rnn2 = rnn_layers[1]

# --------------------------------------------------
# Get RNN weights
# --------------------------------------------------

rnn1_kernel, rnn1_recurrent, rnn1_bias = rnn1.get_weights()
rnn2_kernel, rnn2_recurrent, rnn2_bias = rnn2.get_weights()

# --------------------------------------------------
# Find Dense layers
# --------------------------------------------------

dense_layers = []

for layer in model.layers:
    if layer.__class__.__name__ == "Dense":
        dense_layers.append(layer)

if len(dense_layers) != 2:
    raise ValueError(
        f"Expected 2 Dense layers, found {len(dense_layers)}"
    )

dense1 = dense_layers[0]
dense2 = dense_layers[1]

dense1_kernel, dense1_bias = dense1.get_weights()
dense2_kernel, dense2_bias = dense2.get_weights()

# --------------------------------------------------
# Save everything
# --------------------------------------------------

os.makedirs("models", exist_ok=True)

np.savez(
    OUTPUT_PATH,

    rnn1_kernel=rnn1_kernel,
    rnn1_recurrent=rnn1_recurrent,
    rnn1_bias=rnn1_bias,

    rnn2_kernel=rnn2_kernel,
    rnn2_recurrent=rnn2_recurrent,
    rnn2_bias=rnn2_bias,

    dense1_kernel=dense1_kernel,
    dense1_bias=dense1_bias,

    dense2_kernel=dense2_kernel,
    dense2_bias=dense2_bias
)

print("\n======================================")
print("RNN WEIGHTS EXPORTED SUCCESSFULLY")
print("======================================")
print("Saved to:")
print(OUTPUT_PATH)