import numpy as np
from tensorflow.keras.models import load_model


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/stock_rnn.keras"
WEIGHTS_PATH = "models/rnn_weights.npz"


# ============================================================
# LOAD TENSORFLOW MODEL
# ============================================================

print("Loading TensorFlow model...")

model = load_model(MODEL_PATH)

print("TensorFlow model loaded.")


# ============================================================
# LOAD EXPORTED WEIGHTS
# ============================================================

print("\nLoading NumPy weights...")

weights = np.load(WEIGHTS_PATH)

print("Weights loaded.")


# ============================================================
# NUMPY SIMPLE RNN
# ============================================================

def simple_rnn(
    x,
    kernel,
    recurrent_kernel,
    bias
):
    """
    Reproduce Keras SimpleRNN inference.

    x shape:
        (timesteps, features)

    returns:
        final hidden state
    """

    hidden_size = recurrent_kernel.shape[0]

    h = np.zeros(
        hidden_size,
        dtype=np.float32
    )

    for t in range(x.shape[0]):

        x_t = x[t]

        h = np.tanh(
            np.dot(
                x_t,
                kernel
            )
            +
            np.dot(
                h,
                recurrent_kernel
            )
            +
            bias
        )

    return h


# ============================================================
# NUMPY MODEL
# ============================================================

def numpy_model(x):

    # --------------------------------------------------------
    # RNN 1
    # --------------------------------------------------------

    rnn1_kernel = weights["rnn1_kernel"]
    rnn1_recurrent = weights["rnn1_recurrent"]
    rnn1_bias = weights["rnn1_bias"]

    sequence = []

    h = np.zeros(
        rnn1_recurrent.shape[0],
        dtype=np.float32
    )

    for t in range(x.shape[0]):

        x_t = x[t]

        h = np.tanh(
            np.dot(
                x_t,
                rnn1_kernel
            )
            +
            np.dot(
                h,
                rnn1_recurrent
            )
            +
            rnn1_bias
        )

        sequence.append(h.copy())

    sequence = np.array(
        sequence,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # RNN 2
    # --------------------------------------------------------

    rnn2_output = simple_rnn(
        sequence,
        weights["rnn2_kernel"],
        weights["rnn2_recurrent"],
        weights["rnn2_bias"]
    )


    # --------------------------------------------------------
    # Dense 1
    # --------------------------------------------------------

    dense1_output = np.dot(
        rnn2_output,
        weights["dense1_kernel"]
    ) + weights["dense1_bias"]


    # --------------------------------------------------------
    # Dense 2
    # --------------------------------------------------------

    dense2_output = np.dot(
        dense1_output,
        weights["dense2_kernel"]
    ) + weights["dense2_bias"]


    return dense2_output


# ============================================================
# CREATE TEST INPUT
# ============================================================

print("\nCreating test input...")

np.random.seed(42)

test_input = np.random.random(
    (1, 60, 1)
).astype(np.float32)


# ============================================================
# TENSORFLOW PREDICTION
# ============================================================

print("\nRunning TensorFlow prediction...")

tensorflow_prediction = model.predict(
    test_input,
    verbose=0
)

tensorflow_value = float(
    tensorflow_prediction[0][0]
)


# ============================================================
# NUMPY PREDICTION
# ============================================================

print("\nRunning NumPy prediction...")

numpy_prediction = numpy_model(
    test_input[0]
)

numpy_value = float(
    numpy_prediction[0]
)


# ============================================================
# COMPARE
# ============================================================

difference = abs(
    tensorflow_value
    - numpy_value
)


print("\n======================================")
print("       MODEL VERIFICATION")
print("======================================")

print(
    f"TensorFlow : {tensorflow_value:.10f}"
)

print(
    f"NumPy      : {numpy_value:.10f}"
)

print(
    f"Difference : {difference:.10f}"
)

print("======================================")


# ============================================================
# RESULT
# ============================================================

if difference < 0.00001:

    print("\nSUCCESS!")
    print(
        "NumPy prediction matches "
        "TensorFlow prediction."
    )

else:

    print("\nWARNING!")
    print(
        "The predictions are still different."
    )