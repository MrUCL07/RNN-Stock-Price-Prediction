# ============================================================
# RELIANCE STOCK PRICE PREDICTION
# NumPy RNN Prediction
# ============================================================

import numpy as np
import pandas as pd
import joblib


# ============================================================
# SETTINGS
# ============================================================

FILE_PATH = "data/RELIANCE.csv"

WEIGHTS_PATH = "models/rnn_weights.npz"

SCALER_PATH = "models/scaler.pkl"

SEQUENCE_LENGTH = 60


# ============================================================
# LOAD DATA
# ============================================================

print("\n==============================================")
print("       NUMPY STOCK PRICE PREDICTION")
print("==============================================")

print("\nLoading RELIANCE data...")

data = pd.read_csv(
    FILE_PATH,
    skiprows=[1, 2]
)


# ============================================================
# CLEAN DATA
# ============================================================

data["Close"] = pd.to_numeric(
    data["Close"],
    errors="coerce"
)

data = data.dropna(
    subset=["Close"]
)

prices = data["Close"].values.reshape(-1, 1)

print("Valid price records:", len(prices))


# ============================================================
# LOAD SCALER
# ============================================================

print("\nLoading scaler...")

scaler = joblib.load(
    SCALER_PATH
)

print("Scaler loaded.")


# ============================================================
# SCALE DATA
# ============================================================

scaled_prices = scaler.transform(
    prices
)


# ============================================================
# GET LAST 60 DAYS
# ============================================================

if len(scaled_prices) < SEQUENCE_LENGTH:

    raise ValueError(
        "Not enough stock data for prediction."
    )


latest_sequence = scaled_prices[
    -SEQUENCE_LENGTH:
]


# Add batch dimension

X = latest_sequence.reshape(
    1,
    SEQUENCE_LENGTH,
    1
).astype(np.float32)


print("\nInput shape:", X.shape)


# ============================================================
# LOAD NUMPY WEIGHTS
# ============================================================

print("\nLoading RNN weights...")

weights = np.load(
    WEIGHTS_PATH
)

print("Weights loaded.")


# ============================================================
# SIMPLE RNN
# ============================================================

def simple_rnn(
    x,
    kernel,
    recurrent_kernel,
    bias,
    return_sequences=True
):

    batch_size = x.shape[0]

    units = recurrent_kernel.shape[0]

    h = np.zeros(
        (batch_size, units),
        dtype=np.float32
    )

    outputs = []

    for t in range(x.shape[1]):

        current_x = x[:, t, :]

        h = np.tanh(
            current_x @ kernel
            + h @ recurrent_kernel
            + bias
        )

        outputs.append(
            h.copy()
        )

    if return_sequences:

        return np.stack(
            outputs,
            axis=1
        )

    return h


# ============================================================
# NUMPY MODEL
# ============================================================

def numpy_predict(x):

    # --------------------------------------------------------
    # RNN 1
    # --------------------------------------------------------

    x = simple_rnn(
        x,
        weights["rnn1_kernel"],
        weights["rnn1_recurrent"],
        weights["rnn1_bias"],
        return_sequences=True
    )


    # --------------------------------------------------------
    # Dropout
    # --------------------------------------------------------

    # Dropout is disabled during prediction.


    # --------------------------------------------------------
    # RNN 2
    # --------------------------------------------------------

    x = simple_rnn(
        x,
        weights["rnn2_kernel"],
        weights["rnn2_recurrent"],
        weights["rnn2_bias"],
        return_sequences=False
    )


    # --------------------------------------------------------
    # Dense 1
    # --------------------------------------------------------

    # IMPORTANT:
    # Dense(25) in the training script has the default
    # linear activation.

    x = (
        x @ weights["dense1_kernel"]
        + weights["dense1_bias"]
    )


    # --------------------------------------------------------
    # Dense 2
    # --------------------------------------------------------

    x = (
        x @ weights["dense2_kernel"]
        + weights["dense2_bias"]
    )


    return x


# ============================================================
# MAKE PREDICTION
# ============================================================

print("\nRunning NumPy RNN...")

prediction_scaled = numpy_predict(
    X
)

print(
    "Scaled prediction:",
    float(prediction_scaled[0][0])
)


# ============================================================
# CONVERT BACK TO RUPEES
# ============================================================

prediction_price = scaler.inverse_transform(
    prediction_scaled.reshape(1, 1)
)


predicted_price = float(
    prediction_price[0][0]
)


# ============================================================
# GET LAST ACTUAL PRICE
# ============================================================

last_actual_price = float(
    prices[-1][0]
)


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n==============================================")
print("             PREDICTION RESULT")
print("==============================================")

print(
    f"Last actual Close : ₹{last_actual_price:.2f}"
)

print(
    f"Predicted Close   : ₹{predicted_price:.2f}"
)

print("==============================================")


# ============================================================
# PRICE CHANGE
# ============================================================

change = (
    predicted_price
    - last_actual_price
)

percentage = (
    change / last_actual_price
) * 100


print("\nExpected change:")

print(
    f"₹{change:.2f}"
)

print(
    f"{percentage:.2f}%"
)


# ============================================================
# DIRECTION
# ============================================================

if predicted_price > last_actual_price:

    print("\nPrediction direction: UP")

elif predicted_price < last_actual_price:

    print("\nPrediction direction: DOWN")

else:

    print("\nPrediction direction: UNCHANGED")


print("\n==============================================")