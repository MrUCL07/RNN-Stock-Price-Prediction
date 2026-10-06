from flask import Flask, render_template, jsonify
import os
import numpy as np
import pandas as pd
import joblib


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

WEIGHTS_PATH = os.path.join(
    BASE_DIR,
    "models",
    "rnn_weights.npz"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "scaler.pkl"
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "RELIANCE.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

SEQUENCE_LENGTH = 60
TICKER = "RELIANCE.NS"


# ============================================================
# LOAD NUMPY WEIGHTS
# ============================================================

print("\n========================================")
print("Loading NumPy RNN weights...")
print("========================================")

try:

    weights = np.load(
        WEIGHTS_PATH
    )

    print(
        "NumPy weights loaded successfully."
    )

except Exception as e:

    print(
        "ERROR loading NumPy weights:"
    )

    print(e)

    weights = None


# ============================================================
# LOAD SCALER
# ============================================================

print("\n========================================")
print("Loading scaler...")
print("========================================")

try:

    scaler = joblib.load(
        SCALER_PATH
    )

    print(
        "Scaler loaded successfully."
    )

except Exception as e:

    print(
        "ERROR loading scaler:"
    )

    print(e)

    scaler = None


# ============================================================
# LOAD LOCAL STOCK DATA
# ============================================================

def get_stock_data():

    print("\n========================================")
    print("Loading local stock data...")
    print("========================================")

    if not os.path.exists(DATA_PATH):

        raise FileNotFoundError(
            f"Stock data file not found: {DATA_PATH}"
        )

    data = pd.read_csv(
        DATA_PATH,
        skiprows=[1, 2]
    )

    print(
        "CSV loaded successfully."
    )

    print(
        "Total rows:",
        len(data)
    )

    print(
        "Columns:",
        list(data.columns)
    )

    if "Close" not in data.columns:

        raise ValueError(
            "Close column was not found."
        )

    close_prices = pd.to_numeric(
        data["Close"],
        errors="coerce"
    )

    close_prices = close_prices.dropna()

    print(
        "Valid closing prices:",
        len(close_prices)
    )

    if len(close_prices) < SEQUENCE_LENGTH:

        raise ValueError(
            "Not enough stock data."
        )

    return close_prices.values.reshape(
        -1,
        1
    )


# ============================================================
# NUMPY RNN
# ============================================================

def numpy_rnn(
    x,
    kernel,
    recurrent_kernel,
    bias
):

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

def predict_numpy(x):

    # --------------------------------------------------------
    # First SimpleRNN
    # --------------------------------------------------------

    rnn1_kernel = weights[
        "rnn1_kernel"
    ]

    rnn1_recurrent = weights[
        "rnn1_recurrent"
    ]

    rnn1_bias = weights[
        "rnn1_bias"
    ]

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

        sequence.append(
            h.copy()
        )

    sequence = np.array(
        sequence,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # Second SimpleRNN
    # --------------------------------------------------------

    rnn2_output = numpy_rnn(
        sequence,
        weights["rnn2_kernel"],
        weights["rnn2_recurrent"],
        weights["rnn2_bias"]
    )


    # --------------------------------------------------------
    # Dense layer 1
    # --------------------------------------------------------

    dense1_output = (
        np.dot(
            rnn2_output,
            weights["dense1_kernel"]
        )
        +
        weights["dense1_bias"]
    )


    # --------------------------------------------------------
    # Dense layer 2
    # --------------------------------------------------------

    dense2_output = (
        np.dot(
            dense1_output,
            weights["dense2_kernel"]
        )
        +
        weights["dense2_bias"]
    )


    return dense2_output


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# PREDICTION API
# ============================================================

@app.route("/predict")
def predict():

    print("\n========================================")
    print("PREDICTION REQUEST")
    print("========================================")

    try:

        # ----------------------------------------------------
        # Check weights
        # ----------------------------------------------------

        if weights is None:

            return jsonify({
                "success": False,
                "error": "RNN weights could not be loaded."
            }), 500


        # ----------------------------------------------------
        # Check scaler
        # ----------------------------------------------------

        if scaler is None:

            return jsonify({
                "success": False,
                "error": "Scaler could not be loaded."
            }), 500


        # ----------------------------------------------------
        # Load stock data
        # ----------------------------------------------------

        prices = get_stock_data()


        # ----------------------------------------------------
        # Latest price
        # ----------------------------------------------------

        latest_price = float(
            prices[-1][0]
        )

        print(
            f"Latest price: ₹{latest_price:.2f}"
        )


        # ----------------------------------------------------
        # Scale prices
        # ----------------------------------------------------

        scaled_prices = scaler.transform(
            prices
        )


        # ----------------------------------------------------
        # Last 60 trading days
        # ----------------------------------------------------

        last_60_days = scaled_prices[
            -SEQUENCE_LENGTH:
        ]


        # ----------------------------------------------------
        # RNN input
        #
        # Shape:
        # (60, 1)
        # ----------------------------------------------------

        X = last_60_days.astype(
            np.float32
        )

        print(
            "Input shape:",
            X.shape
        )


        # ----------------------------------------------------
        # NumPy prediction
        # ----------------------------------------------------

        print(
            "Running NumPy RNN prediction..."
        )

        prediction = predict_numpy(
            X
        )


        # ----------------------------------------------------
        # Reshape prediction
        # ----------------------------------------------------

        prediction = np.array(
            prediction,
            dtype=np.float32
        ).reshape(
            1,
            1
        )


        # ----------------------------------------------------
        # Convert back to original price
        # ----------------------------------------------------

        predicted_price = scaler.inverse_transform(
            prediction
        )

        predicted_price = float(
            predicted_price[0][0]
        )


        print(
            f"Predicted price: ₹{predicted_price:.2f}"
        )


        # ----------------------------------------------------
        # Price difference
        # ----------------------------------------------------

        price_difference = (
            predicted_price
            - latest_price
        )


        # ----------------------------------------------------
        # Percentage change
        # ----------------------------------------------------

        percentage_change = (
            price_difference
            / latest_price
        ) * 100


        # ----------------------------------------------------
        # Direction
        # ----------------------------------------------------

        if percentage_change > 0.1:

            direction = "UP"

        elif percentage_change < -0.1:

            direction = "DOWN"

        else:

            direction = "SAME"


        print(
            f"Expected change: "
            f"{percentage_change:.2f}%"
        )


        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        response = {

            "success": True,

            "data": {

                "ticker": TICKER,

                "latest_price": round(
                    latest_price,
                    2
                ),

                "predicted_price": round(
                    predicted_price,
                    2
                ),

                "price_difference": round(
                    price_difference,
                    2
                ),

                "percentage_change": round(
                    percentage_change,
                    2
                ),

                "direction": direction

            }

        }


        print(
            "Prediction completed successfully."
        )


        return jsonify(
            response
        )


    except Exception as e:

        print("\n========================================")
        print("PREDICTION ERROR")
        print("========================================")

        print(
            str(e)
        )

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# ============================================================
# RUN FLASK
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    print("\n========================================")
    print("RNN STOCK PREDICTION WEB APP")
    print("========================================")

    print(
        f"Starting server on port {port}..."
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )