from flask import Flask, render_template, jsonify
import os
import numpy as np
import pandas as pd
import joblib

from tensorflow.keras.models import load_model


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "stock_rnn.keras"
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

SEQUENCE_LENGTH = 60

TICKER = "RELIANCE.NS"


# ============================================================
# LOAD MODEL
# ============================================================

print("\n========================================")
print("Loading RNN model...")
print("========================================")

try:
    model = load_model(MODEL_PATH)

    print("RNN model loaded successfully.")

except Exception as e:

    print("ERROR loading model:")
    print(e)

    model = None


# ============================================================
# LOAD SCALER
# ============================================================

print("\n========================================")
print("Loading scaler...")
print("========================================")

try:
    scaler = joblib.load(SCALER_PATH)

    print("Scaler loaded successfully.")

except Exception as e:

    print("ERROR loading scaler:")
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

    # Read CSV
    data = pd.read_csv(
        DATA_PATH,
        skiprows=[1, 2]
    )

    print("CSV loaded successfully.")
    print("Total rows:", len(data))

    # Show columns
    print("Columns:", list(data.columns))

    # Find Close column
    if "Close" not in data.columns:

        raise ValueError(
            "Close column was not found in RELIANCE.csv"
        )

    # Convert Close column to numeric
    close_prices = pd.to_numeric(
        data["Close"],
        errors="coerce"
    )

    # Remove missing values
    close_prices = close_prices.dropna()

    print(
        "Valid closing prices:",
        len(close_prices)
    )

    if len(close_prices) < SEQUENCE_LENGTH:

        raise ValueError(
            "Not enough stock data for prediction."
        )

    return close_prices.values.reshape(-1, 1)


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template("index.html")


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
        # Check model
        # ----------------------------------------------------

        if model is None:

            return jsonify({
                "success": False,
                "error": "RNN model could not be loaded."
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
        # Load local stock data
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
        # Get last 60 trading days
        # ----------------------------------------------------

        last_60_days = scaled_prices[
            -SEQUENCE_LENGTH:
        ]


        # ----------------------------------------------------
        # Reshape for RNN
        #
        # Shape:
        # (1, 60, 1)
        # ----------------------------------------------------

        X = last_60_days.reshape(
            1,
            SEQUENCE_LENGTH,
            1
        )


        print(
            "Input shape:",
            X.shape
        )


        # ----------------------------------------------------
        # Make prediction
        # ----------------------------------------------------

        print(
            "Running RNN prediction..."
        )

        prediction = model.predict(
            X,
            verbose=0
        )


        # ----------------------------------------------------
        # Convert prediction back to price
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
        # Calculate difference
        # ----------------------------------------------------

        price_difference = (
            predicted_price
            - latest_price
        )


        # ----------------------------------------------------
        # Calculate percentage change
        # ----------------------------------------------------

        percentage_change = (
            price_difference
            / latest_price
        ) * 100


        # ----------------------------------------------------
        # Determine direction
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
        # Return JSON
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


        return jsonify(response)


    except Exception as e:

        print("\n========================================")
        print("PREDICTION ERROR")
        print("========================================")

        print(str(e))

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