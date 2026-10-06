from flask import Flask, render_template, jsonify
import numpy as np
import pandas as pd
import yfinance as yf
import joblib

from tensorflow.keras.models import load_model


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# SETTINGS
# ============================================================

TICKER = "RELIANCE.NS"

SEQUENCE_LENGTH = 60

MODEL_PATH = "models/stock_rnn.keras"

SCALER_PATH = "models/scaler.pkl"


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading RNN model...")

model = load_model(MODEL_PATH)

print("RNN model loaded successfully.")


# ============================================================
# LOAD SCALER
# ============================================================

print("\nLoading scaler...")

scaler = joblib.load(SCALER_PATH)

print("Scaler loaded successfully.")


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template("index.html")


# ============================================================
# STOCK PREDICTION
# ============================================================

@app.route("/predict")
def predict():

    try:

        print("\n======================================")
        print("Downloading latest stock data...")
        print("======================================")


        # ----------------------------------------------------
        # Download latest data
        # ----------------------------------------------------

        data = yf.download(
            TICKER,
            period="6mo",
            interval="1d",
            auto_adjust=True,
            progress=False
        )


        if data.empty:

            return jsonify({
                "success": False,
                "error": "Could not download stock data."
            })


        # ----------------------------------------------------
        # Get closing price
        # ----------------------------------------------------

        prices = data["Close"]


        # Handle yfinance DataFrame format
        if isinstance(prices, pd.DataFrame):

            prices = prices.iloc[:, 0]


        prices = pd.to_numeric(
            prices,
            errors="coerce"
        )


        prices = prices.dropna()


        # Convert to NumPy
        prices = prices.values.reshape(-1, 1)


        # ----------------------------------------------------
        # Check data
        # ----------------------------------------------------

        if len(prices) < SEQUENCE_LENGTH:

            return jsonify({
                "success": False,
                "error": "Not enough stock data."
            })


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
        # Scale data
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
        # ----------------------------------------------------

        X = last_60_days.reshape(
            1,
            SEQUENCE_LENGTH,
            1
        )


        # ----------------------------------------------------
        # Make prediction
        # ----------------------------------------------------

        prediction = model.predict(
            X,
            verbose=0
        )


        # ----------------------------------------------------
        # Convert prediction back to ₹
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
        # Calculate change
        # ----------------------------------------------------

        price_difference = (
            predicted_price -
            latest_price
        )


        percentage_change = (
            price_difference /
            latest_price
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
            f"Expected change: {percentage_change:.2f}%"
        )


        # ----------------------------------------------------
        # Send result to browser
        # ----------------------------------------------------

        return jsonify({

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

        })


    except Exception as e:

        print("\nERROR:")
        print(str(e))


        return jsonify({

            "success": False,

            "error": str(e)

        })


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("==============================================")
    print("       RNN STOCK PREDICTION WEB APP")
    print("==============================================")

    print("\nStarting Flask server...")

    print(
        "\nOpen this in your browser:"
    )

    print(
        "http://127.0.0.1:5000"
    )


    app.run(

        host="127.0.0.1",

        port=5000,

        debug=False

    )