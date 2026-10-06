from flask import Flask, render_template, jsonify
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import os

from tensorflow.keras.models import load_model

app = Flask(__name__)

TICKER = "RELIANCE.NS"
SEQUENCE_LENGTH = 60

MODEL_PATH = "models/stock_rnn.keras"
SCALER_PATH = "models/scaler.pkl"
LOCAL_DATA_PATH = "data/RELIANCE.csv"

print("\nLoading RNN model...")
model = load_model(MODEL_PATH)
print("RNN model loaded successfully.")

print("\nLoading scaler...")
scaler = joblib.load(SCALER_PATH)
print("Scaler loaded successfully.")


@app.route("/")
def home():
    return render_template("index.html")


def get_stock_data():
    """
    Try Yahoo Finance first.
    If Yahoo Finance is unavailable or rate-limited,
    use the locally saved CSV file.
    """

    print("\nTrying Yahoo Finance...")

    try:
        data = yf.download(
            TICKER,
            period="6mo",
            interval="1d",
            auto_adjust=True,
            progress=False,
            timeout=10
        )

        if not data.empty:
            print("Yahoo Finance data downloaded successfully.")
            return data

        print("Yahoo Finance returned empty data.")

    except Exception as e:
        print("Yahoo Finance error:", str(e))

    print("\nUsing local CSV data instead...")

    if not os.path.exists(LOCAL_DATA_PATH):
        raise FileNotFoundError(
            "Local stock data file was not found."
        )

    data = pd.read_csv(
        LOCAL_DATA_PATH,
        skiprows=[1, 2]
    )

    print("Local CSV loaded successfully.")

    return data


@app.route("/predict")
def predict():

    try:

        print("\n======================================")
        print("Starting stock prediction...")
        print("======================================")

        # --------------------------------
        # 1. Get stock data
        # --------------------------------

        data = get_stock_data()

        if data.empty:
            return jsonify({
                "success": False,
                "error": "No stock data available."
            }), 500

        # --------------------------------
        # 2. Extract Close price
        # --------------------------------

        prices = data["Close"]

        if isinstance(prices, pd.DataFrame):
            prices = prices.iloc[:, 0]

        prices = pd.to_numeric(
            prices,
            errors="coerce"
        )

        prices = prices.dropna()

        prices = prices.values.reshape(-1, 1)

        if len(prices) < SEQUENCE_LENGTH:
            return jsonify({
                "success": False,
                "error": "Not enough stock data for prediction."
            }), 500

        # --------------------------------
        # 3. Latest price
        # --------------------------------

        latest_price = float(prices[-1][0])

        print(
            f"Latest price: ₹{latest_price:.2f}"
        )

        # --------------------------------
        # 4. Scale using training scaler
        # --------------------------------

        scaled_prices = scaler.transform(prices)

        # --------------------------------
        # 5. Last 60 days
        # --------------------------------

        last_60_days = scaled_prices[
            -SEQUENCE_LENGTH:
        ]

        X = last_60_days.reshape(
            1,
            SEQUENCE_LENGTH,
            1
        )

        # --------------------------------
        # 6. RNN prediction
        # --------------------------------

        print("Running RNN prediction...")

        prediction = model.predict(
            X,
            verbose=0
        )

        # --------------------------------
        # 7. Convert back to price
        # --------------------------------

        predicted_price = scaler.inverse_transform(
            prediction
        )

        predicted_price = float(
            predicted_price[0][0]
        )

        print(
            f"Predicted price: ₹{predicted_price:.2f}"
        )

        # --------------------------------
        # 8. Calculate movement
        # --------------------------------

        price_difference = (
            predicted_price - latest_price
        )

        percentage_change = (
            price_difference / latest_price
        ) * 100

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

        # --------------------------------
        # 9. Return JSON
        # --------------------------------

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

        print("\nPREDICTION ERROR:")
        print(str(e))

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    print("\n==============================================")
    print("       RNN STOCK PREDICTION WEB APP")
    print("==============================================")

    print("\nStarting Flask server...")

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )