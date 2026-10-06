from flask import Flask, render_template, jsonify
import numpy as np
import pandas as pd
import joblib
import os

from tensorflow.keras.models import load_model


# ======================================
# Flask App
# ======================================

app = Flask(__name__)


# ======================================
# Configuration
# ======================================

TICKER = "RELIANCE.NS"
SEQUENCE_LENGTH = 60

MODEL_PATH = "models/stock_rnn.keras"
SCALER_PATH = "models/scaler.pkl"
LOCAL_DATA_PATH = "data/RELIANCE.csv"


# ======================================
# Load RNN Model
# ======================================

print("\nLoading RNN model...")

model = load_model(MODEL_PATH)

print("RNN model loaded successfully.")


# ======================================
# Load Scaler
# ======================================

print("\nLoading scaler...")

scaler = joblib.load(SCALER_PATH)

print("Scaler loaded successfully.")


# ======================================
# Home Page
# ======================================

@app.route("/")
def home():
    return render_template("index.html")


# ======================================
# Load Local Stock Data
# ======================================

def get_stock_data():

    print("\n======================================")
    print("Loading local stock data...")
    print("======================================")

    if not os.path.exists(LOCAL_DATA_PATH):

        raise FileNotFoundError(
            "Local stock data file was not found."
        )

    # Load CSV
    data = pd.read_csv(
        LOCAL_DATA_PATH,
        skiprows=[1, 2]
    )

    print("Local CSV loaded successfully.")

    print("Total rows:", len(data))

    return data


# ======================================
# Prediction API
# ======================================

@app.route("/predict")
def predict():

    try:

        print("\n======================================")
        print("Starting stock prediction...")
        print("======================================")


        # --------------------------------
        # 1. Load stock data
        # --------------------------------

        data = get_stock_data()


        if data.empty:

            return jsonify({
                "success": False,
                "error": "No stock data available."
            }), 500


        # --------------------------------
        # 2. Get Close price
        # --------------------------------

        prices = data["Close"]


        # Handle possible DataFrame
        if isinstance(prices, pd.DataFrame):

            prices = prices.iloc[:, 0]


        # Convert to numeric
        prices = pd.to_numeric(
            prices,
            errors="coerce"
        )


        # Remove missing values
        prices = prices.dropna()


        # Convert to NumPy
        prices = prices.values.reshape(
            -1,
            1
        )


        # --------------------------------
        # 3. Check data length
        # --------------------------------

        if len(prices) < SEQUENCE_LENGTH:

            return jsonify({
                "success": False,
                "error": (
                    "Not enough stock data "
                    "for prediction."
                )
            }), 500


        # --------------------------------
        # 4. Latest stock price
        # --------------------------------

        latest_price = float(
            prices[-1][0]
        )

        print(
            f"Latest price: ₹{latest_price:.2f}"
        )


        # --------------------------------
        # 5. Scale prices
        # --------------------------------

        scaled_prices = scaler.transform(
            prices
        )


        # --------------------------------
        # 6. Select last 60 days
        # --------------------------------

        last_60_days = scaled_prices[
            -SEQUENCE_LENGTH:
        ]


        # --------------------------------
        # 7. Reshape for RNN
        # --------------------------------

        X = last_60_days.reshape(
            1,
            SEQUENCE_LENGTH,
            1
        )


        print(
            "Input shape:",
            X.shape
        )


        # --------------------------------
        # 8. Run RNN prediction
        # --------------------------------

        print(
            "Running RNN prediction..."
        )

        prediction = model.predict(
            X,
            verbose=0
        )


        # --------------------------------
        # 9. Convert prediction
        #    back to original price
        # --------------------------------

        predicted_price = (
            scaler.inverse_transform(
                prediction
            )
        )


        predicted_price = float(
            predicted_price[0][0]
        )


        print(
            f"Predicted price: "
            f"₹{predicted_price:.2f}"
        )


        # --------------------------------
        # 10. Calculate price difference
        # --------------------------------

        price_difference = (
            predicted_price -
            latest_price
        )


        # --------------------------------
        # 11. Calculate percentage change
        # --------------------------------

        percentage_change = (
            price_difference /
            latest_price
        ) * 100


        # --------------------------------
        # 12. Determine direction
        # --------------------------------

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
        # 13. Return JSON response
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


    # ==================================
    # Error Handling
    # ==================================

    except Exception as e:

        print("\n======================================")
        print("PREDICTION ERROR")
        print("======================================")

        print(str(e))


        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# ======================================
# Run Flask App
# ======================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )


    print("\n")
    print("==============================================")
    print("       RNN STOCK PREDICTION WEB APP")
    print("==============================================")


    print("\nStarting Flask server...")


    app.run(

        host="0.0.0.0",

        port=port,

        debug=False

    )