# ============================================================
# RNN STOCK PRICE PREDICTION
# Prediction Script
# ============================================================

import numpy as np
import pandas as pd
import yfinance as yf
import joblib

from tensorflow.keras.models import load_model


# ============================================================
# 1. SETTINGS
# ============================================================

TICKER = "RELIANCE.NS"

SEQUENCE_LENGTH = 60

MODEL_PATH = "models/stock_rnn.keras"

SCALER_PATH = "models/scaler.pkl"


# ============================================================
# 2. DOWNLOAD LATEST STOCK DATA
# ============================================================

print("\n==============================================")
print("        STOCK PRICE PREDICTION")
print("==============================================")

print("\nDownloading latest stock data...")

data = yf.download(
    TICKER,
    period="6mo",
    interval="1d",
    auto_adjust=True
)


# ============================================================
# 3. CHECK DATA
# ============================================================

if data.empty:

    print("\nCould not download stock data.")

    exit()


print(
    "Downloaded",
    len(data),
    "records."
)


# ============================================================
# 4. EXTRACT CLOSING PRICE
# ============================================================

print("\nExtracting closing prices...")


prices = data["Close"]


# Handle yfinance multi-column format
if isinstance(prices, pd.DataFrame):

    prices = prices.iloc[:, 0]


prices = pd.to_numeric(
    prices,
    errors="coerce"
)


prices = prices.dropna()


prices = prices.values.reshape(-1, 1)


print(
    "Valid closing prices:",
    len(prices)
)


# ============================================================
# 5. CHECK WHETHER WE HAVE ENOUGH DATA
# ============================================================

if len(prices) < SEQUENCE_LENGTH:

    print(
        "\nNot enough data to make a prediction."
    )

    exit()


# ============================================================
# 6. DISPLAY LATEST PRICE
# ============================================================

latest_price = prices[-1][0]

print(
    f"\nLatest closing price: ₹{latest_price:.2f}"
)


# ============================================================
# 7. LOAD THE TRAINING SCALER
# ============================================================

print("\nLoading saved scaler...")

scaler = joblib.load(
    SCALER_PATH
)

print("Scaler loaded successfully.")


# ============================================================
# 8. SCALE THE LATEST DATA
# ============================================================

scaled_prices = scaler.transform(
    prices
)


# ============================================================
# 9. GET LAST 60 TRADING DAYS
# ============================================================

last_60_days = scaled_prices[
    -SEQUENCE_LENGTH:
]


# ============================================================
# 10. RESHAPE DATA FOR RNN
# ============================================================

X = last_60_days.reshape(
    1,
    SEQUENCE_LENGTH,
    1
)


print(
    "\nInput shape for RNN:",
    X.shape
)


# ============================================================
# 11. LOAD TRAINED MODEL
# ============================================================

print("\nLoading trained RNN model...")

model = load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# 12. MAKE PREDICTION
# ============================================================

print("\nMaking prediction...")

prediction = model.predict(
    X,
    verbose=0
)


# ============================================================
# 13. CONVERT PREDICTION BACK TO RUPEES
# ============================================================

predicted_price = scaler.inverse_transform(
    prediction
)


predicted_price = predicted_price[0][0]


# ============================================================
# 14. CALCULATE PRICE CHANGE
# ============================================================

price_difference = (
    predicted_price - latest_price
)


percentage_change = (
    price_difference / latest_price
) * 100


# ============================================================
# 15. DISPLAY FINAL RESULT
# ============================================================

print("\n==============================================")
print("          STOCK PRICE PREDICTION")
print("==============================================")

print(
    f"Latest price       : ₹{latest_price:.2f}"
)

print(
    f"Predicted next price: ₹{predicted_price:.2f}"
)

print(
    f"Expected change    : ₹{price_difference:.2f}"
)

print(
    f"Percentage change  : {percentage_change:.2f}%"
)


if predicted_price > latest_price:

    print("\nPrediction: 📈 PRICE MAY INCREASE")

elif predicted_price < latest_price:

    print("\nPrediction: 📉 PRICE MAY DECREASE")

else:

    print("\nPrediction: ➡️ PRICE MAY REMAIN SIMILAR")


print("==============================================")

print("\nPrediction completed successfully!")