# ============================================================
# RNN STOCK PRICE PREDICTION
# Training Script
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout


# ============================================================
# 1. SETTINGS
# ============================================================

FILE_PATH = "data/RELIANCE.csv"

SEQUENCE_LENGTH = 60

TRAIN_SPLIT = 0.80

EPOCHS = 20

BATCH_SIZE = 32


# ============================================================
# 2. CREATE REQUIRED FOLDERS
# ============================================================

os.makedirs("plots", exist_ok=True)
os.makedirs("models", exist_ok=True)


# ============================================================
# 3. LOAD DATASET
# ============================================================

print("\n==============================================")
print("       RNN STOCK PRICE PREDICTION")
print("==============================================")

print("\nLoading stock dataset...")

data = pd.read_csv(
    FILE_PATH,
    skiprows=[1, 2]
)

print("\nFirst 5 rows:")
print(data.head())

print("\nTotal price records:", len(data))


# ============================================================
# 4. CLEAN DATA
# ============================================================

print("\nCleaning data...")

# Convert Close column to numbers
data["Close"] = pd.to_numeric(
    data["Close"],
    errors="coerce"
)

# Remove missing values
data = data.dropna(subset=["Close"])

# Convert Close prices to NumPy array
prices = data["Close"].values.reshape(-1, 1)

print("Valid price records:", len(prices))


# ============================================================
# 5. VISUALIZE STOCK PRICE
# ============================================================

print("\nCreating stock price graph...")

plt.figure(figsize=(12, 6))

plt.plot(prices)

plt.title("Reliance Industries Stock Price")
plt.xlabel("Trading Days")
plt.ylabel("Closing Price (₹)")

plt.grid(True)

plt.tight_layout()

# Save graph instead of opening a window
plt.savefig("plots/stock_price.png")

plt.close()

print("Stock price graph saved:")
print("plots/stock_price.png")


# ============================================================
# 6. SCALE DATA
# ============================================================

print("\nScaling stock prices...")

scaler = MinMaxScaler(
    feature_range=(0, 1)
)

scaled_prices = scaler.fit_transform(prices)


# ============================================================
# 7. SAVE SCALER
# ============================================================

import joblib

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

print("Scaler saved:")
print("models/scaler.pkl")


# ============================================================
# 8. CREATE SEQUENCES
# ============================================================

print("\nCreating sequences...")

def create_sequences(data, sequence_length):

    X = []
    y = []

    for i in range(sequence_length, len(data)):

        X.append(
            data[i - sequence_length:i]
        )

        y.append(
            data[i]
        )

    return np.array(X), np.array(y)


X, y = create_sequences(
    scaled_prices,
    SEQUENCE_LENGTH
)

print("X shape:", X.shape)
print("y shape:", y.shape)


# ============================================================
# 9. TRAIN / TEST SPLIT
# ============================================================

print("\nSplitting data into training and testing sets...")

train_size = int(
    len(X) * TRAIN_SPLIT
)

X_train = X[:train_size]

X_test = X[train_size:]

y_train = y[:train_size]

y_test = y[train_size:]


print("\nTraining samples:", len(X_train))

print("Testing samples:", len(X_test))


# ============================================================
# 10. BUILD RNN MODEL
# ============================================================

print("\nBuilding SimpleRNN model...")

model = Sequential([

    SimpleRNN(
        50,
        return_sequences=True,
        input_shape=(SEQUENCE_LENGTH, 1)
    ),

    Dropout(0.2),

    SimpleRNN(
        50,
        return_sequences=False
    ),

    Dropout(0.2),

    Dense(25),

    Dense(1)

])


# ============================================================
# 11. COMPILE MODEL
# ============================================================

model.compile(

    optimizer="adam",

    loss="mean_squared_error"

)


print("\nModel architecture:")

model.summary()


# ============================================================
# 12. TRAIN MODEL
# ============================================================

print("\n==============================================")
print("              STARTING TRAINING")
print("==============================================")

history = model.fit(

    X_train,

    y_train,

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    validation_split=0.1,

    verbose=1

)


print("\nTraining completed!")


# ============================================================
# 13. SAVE MODEL
# ============================================================

model.save(
    "models/stock_rnn.keras"
)

print("\nModel saved successfully!")

print(
    "Location: models/stock_rnn.keras"
)


# ============================================================
# 14. MAKE TEST PREDICTIONS
# ============================================================

print("\nMaking predictions on test data...")

predictions = model.predict(
    X_test,
    verbose=0
)


# ============================================================
# 15. CONVERT PREDICTIONS BACK TO RUPEES
# ============================================================

predicted_prices = scaler.inverse_transform(
    predictions
)

actual_prices = scaler.inverse_transform(
    y_test
)


# ============================================================
# 16. CALCULATE RMSE
# ============================================================

rmse = np.sqrt(
    mean_squared_error(
        actual_prices,
        predicted_prices
    )
)


# ============================================================
# 17. CALCULATE MAE
# ============================================================

mae = mean_absolute_error(
    actual_prices,
    predicted_prices
)


# ============================================================
# 18. DISPLAY RESULTS
# ============================================================

print("\n==============================================")
print("              MODEL RESULTS")
print("==============================================")

print(
    f"RMSE: ₹{rmse:.2f}"
)

print(
    f"MAE : ₹{mae:.2f}"
)


# ============================================================
# 19. PLOT ACTUAL VS PREDICTED PRICES
# ============================================================

print("\nCreating prediction graph...")

plt.figure(figsize=(12, 6))

plt.plot(
    actual_prices,
    label="Actual Price"
)

plt.plot(
    predicted_prices,
    label="Predicted Price"
)

plt.title(
    "Reliance Stock Price - Actual vs Predicted"
)

plt.xlabel("Trading Days")

plt.ylabel("Closing Price (₹)")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "plots/prediction.png"
)

plt.close()


print(
    "Prediction graph saved:"
)

print(
    "plots/prediction.png"
)


# ============================================================
# 20. SAVE TRAINING GRAPH
# ============================================================

print("\nCreating training loss graph...")

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title(
    "RNN Training and Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "plots/training_loss.png"
)

plt.close()


print(
    "Training graph saved:"
)

print(
    "plots/training_loss.png"
)


# ============================================================
# 21. FINAL MESSAGE
# ============================================================

print("\n==============================================")
print("             PROJECT COMPLETED")
print("==============================================")

print("\nFiles created:")

print("1. models/stock_rnn.keras")

print("2. models/scaler.pkl")

print("3. plots/stock_price.png")

print("4. plots/prediction.png")

print("5. plots/training_loss.png")

print("\nYour RNN model has been trained successfully!")

print("==============================================")