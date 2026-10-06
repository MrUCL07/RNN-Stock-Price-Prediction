
import yfinance as yf
import os

# 1. Select the stock
ticker = "RELIANCE.NS"

# 2. Download 10 years of daily stock data
print("Downloading stock data...")

data = yf.download(
    ticker,
    period="10y",
    interval="1d",
    auto_adjust=True
)

# 3. Check whether data was downloaded
if data.empty:
    print("No data downloaded. Check your internet connection.")
else:
    # 4. Create a folder to store the dataset
    os.makedirs("data", exist_ok=True)

    # 5. Save the dataset as a CSV file
    data.to_csv("data/RELIANCE.csv")

    print("\nDownload successful!")
    print("Total rows:", len(data))
    print("\nFirst five rows:")
    print(data.head())
    print("\nSaved to: data/RELIANCE.csv")