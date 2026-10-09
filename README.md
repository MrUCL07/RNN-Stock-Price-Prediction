# 📈 RNN Stock Price Prediction

An end-to-end stock price prediction project using **SimpleRNN, TensorFlow, NumPy, and Flask** to forecast the next closing price of Reliance Industries (NSE: `RELIANCE.NS`) based on historical stock market data.

The project combines deep learning, data preprocessing, lightweight model inference, and web application development in a single pipeline.

## 🚀 Project Overview

This project uses a Recurrent Neural Network (RNN) to learn patterns from historical stock prices and generate a next-step price prediction. An interactive Flask dashboard presents the latest closing price, predicted price, and expected percentage movement.

A key optimization was reproducing the trained RNN's forward pass using **NumPy**, enabling lightweight inference without loading TensorFlow in the deployed web application.

## ✨ Features

* 📊 Historical stock data processing
* 🧹 Data preprocessing using `MinMaxScaler`
* 🧠 Two-layer SimpleRNN deep learning architecture
* ⏳ Uses the previous 60 trading days as input
* 📉 Model evaluation using RMSE and MAE
* ⚡ NumPy-based inference using exported model weights
* 🌐 Interactive Flask web dashboard
* 📈 Displays latest price, predicted price, and expected movement
* 🚀 Deployment-ready using Gunicorn and Render

## 🛠️ Tech Stack

| Technology         | Purpose                            |
| ------------------ | ---------------------------------- |
| Python             | Core programming language          |
| TensorFlow / Keras | Model training                     |
| SimpleRNN          | Sequential stock-price modelling   |
| NumPy              | Lightweight model inference        |
| Pandas             | Data processing                    |
| Scikit-learn       | Data scaling and evaluation        |
| Matplotlib         | Visualizing training results       |
| yfinance           | Downloading historical stock data  |
| Flask              | Web application and prediction API |
| Gunicorn           | Production WSGI server             |
| Git & GitHub       | Version control                    |
| Render             | Web application hosting            |

## 🧠 Model Architecture

The model is built using TensorFlow/Keras.

* **Input:** Previous 60 trading days
* **First recurrent layer:** SimpleRNN with 50 units, returning sequences
* **Dropout:** 20%
* **Second recurrent layer:** SimpleRNN with 50 units
* **Dropout:** 20%
* **Dense layer:** 25 units
* **Output layer:** 1 predicted scaled price
* **Optimizer:** Adam
* **Loss function:** Mean Squared Error
* **Training epochs:** 20

The trained model learns temporal patterns in historical closing prices to estimate the next closing price.

## ⚡ NumPy Inference Optimization

During deployment, TensorFlow-based inference encountered resource and worker-timeout issues.

To address this, I exported the trained SimpleRNN and Dense layer weights and implemented the forward pass using NumPy.

The NumPy implementation was verified against TensorFlow:

* TensorFlow output: `0.3525329530`
* NumPy output: `0.3525328934`
* Absolute difference: approximately `0.0000000596`

This verification demonstrates that the NumPy implementation closely reproduces the trained model's output for the tested input.

## 🔄 Project Workflow

1. Download historical stock data.
2. Clean and preprocess closing prices.
3. Scale the data using MinMaxScaler.
4. Create sequences of 60 trading days.
5. Train the SimpleRNN model.
6. Evaluate model predictions.
7. Export trained weights for NumPy inference.
8. Verify NumPy predictions against TensorFlow.
9. Serve predictions through a Flask application.
10. Deploy the application using Gunicorn and Render.

## 📂 Project Structure

```text
RNN-Stock-Price-Prediction/
│
├── app.py
├── download_data.py
├── train_model.py
├── export_model.py
├── predict_numpy.py
├── test_numpy_model.py
├── requirements.txt
├── .python-version
│
├── data/
│   └── RELIANCE.csv
│
├── models/
│   ├── stock_rnn.keras
│   ├── scaler.pkl
│   └── rnn_weights.npz
│
├── templates/
│   └── index.html
│
└── README.md
```

*Note: The exact files in your repository may differ depending on which training artifacts and datasets you choose to include.*

## 💻 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/NEW_USERNAME/RNN-Stock-Price-Prediction.git
cd RNN-Stock-Price-Prediction
```

Replace `NEW_USERNAME` with your GitHub username.

### 2. Create a virtual environment

**Windows:**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Start the application

```bash
python app.py
```

### 5. Open the dashboard

Visit:

```text
http://127.0.0.1:5000
```

Click **Run prediction** to request a prediction from the local model.

**Important:** The application requires the stock CSV, scaler, NumPy weights, and Flask template at the paths expected by `app.py`.

## 📊 Evaluation

The model training pipeline uses:

* **RMSE (Root Mean Squared Error):** Measures the typical scale of prediction errors, with larger errors penalized more heavily.
* **MAE (Mean Absolute Error):** Measures the average absolute prediction error.

Refer to the training output and generated evaluation plots for the actual results. Stock-price predictions are experimental and should not be treated as guaranteed outcomes.

## 🔮 Future Improvements

* Support for additional stocks and market indices
* Compare RNN with LSTM and GRU architectures
* Improve evaluation using chronological walk-forward validation
* Add prediction-versus-actual visualizations
* Track model performance over time
* Improve data-refresh automation
* Add automated tests and deployment monitoring

## 🎯 What I Learned

Through this project, I gained practical experience with:

* Sequential data and recurrent neural networks
* Time-series preprocessing and model evaluation
* Exporting trained neural network weights
* Reproducing model inference with NumPy
* Building REST-style prediction endpoints using Flask
* Debugging deployment resource limitations
* Version control and cloud deployment workflows

## ⚠️ Disclaimer

This project is intended for educational purposes only. Stock markets are affected by many factors that historical price data alone cannot capture. Predictions are not financial advice and should not be used as the sole basis for investment decisions.

## 👨‍💻 Author

**Your Name**

GitHub: [Your GitHub Profile](https://github.com/NEW_USERNAME)

---

⭐ If you find this project interesting, consider giving the repository a star!
