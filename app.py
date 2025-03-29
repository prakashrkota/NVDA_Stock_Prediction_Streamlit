#+--------------------------------------------------------------------------------------------+
# Model for Stock Price Prediction via Dense only Neural Network
# Using today vs tomorrow analysis
# 
# Written by: Prakash R. Kota
# Location: East Greenbush, NY
#
# Using just a Dense Neural Network Model
# Adding lots of direct stock parmeters from Yahoo Finance
# Open, High, Low, Close, Volume
#
# Will not be using other parameters such as
# Return, SMA10, EMA10, RollingVol10, 
# SP500, Nasdaq, VIX
# RSI, Day-of-Week
# Removed all saving and graphing from PRK_1a_tf_Stock_NN.ipnyb
# Keeping it minimal for just the stock prediction and output display table
# Goal is to make a MVP - Minimal Viable Product
# Infer the NN Model from the Saved Model - this is the app.py required for HF
#
# PRK_1b_2_Model_Infer_tf_Stock_NN.ipnyb
# based on
# PRK_1b_tf_Stock_NN.ipnyb
# PRK_1a_tf_Stock_NN.ipnyb
# PRK_11b_tf_Stock_DenseOnly.ipynb
# Renamed PRK_10e_tf_Stock_DenseOnly.ipynb for convenience
# This is the best Notebook Code for the NN Model
#
# Written on: 28 Mar 2025
# Last update: 28 Mar 2025
#+--------------------------------------------------------------------------------------------+

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pandas.tseries.offsets import BDay
import os
import tensorflow as tf
import joblib

# --- Load model and scalers --- #
model_dir = os.path.join(os.path.dirname(__file__), "model")

NN_model = tf.keras.models.load_model(os.path.join(model_dir, "NN_model.keras"))
NN_model.compile(optimizer="adam", loss="mse")
NN_model.predict(np.zeros((1, 5)))  # warm-up dummy prediction

scaler_X = joblib.load(os.path.join(model_dir, "scaler_X.pkl"))
scaler_y = joblib.load(os.path.join(model_dir, "scaler_y.pkl"))

# --- Prediction Function --- #
def predict_stock():
    # Clear yfinance cache
    cache_path = os.path.expanduser("~/.cache/py-yfinance")
    if os.path.exists(cache_path):
        import shutil
        shutil.rmtree(cache_path)

    Stock = "NVDA"
    start_date = "2020-01-01"
    train_end_date = "2024-12-31"
    today = datetime.today().strftime('%Y-%m-%d')

    full_data = yf.download(
        tickers=Stock,
        start=start_date,
        end=today,
        interval="1d",
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False
    )

    features = ["Open", "High", "Low", "Close", "Volume"]
    X_scaled = scaler_X.transform(full_data[features])
    y = full_data["Close"].values.reshape(-1, 1)
    y_scaled = scaler_y.transform(y)

    X_all = X_scaled[:-1]
    y_all = y_scaled[1:].flatten()
    dates_all = full_data.index[1:]

    test_mask = dates_all > pd.to_datetime(train_end_date)
    X_test, y_test = X_all[test_mask], y_all[test_mask]
    dates_test = dates_all[test_mask]

    y_pred_scaled = NN_model.predict(X_test).flatten()
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    last_date = full_data.index[-1]
    last_close_price = float(full_data["Close"].iloc[-1].item())
    X_input = X_scaled[-1].reshape(1, -1)

    next_day_pred_scaled = NN_model.predict(X_input)
    next_day_pred = scaler_y.inverse_transform(next_day_pred_scaled)[0][0]

    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    std_ape = np.std(np.abs((y_true - y_pred) / y_true)) * 100

    mape_margin = next_day_pred * (mape / 100)
    sae_margin = next_day_pred * (std_ape / 100)

    next_date = (last_date + BDay(1)).date()

    summary = f"""
Prediction for {Stock}:
Last available date: {last_date.date()}, Close = ${last_close_price:.2f}
Predicted closing price for next trading day ({next_date}): ${next_day_pred:.2f}
Expected range (\u00b1MAPE): ${next_day_pred - mape_margin:.2f} to ${next_day_pred + mape_margin:.2f}
Expected range (\u00b1SAE):  ${next_day_pred - sae_margin:.2f} to ${next_day_pred + sae_margin:.2f}
"""

    prediction_df = pd.DataFrame({
        'Date': dates_test,
        'Actual Close': y_true,
        'Predicted Close': y_pred
    })
    prediction_df['% Error'] = ((prediction_df['Actual Close'] - prediction_df['Predicted Close']) / prediction_df['Actual Close']) * 100
    prediction_df['% Error'] = prediction_df['% Error'].map(lambda x: f"{x:+.2f}%")
    prediction_df['±MAPE Range'] = prediction_df['Predicted Close'].apply(
        lambda x: f"${x * (1 - mape/100):.2f} to ${x * (1 + mape/100):.2f}"
    )

    prediction_df['Date'] = prediction_df['Date'].dt.strftime("%Y-%m-%d")
    prediction_df['Actual Close'] = prediction_df['Actual Close'].map(lambda x: f"${x:.2f}")
    prediction_df['Predicted Close'] = prediction_df['Predicted Close'].map(lambda x: f"${x:.2f}")
    prediction_df = prediction_df.sort_values("Date", ascending=False)

    return summary, prediction_df[["Date", "Actual Close", "Predicted Close", "% Error", "±MAPE Range"]]

# --- Streamlit UI --- #
st.set_page_config(page_title="📈 NVDA Stock Predictor")
st.title("📈 NVDA Stock Predictor")
st.write("This app uses a Dense Neural Network to predict NVDA's next trading day's closing price.")

if st.button("Predict Now"):
    summary, table = predict_stock()
    st.text(summary)
    st.dataframe(table)
