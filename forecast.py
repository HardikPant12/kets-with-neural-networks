import yfinance as yf
import torch
import torch.nn as nn
import numpy as np
import plotly.graph_objs as go
import plotly.offline as pyo
from model import LSTMForecast
import os

def prepare_data(data, seq_length=60):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def forecast_stock(ticker, forecast_days):
    df = yf.download(ticker, period="2y")
    prices = df["Close"].values
    prices_norm = (prices - prices.mean()) / prices.std()
    
    seq_len = 60
    X, y = prepare_data(prices_norm, seq_len)

    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    model = LSTMForecast(1, 64, 2, 1)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(20):
        output = model(X_tensor)
        loss = loss_fn(output.squeeze(), y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    forecast = []
    last_seq = X_tensor[-1].unsqueeze(0)

    for _ in range(forecast_days):
        pred = model(last_seq).item()
        forecast.append(pred)
        new_seq = torch.cat((last_seq[:, 1:, :], torch.tensor([[[pred]]])), dim=1)
        last_seq = new_seq

    forecast_actual = [p * prices.std() + prices.mean() for p in forecast]
    forecast_dates = pd.date_range(df.index[-1], periods=forecast_days+1, freq="B")[1:]

    actual_prices = prices[-100:]
    actual_dates = df.index[-100:]

    trace_actual = go.Scatter(x=actual_dates, y=actual_prices, mode='lines', name='Actual')
    trace_forecast = go.Scatter(x=forecast_dates, y=forecast_actual, mode='lines+markers', name='Forecast')
    layout = go.Layout(title=f"{ticker.upper()} Price Forecast", xaxis=dict(title='Date'), yaxis=dict(title='Price'))
    fig = go.Figure(data=[trace_actual, trace_forecast], layout=layout)

    plot_path = "static/forecast.html"
    pyo.plot(fig, filename=plot_path, auto_open=False)

    return {
        "message": f"Forecasted next {forecast_days} days for {ticker.upper()}",
        "plot_path": plot_path
    }