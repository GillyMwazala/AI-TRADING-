# Smart Money Concepts + ICT Trading System in Streamlit
# Features: BOS, FVG, Buy/Sell signals, TP Zones

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ========== FUNCTIONS ==========
def get_data(symbol, period="6mo", interval="1h"):
    df = yf.download(symbol, period=period, interval=interval)
    df.dropna(inplace=True)
    df["High"] = df["High"].astype(float)
    df["Low"] = df["Low"].astype(float)
    df["Close"] = df["Close"].astype(float)
    return df

def find_bos(df):
    # Detect Break of Structure: Higher Highs and Lower Lows
    df["HH"] = df["High"] > df["High"].shift(1)
    df["LL"] = df["Low"] < df["Low"].shift(1)
    df["BOS"] = np.where((df["HH"] & (df["High"] > df["High"].shift(2))) | 
                         (df["LL"] & (df["Low"] < df["Low"].shift(2))), True, False)
    return df

def detect_fvg(df):
    # Fair Value Gap detection (imbalance between candles)
    df["FVG"] = ((df["Low"].shift(-1) > df["High"]) & (df["Low"].shift(-1) > df["High"].shift(2)))
    return df

def generate_signals(df):
    df = find_bos(df)
    df = detect_fvg(df)
    df["Signal"] = np.where((df["BOS"] & df["FVG"]), "BUY", None)
    df["TP"] = np.where(df["Signal"] == "BUY", df["High"] + (df["High"] - df["Low"])*1.5, None)
    return df

def plot_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df["Open"],
                                 high=df["High"],
                                 low=df["Low"],
                                 close=df["Close"],
                                 name="Candles"))
    # Plot signals
    buy_signals = df[df["Signal"] == "BUY"]
    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals["Low"],
                             mode="markers", name="Buy Signal",
                             marker=dict(color="green", size=10)))
    # TP areas
    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals["TP"],
                             mode="markers", name="TP Target",
                             marker=dict(color="blue", symbol="x", size=8)))
    fig.update_layout(title="SMC + ICT Trading Signals", height=600)
    return fig

# ========== STREAMLIT APP ==========
def main():
    st.set_page_config(page_title="SMC ICT Trading System", layout="wide")
    st.title("Smart Money + ICT Based BTC Trading System")

    symbol = st.sidebar.selectbox("Select Symbol", ["BTC-USD", "ETH-USD", "SOL-USD"])
    period = st.sidebar.selectbox("Period", ["1mo", "3mo", "6mo", "1y"])
    interval = st.sidebar.selectbox("Interval", ["1h", "4h", "1d"])

    st.info("Using Break of Structure (BOS) and Fair Value Gaps (FVG) to generate signals")

    # Fetch and analyze data
    df = get_data(symbol, period, interval)
    df = generate_signals(df)

    # Display chart
    fig = plot_chart(df)
    st.plotly_chart(fig, use_container_width=True)

    # Display table
    if st.checkbox("Show Raw Signal Data"):
        st.dataframe(df[df["Signal"].notnull()][["Close", "Signal", "TP"]])

if __name__ == "__main__":
    main()
