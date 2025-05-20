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
    # Ensure data types after download and dropna
    if not df.empty:
        for col in ["High", "Low", "Close", "Open"]: # Added Open for candlestick
            if col in df.columns:
                df[col] = df[col].astype(float)
    return df

def find_bos(df):
    # Initialize BOS column with False, ensuring it's aligned with df's index from the start
    df["BOS"] = pd.Series(False, index=df.index, dtype=bool)

    if len(df) < 3: # Not enough data for meaningful shift(2)
        return df # BOS is already all False, and correctly indexed

    # Use .loc to assign to avoid potential SettingWithCopyWarning and ensure view/copy semantics are clear
    # Calculate conditions directly. NaNs from shifts will be handled by fillna(False).
    high = df["High"]
    low = df["Low"]
    
    # Condition for Higher High Break of Structure
    # Current high is greater than previous high, AND current high is greater than high two periods ago
    cond_hh_bos = (high > high.shift(1)) & (high > high.shift(2))
    
    # Condition for Lower Low Break of Structure
    # Current low is less than previous low, AND current low is less than low two periods ago
    cond_ll_bos = (low < low.shift(1)) & (low < low.shift(2))
    
    # Combine conditions. fillna(False) handles NaNs from shifts, ensuring they don't break the OR logic.
    # The resulting series 'combined_conditions' will be aligned with df.index.
    combined_conditions = cond_hh_bos.fillna(False) | cond_ll_bos.fillna(False)
    
    # Assign the boolean results to the "BOS" column.
    # This assignment should be safe as combined_conditions is a boolean Series aligned with df.index.
    df.loc[:, "BOS"] = combined_conditions.astype(bool)
    
    return df

def detect_fvg(df):
    # Fair Value Gap detection
    # Note: Your FVG logic is specific. A common FVG might be:
    # Bullish: df['Low'] > df['High'].shift(2) (gap above previous high)
    # Bearish: df['High'] < df['Low'].shift(2) (gap below previous low)
    # Current logic: ((df["Low"].shift(-1) > df["High"]) & (df["Low"].shift(-1) > df["High"].shift(2)))
    # This identifies a specific 3-candle pattern.
    if len(df) < 3: # Needs at least 3 candles for shift(-1) and shift(2) relative to a middle candle
        df["FVG"] = pd.Series([False] * len(df), index=df.index, dtype=bool)
        return df
        
    # Calculate FVG based on your logic
    # Ensure Series are aligned and handle potential NaNs from shifts by converting to bool (False for NaN)
    low_shift_neg1 = df["Low"].shift(-1)
    high_current = df["High"]
    high_shift_2 = df["High"].shift(2)

    # Perform comparisons, results will be boolean Series (NaNs become False)
    cond1 = low_shift_neg1 > high_current
    cond2 = low_shift_neg1 > high_shift_2
    
    fvg_series = cond1 & cond2
    
    # Reindex to ensure alignment, fill_value=False for rows where FVG can't be calculated
    df["FVG"] = fvg_series.reindex(df.index, fill_value=False).astype(bool)
    return df

def generate_signals(df):
    if df.empty:
        df["Signal"] = pd.Series(dtype='object') # Use object for None
        df["TP"] = pd.Series(dtype='float')
        return df

    df = find_bos(df)
    df = detect_fvg(df)
    
    # Ensure "BOS" and "FVG" columns exist and are boolean
    if "BOS" not in df.columns: df["BOS"] = False
    if "FVG" not in df.columns: df["FVG"] = False
    
    df["BOS"] = df["BOS"].astype(bool)
    df["FVG"] = df["FVG"].astype(bool)

    df["Signal"] = np.where((df["BOS"] & df["FVG"]), "BUY", None) # None will make dtype object
    
    # Calculate TP values using underlying NumPy arrays to ensure 1D operations
    # Condition for BUY signal
    buy_condition_arr = (df["Signal"].values == "BUY") # Results in a 1D boolean NumPy array
    
    # Values if condition is True
    # Ensure High and Low are treated as 1D NumPy arrays
    high_arr = df["High"].values
    low_arr = df["Low"].values
    tp_values_if_buy_arr = high_arr + (high_arr - low_arr) * 1.5 # Results in a 1D float NumPy array
    
    # Assign using np.where with NumPy arrays; result should be a 1D NumPy array
    df["TP"] = np.where(buy_condition_arr, tp_values_if_buy_arr, np.nan)
    
    df["TP"] = df["TP"].astype(float) # Ensure TP is float
    return df

def plot_chart(df):
    fig = go.Figure()
    if not df.empty and all(col in df.columns for col in ["Open", "High", "Low", "Close"]):
        fig.add_trace(go.Candlestick(x=df.index,
                                     open=df["Open"],
                                     high=df["High"],
                                     low=df["Low"],
                                     close=df["Close"],
                                     name="Candles"))
    # Plot signals
    if "Signal" in df.columns:
        buy_signals = df[df["Signal"] == "BUY"]
        if not buy_signals.empty:
            fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals["Low"],
                                     mode="markers", name="Buy Signal",
                                     marker=dict(color="green", size=10)))
            # TP areas
            if "TP" in buy_signals.columns:
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
    
    if df is None or df.empty:
        st.warning(f"No data loaded for {symbol} with period {period} and interval {interval}. Cannot generate signals or plot chart.")
        return # Stop execution if no data

    df = generate_signals(df)

    # Display chart
    fig = plot_chart(df)
    st.plotly_chart(fig, use_container_width=True)

    # Display table
    if "Signal" in df.columns and st.checkbox("Show Raw Signal Data"):
        st.dataframe(df[df["Signal"].notnull()][["Close", "Signal", "TP"]])

if __name__ == "__main__":
    main()
