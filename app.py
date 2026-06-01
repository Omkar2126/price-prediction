import sys
import asyncio

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="AI Stock Predictor", layout="centered", page_icon="📈")

st.title("📈 AI Stock Price Predictor")
st.write(
    "This application uses a Machine Learning **Linear Regression** model "
    "to predict the next day's closing stock price based on historical trends."
)

st.sidebar.header("User Input Parameters")
ticker = st.sidebar.text_input("Enter Stock Ticker Symbol:", value="AAPL").upper()

timeframe = st.sidebar.selectbox(
    "Select Historical Data Timeframe:",
    options=["1 Month", "3 Months", "6 Months", "1 Year", "3 Years", "5 Years"],
    index=3
)

end_date = datetime.today()
if timeframe == "1 Month":
    start_date = end_date - timedelta(days=30)
elif timeframe == "3 Months":
    start_date = end_date - timedelta(days=90)
elif timeframe == "6 Months":
    start_date = end_date - timedelta(days=180)
elif timeframe == "1 Year":
    start_date = end_date - timedelta(days=365)
elif timeframe == "3 Years":
    start_date = end_date - timedelta(days=365 * 3)
elif timeframe == "5 Years":
    start_date = end_date - timedelta(days=365 * 5)

if st.sidebar.button("Run AI Prediction Model"):
    with st.spinner(f"Fetching data and training model for {ticker}..."):
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            
            if data.empty or len(data) < 2:
                st.error("⚠️ Not enough data found to train the AI! Please try a wider timeframe or check your ticker.")
            else:
                df = data[['Open', 'High', 'Low', 'Close']].copy()
                df['Prediction'] = df['Close'].shift(-1)
                
                X = np.array(df[['Close']])[:-1]
                y = np.array(df['Prediction'])[:-1]
                
                model = LinearRegression()
                model.fit(X, y)
                
                latest_price = float(df['Close'].iloc[-1].item())
                next_day_prediction = float(model.predict([[latest_price]])[0].item())
                
                currency_symbol = "₹" if ticker.endswith((".NS", ".BO")) else "$"
                
                st.success("AI Model successfully trained!")
                
                col1, col2 = st.columns(2)
                col1.metric(label=f"Latest Closing Price ({ticker})", value=f"{currency_symbol}{latest_price:.2f}")
                col2.metric(label="AI Predicted Next Day Price", value=f"{currency_symbol}{next_day_prediction:.2f}")
                
                st.subheader("Interactive Candlestick Analytics")
                
                fig = go.Figure()
                
                fig.add_trace(go.Candlestick(
                    x=df.index,
                    open=df['Open'].values.flatten(),
                    high=df['High'].values.flatten(),
                    low=df['Low'].values.flatten(),
                    close=df['Close'].values.flatten(),
                    name="Market Price"
                ))
                
                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title=f"Price ({currency_symbol})",
                    xaxis_rangeslider_visible=True,
                    template="plotly_white",
                    margin=dict(l=10, r=10, t=10, b=10)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"An unexpected pipeline error occurred: {e}")
else:
    st.info("👈 Choose your stock asset and timeframe in the sidebar menu, then click 'Run AI Prediction Model' to execute.")