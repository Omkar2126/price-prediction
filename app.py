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

st.sidebar.header("Stock Configuration")
ticker = st.sidebar.text_input("Enter Stock Ticker Symbol:", value="AAPL").upper()

st.subheader("Select Timeframe")
timeframe = st.segmented_control(
    label="Choose Period Scope:",
    options=["1 Month", "3 Months", "6 Months", "1 Year", "3 Years", "5 Years"],
    default="1 Year",
    label_visibility="collapsed"
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

if st.sidebar.button("Run AI Prediction Model", use_container_width=True):
    with st.spinner(f"Fetching data and training model for {ticker}..."):
        try:
            # FIX: Force yfinance to output flat columns without complex multi-index headers
            data = yf.download(ticker, start=start_date, end=end_date, multi_level_index=False)
            
            if data.empty or len(data) < 2:
                st.error("⚠️ Not enough data found to train the AI! Please try a wider timeframe or check your ticker.")
            else:
                # Double-safetly: Ensure flat column labels in case of package variations
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)

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
                
                st.subheader("🤖 AI Technical Opinion")
                
                price_difference = next_day_prediction - latest_price
                percent_change = (price_difference / latest_price) * 100
                
                historical_trend_slope = model.coef_[0]
                
                if percent_change > 0.75:
                    opinion = "STRONG BUY"
                    color = "green"
                    explanation = f"The AI model detects a significant short-term upward momentum. The predicted next-day target represents a projected gain of **{percent_change:.2f}%**."
                elif percent_change > 0.10:
                    opinion = "BUY"
                    color = "lightgreen"
                    explanation = f"The AI models a positive marginal increase of **{percent_change:.2f}%** for the next session. Trend alignment remains generally constructive."
                elif percent_change < -0.75:
                    opinion = "STRONG SELL"
                    color = "red"
                    explanation = f"The mathematical model projects a sharp immediate drop of **{abs(percent_change):.2f}%**. Technical indicators point toward heavy overhead resistance."
                elif percent_change < -0.10:
                    opinion = "SELL"
                    color = "orange"
                    explanation = f"The regression matrix predicts a down-swing of **{abs(percent_change):.2f}%** tomorrow. Risk parameters suggest reducing exposure or pausing additions."
                else:
                    opinion = "HOLD"
                    color = "blue"
                    explanation = f"The next-day prediction is practically flat (**{percent_change:.2f}%** variance). The asset is currently consolidating, suggesting it's best to observe current positions."

                if historical_trend_slope > 1:
                    trend_context = "Furthermore, looking across the historical timeframe, the stock has a strong macro upward trajectory."
                elif historical_trend_slope < -1:
                    trend_context = "Caution: The overarching macro trend for this selected historical scope is fundamentally downward."
                else:
                    trend_context = "The historical range indicates a sideways market structure with low directional bias."

                st.markdown(f"### Strategy Recommendation: :{color}[**{opinion}**]")
                st.info(f"{explanation} {trend_context}")
                
                st.subheader("Interactive Candlestick Analytics")
                
                df_chart = df.reset_index()
                df_chart['Date_Str'] = df_chart['Date'].dt.strftime('%Y-%m-%d')
                
                fig = go.Figure()
                
                # Plotly now receives perfectly isolated 1D arrays for open/high/low/close vectors
                fig.add_trace(go.Candlestick(
                    x=df_chart['Date_Str'],
                    open=df_chart['Open'].to_numpy().flatten(),
                    high=df_chart['High'].to_numpy().flatten(),
                    low=df_chart['Low'].to_numpy().flatten(),
                    close=df_chart['Close'].to_numpy().flatten(),
                    name="Market Price"
                ))
                
                fig.update_layout(
                    xaxis_title="Date Timeline",
                    yaxis_title=f"Price ({currency_symbol})",
                    xaxis_rangeslider_visible=False,
                    template="plotly_white",
                    margin=dict(l=10, r=10, t=10, b=10)
                )
                
                fig.update_xaxes(
                    type="category",
                    tickangle=45,
                    nticks=15,
                    showgrid=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"An unexpected pipeline error occurred: {e}")
else:
    st.info("👈 Enter your stock ticker on the left sidebar, select a timeline period tab above, and hit 'Run AI Prediction Model'.")