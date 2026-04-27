import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, date
import requests
from requests.exceptions import RequestException, HTTPError
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import numpy as np
from keras.models import load_model, Sequential
from keras.layers import LSTM, Dense
import joblib
import os
import time
import json
import base64
import streamlit.components.v1 as components
from sklearn.preprocessing import MinMaxScaler
from contextlib import contextmanager
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

DEFAULT_ALPHA_VANTAGE_API_KEY = os.getenv("DEFAULT_ALPHA_VANTAGE_API_KEY")
MODEL_DIR = "models"
SEQ_LEN = 60
TRAIN_EPOCHS = 5
TRAIN_BATCH_SIZE = 32
LOADER_JSON_FILE = "Growth Chart.json"

# --- Page Config ---
st.set_page_config(page_title="Stock Vision", page_icon="💹", layout="wide")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root {
    --accent: #ff5f6d;
    --accent-light: rgba(255,95,109,0.18);
    --surface: #131722;
    --surface-soft: #1c2233;
    --stroke: rgba(255,255,255,0.08);
    --text: #f3f6ff;
    --muted: #a1a7c4;
}
body, .stApp {
    background: radial-gradient(circle at top, #1f283d, #080a0f 58%);
    font-family: 'Space Grotesk', sans-serif;
    color: var(--text);
}
.block-container {
    padding: 2.25rem 3rem 3rem 3rem;
}
.main > div:first-child {
    padding-top: 0.5rem;
}
.hero-title {
    text-align: center;
    font-size: clamp(2.2rem, 4vw, 3.2rem);
    font-weight: 700;
    margin: 0.75rem auto 0.35rem auto;
}
.hero-subtitle {
    text-align: center;
    color: var(--muted);
    margin-bottom: 2rem;
}
.glass-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
    border: 1px solid var(--stroke);
    border-radius: 22px;
    padding: 1.25rem 1.5rem;
    box-shadow: 0 20px 40px rgba(0,0,0,0.35);
}
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
}
.stTabs [data-baseweb="tab"] {
    background-color: var(--surface);
    border-radius: 999px;
    padding: 0.5rem 1.35rem;
    color: var(--muted);
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(120deg, #ff5f6d, #ffc371);
    color: #0b0d14;
}
.metric-card {
    background: var(--surface-soft);
    border: 1px solid var(--stroke);
    border-radius: 18px;
    padding: 1rem 1.25rem;
    text-align: center;
}
.metric-label {
    font-size: 0.85rem;
    color: var(--muted);
    margin-bottom: 0.25rem;
}
.metric-value {
    font-size: 1.65rem;
    font-weight: 600;
}
.metric-delta {
    font-size: 0.9rem;
}
.stDataFrame, .stTable {
    border-radius: 18px;
    overflow: hidden;
}
.sidebar .sidebar-content {
    background: var(--surface);
}
.stSelectbox, .stTextInput, .stDateInput {
    background-color: var(--surface-soft);
    border-radius: 12px;
}
.stButton > button {
    background: linear-gradient(120deg, #ff5f6d, #ffc371);
    border: none;
    color: #0b0d14;
    font-weight: 600;
    border-radius: 999px;
    padding: 0.5rem 1.5rem;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown(
    """
    <div class="hero-title">Stock Vision <span style="font-size:2.2rem;">💹</span></div>
    <p class="hero-subtitle">Real-time insights, news, and AI forecasts for your favorite tickers.</p>
    """,
    unsafe_allow_html=True
)


def metric_card(label: str, value: str, delta: Optional[float] = None, suffix: str = ""):
    delta_markup = ""
    if delta is not None:
        color = "#37d67a" if delta >= 0 else "#ff6b81"
        icon = "▲" if delta >= 0 else "▼"
        delta_markup = f'<div class="metric-delta" style="color:{color};">{icon} {abs(delta):.2f}{suffix}</div>'
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {delta_markup}
        </div>
        """,
        unsafe_allow_html=True
    )

# ------------------------------------------
# Helper Functions
# ------------------------------------------
@st.cache_data
def prepare_data_for_prediction(ticker, _scaler):
    data = download_price_data(ticker, period="5y")
    close_data = data[['Close']].values
    scaled_data = _scaler.transform(close_data)
    last_60 = scaled_data[-SEQ_LEN:]
    X_input = np.array([last_60])
    return data, X_input

def get_stock_data(ticker, start_date, end_date):
    try:
        stock_data = download_price_data(ticker, start=start_date, end=end_date)
        return stock_data, None
    except Exception as e:
        return None, f"Error fetching data: {e}"

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


def _render_overlay(html):
    components.html(html, height=0, width=0)


@st.cache_resource(show_spinner=False)
def get_loader_animation():
    try:
        loader_path = os.path.join(os.getcwd(), LOADER_JSON_FILE)
        with open(loader_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def show_fullscreen_loader(message="Working..."):
    animation = get_loader_animation()
    if not animation:
        return False

    encoded = base64.b64encode(json.dumps(animation).encode("utf-8")).decode("utf-8")
    overlay_script = f"""
    <script>
    const doc = window.parent.document;
    if (!doc.getElementById('lottie-player-script')) {{
        const script = doc.createElement('script');
        script.id = 'lottie-player-script';
        script.src = 'https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js';
        doc.head.appendChild(script);
    }}
    if (!doc.getElementById('growth-loader-overlay')) {{
        const overlay = doc.createElement('div');
        overlay.id = 'growth-loader-overlay';
        overlay.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(9,9,15,0.65);backdrop-filter:blur(8px);z-index:99999;display:flex;align-items:center;justify-content:center;';
        overlay.innerHTML = `
            <div style="text-align:center;color:#fff;">
                <div style="width:260px;height:260px;margin:0 auto;">
                    <lottie-player src="data:application/json;base64,{encoded}" background="transparent" speed="1" loop autoplay></lottie-player>
                </div>
                <p style="font-size:1.1rem;margin-top:0.75rem;">{message}</p>
            </div>`;
        doc.body.appendChild(overlay);
    }}
    </script>
    """
    _render_overlay(overlay_script)
    return True


def hide_fullscreen_loader():
    cleanup_script = """
    <script>
    const doc = window.parent.document;
    const overlay = doc.getElementById('growth-loader-overlay');
    if (overlay) {{
        overlay.remove();
    }}
    </script>
    """
    _render_overlay(cleanup_script)


@contextmanager
def loader_context(message: str):
    shown = show_fullscreen_loader(message)
    try:
        yield
    finally:
        if shown:
            hide_fullscreen_loader()


def normalize_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(-1)
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        if col in data.columns:
            series = data[col]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            data[col] = pd.to_numeric(series, errors="coerce")
    data.index = pd.to_datetime(data.index)
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)
    data.sort_index(inplace=True)
    data["Date"] = data.index
    return data


def to_float(value) -> float:
    if isinstance(value, pd.Series):
        return float(value.iloc[0])
    return float(value)


def get_alphavantage_token():
    """Return an Alpha Vantage key from Streamlit secrets or environment."""
    token = ""
    try:
        token = st.secrets["ALPHA_VANTAGE_API_KEY"]
    except Exception:
        token = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    return token or DEFAULT_ALPHA_VANTAGE_API_KEY


def download_price_data(ticker, max_attempts=3, retry_delay=2, **kwargs):
    last_error = None
    kwargs.setdefault("progress", False)
    kwargs.setdefault("auto_adjust", False)
    for attempt in range(1, max_attempts + 1):
        try:
            data = yf.download(ticker, **kwargs)
            if not data.empty:
                return data
            last_error = ValueError("Received empty dataset")
        except Exception as exc:
            last_error = exc
        time.sleep(retry_delay * attempt)
    raise RuntimeError(f"Failed to download data for {ticker}: {last_error}")


def get_model_paths(ticker):
    ticker = ticker.upper()
    model_path = os.path.join(MODEL_DIR, f"{ticker}_model.keras")
    scaler_path = os.path.join(MODEL_DIR, f"{ticker}_scaler.pkl")
    return model_path, scaler_path


def create_training_sequences(scaled_values):
    X, y = [], []
    for i in range(SEQ_LEN, len(scaled_values)):
        X.append(scaled_values[i-SEQ_LEN:i, 0])
        y.append(scaled_values[i, 0])
    X = np.array(X)
    y = np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y


def train_model_for_ticker(ticker, model_path, scaler_path):
    with st.spinner(f"Training model for {ticker}..."):
        data = download_price_data(ticker, period="5y")

        close_values = data[['Close']].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(close_values)

        if len(scaled) <= SEQ_LEN:
            raise ValueError(f"Not enough data to train model for {ticker}.")

        X, y = create_training_sequences(scaled)

        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, 1)),
            LSTM(32),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=TRAIN_EPOCHS, batch_size=TRAIN_BATCH_SIZE, verbose=0)

        os.makedirs(MODEL_DIR, exist_ok=True)
        model.save(model_path)
        joblib.dump(scaler, scaler_path)

        return model, scaler


@st.cache_resource(show_spinner=False)
def load_or_create_model(ticker):
    model_path, scaler_path = get_model_paths(ticker)

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler

    return train_model_for_ticker(ticker, model_path, scaler_path)

# ------------------------------------------
# Sidebar
# ------------------------------------------
st.sidebar.header("Stock Selection")
stocks = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA"]
selected_stock = st.sidebar.selectbox("Select a Stock", stocks)
custom_ticker = st.sidebar.text_input("Or Enter a Custom Ticker").upper()

ticker = custom_ticker if custom_ticker else selected_stock

# ------------------------------------------
# Navigation Menu
# ------------------------------------------
selected_option = option_menu(
    None,
    ["Home", "Visual Analysis", "News", "Prediction"],
    icons=['house', 'graph-up', 'newspaper', 'lightbulb'],
    orientation="horizontal",
    default_index=0
)

# =======================================================================================
# HOME
# =======================================================================================
if selected_option == "Home":
    st.subheader(f"Displaying Daily Price Data for {ticker}")

    top_col, chart_col = st.columns([1, 2])
    with top_col:
        lottie_url = "https://lottie.host/19ad9b6a-1882-4957-8216-bafa10a2ceaf/vA8lTy3DBm.json"
        lottie_anim = load_lottieurl(lottie_url)
        if lottie_anim:
            st_lottie(lottie_anim, height=230)

    with chart_col:
        date_cols = st.columns(2)
        with date_cols[0]:
            start_date = st.date_input("Start Date", date.today() - timedelta(days=365))
        with date_cols[1]:
            end_date = st.date_input("End Date", date.today())

        if start_date > end_date:
            st.error("End date cannot be before start date.")
        else:
            with loader_context(f"Loading {ticker} data..."):
                data, err = get_stock_data(ticker, start_date, end_date)

            if err:
                st.error(err)
            else:
                data = normalize_price_frame(data)
                latest = data.iloc[-1]
                prev = data.iloc[-2] if len(data) > 1 else latest
                latest_close = to_float(latest["Close"])
                prev_close = to_float(prev["Close"])
                latest_volume = to_float(latest["Volume"])
                delta_price = latest_close - prev_close
                delta_pct = (delta_price / prev_close * 100) if len(data) > 1 else 0

                metric_cols = st.columns(3)
                with metric_cols[0]:
                    metric_card("Latest Close", f"${latest_close:.2f}", delta_price, "")
                with metric_cols[1]:
                    metric_card("Volume", f"{latest_volume:,.0f}")
                with metric_cols[2]:
                    metric_card("Daily Change %", f"{delta_pct:+.2f}%", delta_pct, "%")

                csv_data = data.reset_index(drop=True).to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download data as CSV",
                    data=csv_data,
                    file_name=f"{ticker}_history.csv",
                    mime="text/csv"
                )

                st.dataframe(data, height=420, use_container_width=True)

# =======================================================================================
# VISUAL ANALYSIS
# =======================================================================================
if selected_option == "Visual Analysis":
    st.subheader(f"📊 Visual Analysis for {ticker}")

    today = date.today()
    default_start = today - timedelta(days=365)

    col_start, col_end = st.columns(2)
    start_date = col_start.date_input("Start Date (visuals)", default_start, key="viz_start")
    end_date = col_end.date_input("End Date (visuals)", today, key="viz_end")

    if start_date > end_date:
        st.error("End date cannot be before start date.")
    else:
        with loader_context(f"Preparing visuals for {ticker}..."):
            data, err = get_stock_data(ticker, start_date, end_date)

    if err:
        st.error(err)
    else:
        data = normalize_price_frame(data)
        data["MA20"] = data["Close"].rolling(window=20).mean()
        data["MA50"] = data["Close"].rolling(window=50).mean()
        data["Daily Return %"] = data["Close"].pct_change().fillna(0) * 100
        data["Daily Change"] = data["Close"] - data["Open"]
        data["Direction"] = np.where(data["Daily Change"] >= 0, "Up Day", "Down Day")
        rolling_std = data["Close"].rolling(window=20).std()
        data["Upper BB"] = data["MA20"] + 2 * rolling_std
        data["Lower BB"] = data["MA20"] - 2 * rolling_std
        data["EMA12"] = data["Close"].ewm(span=12, adjust=False).mean()
        data["EMA26"] = data["Close"].ewm(span=26, adjust=False).mean()
        data["MACD"] = data["EMA12"] - data["EMA26"]
        data["Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()

        tabs = st.tabs([
            "Line & Moving Averages",
            "Candlestick",
            "Volume Histogram",
            "Daily Return Distribution",
            "Price vs Volume Scatter",
            "Up vs Down Days",
            "Bollinger Bands",
            "MACD"
        ])

        with tabs[0]:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], mode='lines', name='Close Price'))
            fig.add_trace(go.Scatter(x=data["Date"], y=data["MA20"], mode='lines', name='MA 20', line=dict(dash='dash')))
            fig.add_trace(go.Scatter(x=data["Date"], y=data["MA50"], mode='lines', name='MA 50', line=dict(dash='dot')))
            fig.update_layout(title=f"{ticker} Closing Price & Moving Averages", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        with tabs[1]:
            candlestick_data = data.dropna(subset=["Open", "High", "Low", "Close"])
            fig = go.Figure(data=[go.Candlestick(
                x=candlestick_data["Date"],
                open=candlestick_data['Open'],
                high=candlestick_data['High'],
                low=candlestick_data['Low'],
                close=candlestick_data['Close'],
                increasing_line_color='green',
                decreasing_line_color='red'
            )])
            fig.update_layout(title=f"{ticker} Candlestick Chart", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        with tabs[2]:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=data["Date"], y=data["Volume"], marker_color='#636EFA', name='Volume'))
            fig.update_layout(title=f"{ticker} Daily Volume", template="plotly_white", xaxis_title="Date", yaxis_title="Volume")
            st.plotly_chart(fig, use_container_width=True)

        with tabs[3]:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=data["Daily Return %"], nbinsx=40, marker_color='#EF553B'))
            fig.update_layout(
                title=f"{ticker} Daily Return Distribution (%)",
                template="plotly_white",
                xaxis_title="Daily Return (%)",
                yaxis_title="Frequency"
            )
            st.plotly_chart(fig, use_container_width=True)

        with tabs[4]:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data["Close"],
                y=data["Volume"],
                mode="markers",
                marker=dict(size=8, color=data["Daily Return %"], colorscale="Viridis", showscale=True, colorbar_title="Daily Return (%)")
            ))
            fig.update_layout(
                title=f"{ticker} Price vs Volume",
                template="plotly_white",
                xaxis_title="Close Price",
                yaxis_title="Volume"
            )
            st.plotly_chart(fig, use_container_width=True)

        with tabs[5]:
            direction_counts = data["Direction"].value_counts()
            fig = go.Figure(data=[go.Pie(
                labels=direction_counts.index,
                values=direction_counts.values,
                hole=0.4
            )])
            fig.update_layout(title=f"{ticker} Up vs Down Days", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        with tabs[6]:
            bb_data = data.dropna(subset=["Upper BB", "Lower BB"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=bb_data["Date"], y=bb_data["Close"], mode='lines', name='Close Price'))
            fig.add_trace(go.Scatter(x=bb_data["Date"], y=bb_data["Upper BB"], mode='lines', name='Upper Band', line=dict(dash='dash')))
            fig.add_trace(go.Scatter(x=bb_data["Date"], y=bb_data["Lower BB"], mode='lines', name='Lower Band', line=dict(dash='dash')))
            fig.update_layout(title=f"{ticker} Bollinger Bands", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        with tabs[7]:
            macd_data = data.dropna(subset=["MACD", "Signal"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=macd_data["Date"], y=macd_data["MACD"], mode='lines', name='MACD'))
            fig.add_trace(go.Scatter(x=macd_data["Date"], y=macd_data["Signal"], mode='lines', name='Signal'))
            fig.add_trace(go.Bar(x=macd_data["Date"], y=macd_data["MACD"] - macd_data["Signal"], name='Histogram', marker_color='#636EFA', opacity=0.3))
            fig.update_layout(title=f"{ticker} MACD Indicator", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

# =======================================================================================
# NEWS (UPDATED – WORKING FREE API)
# =======================================================================================
if selected_option == "News":
    st.subheader(f"📰 Latest News for {ticker}")

    token_input = get_alphavantage_token().strip()

    if not token_input:
        st.error("Alpha Vantage API key missing. Please set ALPHA_VANTAGE_API_KEY in secrets or environment.")
        st.stop()

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "sort": "LATEST",
        "limit": 10,
        "apikey": token_input
    }

    try:
        with loader_context(f"Fetching latest news for {ticker}..."):
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

        news_data = response.json()

        if "Note" in news_data:
            st.warning(news_data["Note"])
        elif "Information" in news_data:
            st.warning(news_data["Information"])
        else:
            articles = news_data.get("feed", [])

            if not articles:
                st.warning("No news found for this stock.")
            else:
                for item in articles[:10]:
                    title = item.get("title", "No Title")
                    description = item.get("summary", "")
                    url_news = item.get("url")
                    source = item.get("source")
                    published = item.get("time_published")

                    st.markdown(f"### [{title}]({url_news})")
                    st.caption(f"📰 {source} | 📅 {published}")
                    st.write(description)
                    st.write("---")

    except HTTPError as e:
        st.error(f"Failed to fetch news: {e}")
    except RequestException as e:
        st.error(f"Failed to fetch news: {e}")

# =======================================================================================
# PREDICTION
# =======================================================================================
if selected_option == "Prediction":
    st.subheader(f"📈 Prediction for {selected_stock}")

    try:
        with loader_context(f"Preparing predictions for {selected_stock}..."):
            model, scaler = load_or_create_model(selected_stock)
            data, X_input = prepare_data_for_prediction(selected_stock, scaler)

        def forecast_days(X_input, days):
            preds, current_input = [], X_input.copy()
            for _ in range(days):
                pred_scaled = model.predict(current_input, verbose=0)
                pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1))
                preds.append(pred[0][0])
                new_scaled = scaler.transform(pred)
                current_input = np.append(current_input[:, 1:, :], [[new_scaled[0]]], axis=1)
            return preds

        horizons = {"Today & Tomorrow": 2, "Next 7 Days": 7, "Next 30 Days": 30}

        for label, days in horizons.items():
            preds = forecast_days(X_input, days)
            future_dates = pd.date_range(start=date.today(), periods=days)
            df_forecast = pd.DataFrame({"Date": future_dates, "Predicted Close Price": preds})

            st.markdown(f"### 📅 Predicted Stock Prices – {label}")
            st.dataframe(df_forecast)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_forecast["Date"], y=df_forecast["Predicted Close Price"],
                                     mode="lines+markers", name="Prediction"))
            fig.update_layout(title=f"{selected_stock} – {label}", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"⚠️ Prediction error: {e}")
        

