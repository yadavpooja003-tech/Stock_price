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
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import Callback
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
TRAIN_EPOCHS = 50          # upgraded from 5 → 50 (matches your notebook)
TRAIN_BATCH_SIZE = 32
LOADER_JSON_FILE = "Growth Chart.json"

# Number of features fed into LSTM:
# [Close, Volume, MA20, RSI, SentimentScore] = 5
N_FEATURES = 5

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
.block-container { padding: 2.25rem 3rem 3rem 3rem; }
.main > div:first-child { padding-top: 0.5rem; }
.hero-title {
    text-align: center;
    font-size: clamp(2.2rem, 4vw, 3.2rem);
    font-weight: 700;
    margin: 0.75rem auto 0.35rem auto;
}
.hero-subtitle { text-align: center; color: var(--muted); margin-bottom: 2rem; }
.glass-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
    border: 1px solid var(--stroke);
    border-radius: 22px;
    padding: 1.25rem 1.5rem;
    box-shadow: 0 20px 40px rgba(0,0,0,0.35);
}
.stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
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
.metric-label { font-size: 0.85rem; color: var(--muted); margin-bottom: 0.25rem; }
.metric-value { font-size: 1.65rem; font-weight: 600; }
.metric-delta { font-size: 0.9rem; }
.stDataFrame, .stTable { border-radius: 18px; overflow: hidden; }
.sidebar .sidebar-content { background: var(--surface); }
.stSelectbox, .stTextInput, .stDateInput { background-color: var(--surface-soft); border-radius: 12px; }
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
    <p class="hero-subtitle">Real-time insights, news, AI forecasts — now powered by multivariate LSTM.</p>
    """,
    unsafe_allow_html=True
)


# =======================================================================================
# KERAS CALLBACK — streams live epoch updates into Streamlit
# =======================================================================================
class StreamlitProgressCallback(Callback):
    """
    Custom Keras callback that updates a Streamlit progress bar
    and status text after every epoch so the user can watch
    the model train in real time.
    """
    def __init__(self, total_epochs, progress_bar, status_text):
        super().__init__()
        self.total_epochs = total_epochs
        self.progress_bar = progress_bar
        self.status_text  = status_text

    def on_epoch_end(self, epoch, logs=None):
        loss     = logs.get("loss", 0)
        val_loss = logs.get("val_loss", 0)
        progress = (epoch + 1) / self.total_epochs
        self.progress_bar.progress(progress)
        self.status_text.markdown(
            f"**Epoch {epoch + 1} / {self.total_epochs}** &nbsp;|&nbsp; "
            f"Train Loss: `{loss:.6f}` &nbsp;|&nbsp; Val Loss: `{val_loss:.6f}`"
        )


# =======================================================================================
# HELPER FUNCTIONS
# =======================================================================================
def metric_card(label: str, value: str, delta: Optional[float] = None, suffix: str = ""):
    delta_markup = ""
    if delta is not None:
        color = "#37d67a" if delta >= 0 else "#ff6b81"
        icon  = "▲" if delta >= 0 else "▼"
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
    if (overlay) { overlay.remove(); }
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
    # Replace dots with underscores so filenames are safe on all OS
    # e.g. RELIANCE.NS → RELIANCE_NS,  TCS.NS → TCS_NS
    safe_name   = ticker.upper().replace(".", "_").replace("/", "_")
    model_path  = os.path.join(MODEL_DIR, f"{safe_name}_mv_model.keras")
    scaler_path = os.path.join(MODEL_DIR, f"{safe_name}_mv_scaler.pkl")
    return model_path, scaler_path


# =======================================================================================
# RSI CALCULATION
# =======================================================================================
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Standard RSI using Wilder's smoothing (EMA of gains/losses)."""
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs  = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


# =======================================================================================
# NEWS SENTIMENT FETCH  (Alpha Vantage — already used in your News tab)
# =======================================================================================
def fetch_avg_sentiment(ticker: str, api_key: str) -> float:
    """
    Fetches the latest 50 news items from Alpha Vantage and returns
    the average overall_sentiment_score for that ticker.
    Returns 0.0 (neutral) on failure so training can still proceed.
    For non-US tickers (e.g. RELIANCE.NS), Alpha Vantage may have no coverage —
    a warning is stored in session_state and shown on the Prediction page.
    """
    try:
        url    = "https://www.alphavantage.co/query"
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers" : ticker,
            "sort"    : "LATEST",
            "limit"   : 50,
            "apikey"  : api_key,
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        scores = []
        for item in data.get("feed", []):
            for ts in item.get("ticker_sentiment", []):
                if ts.get("ticker", "").upper() == ticker.upper():
                    try:
                        scores.append(float(ts["ticker_sentiment_score"]))
                    except (ValueError, KeyError):
                        pass
        if scores:
            # Clear any old warning if we got data this time
            st.session_state.pop("sentiment_warning", None)
            return float(np.mean(scores))
        else:
            # No articles found — common for Indian / global tickers
            st.session_state["sentiment_warning"] = (
                f"ℹ️ News sentiment data is unavailable for **{ticker}** "
                f"(Alpha Vantage covers mainly US-listed stocks). "
                f"Sentiment feature set to **neutral (0.0)** — model still works normally."
            )
            return 0.0
    except Exception:
        st.session_state["sentiment_warning"] = (
            f"ℹ️ Could not fetch news sentiment for **{ticker}**. "
            f"Using neutral score (0.0) as fallback."
        )
        return 0.0


# =======================================================================================
# FEATURE ENGINEERING — builds the 5-column DataFrame fed to LSTM
# =======================================================================================
def build_feature_df(ticker: str, sentiment_score: float) -> pd.DataFrame:
    """
    Downloads 5 years of daily data for `ticker` and returns a DataFrame
    with exactly these columns (in order):
        Close | Volume | MA20 | RSI | Sentiment

    `sentiment_score` is a single scalar repeated for every row because
    yfinance does not give per-day sentiment history. When more data is
    available (e.g. from a paid API) you can replace this with a time series.
    """
    raw = download_price_data(ticker, period="5y")
    raw = normalize_price_frame(raw)

    df = pd.DataFrame()
    df["Close"]     = raw["Close"]
    df["Volume"]    = raw["Volume"]
    df["MA20"]      = raw["Close"].rolling(window=20).mean()
    df["RSI"]       = compute_rsi(raw["Close"], period=14)
    df["Sentiment"] = sentiment_score          # constant scalar from latest news

    df.dropna(inplace=True)                    # drop first 20 rows (MA20 warm-up)
    return df, raw                             # also return raw for reference


# =======================================================================================
# SEQUENCE BUILDER — works with N_FEATURES columns
# =======================================================================================
def create_multivariate_sequences(scaled_data: np.ndarray, seq_len: int):
    """
    Builds (X, y) pairs where:
        X shape = (samples, seq_len, N_FEATURES)
        y shape = (samples,)   → predicts next Close (column 0 after scaling)
    """
    X, y = [], []
    for i in range(seq_len, len(scaled_data)):
        X.append(scaled_data[i - seq_len : i, :])   # all features
        y.append(scaled_data[i, 0])                  # only Close price
    return np.array(X), np.array(y)


# =======================================================================================
# MODEL ARCHITECTURE — 3 LSTM layers + Dropout (matches your notebook)
# =======================================================================================
def build_lstm_model(seq_len: int, n_features: int) -> Sequential:
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True,
                   input_shape=(seq_len, n_features)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


# =======================================================================================
# TRAIN — called both on first run AND when user clicks Retrain button
# =======================================================================================
def train_multivariate_model(ticker: str, model_path: str, scaler_path: str,
                              api_key: str = "",
                              progress_bar=None, status_text=None):
    """
    Full training pipeline:
      1. Fetch sentiment score from Alpha Vantage
      2. Build 5-feature DataFrame
      3. Scale with MinMaxScaler (fit_transform on train, transform on test)
      4. Create sequences
      5. Train 3-layer LSTM with optional live Streamlit callback
      6. Save model + scaler
    """
    # -- Step 1: sentiment
    sentiment = fetch_avg_sentiment(ticker, api_key) if api_key else 0.0

    # -- Step 2: features
    feat_df, _ = build_feature_df(ticker, sentiment)
    feature_values = feat_df[["Close", "Volume", "MA20", "RSI", "Sentiment"]].values

    if len(feature_values) <= SEQ_LEN:
        raise ValueError(f"Not enough data to train model for {ticker}.")

    # -- Step 3: train / test split (80 / 20) + scaling
    split      = int(len(feature_values) * 0.8)
    train_data = feature_values[:split]
    test_data  = feature_values[split:]

    scaler      = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train_data)
    scaled_test  = scaler.transform(test_data)

    # -- Step 4: sequences
    X_train, y_train = create_multivariate_sequences(scaled_train, SEQ_LEN)
    X_test,  y_test  = create_multivariate_sequences(scaled_test,  SEQ_LEN)

    # -- Step 5: build & train
    model = build_lstm_model(SEQ_LEN, N_FEATURES)

    callbacks = []
    if progress_bar is not None and status_text is not None:
        callbacks.append(
            StreamlitProgressCallback(TRAIN_EPOCHS, progress_bar, status_text)
        )

    model.fit(
        X_train, y_train,
        epochs          = TRAIN_EPOCHS,
        batch_size      = TRAIN_BATCH_SIZE,
        validation_data = (X_test, y_test),
        verbose         = 0,
        callbacks       = callbacks,
    )

    # -- Step 6: save
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    return model, scaler, sentiment


# =======================================================================================
# LOAD OR CREATE MODEL (cached — skips training if files already exist)
# =======================================================================================
@st.cache_resource(show_spinner=False)
def load_or_create_model(ticker, api_key=""):
    model_path, scaler_path = get_model_paths(ticker)

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model  = load_model(model_path)
        scaler = joblib.load(scaler_path)
        # sentiment is not stored — fetch fresh for prediction
        sentiment = fetch_avg_sentiment(ticker, api_key) if api_key else 0.0
        return model, scaler, sentiment

    # First time — train silently (no progress bar)
    with st.spinner(f"Training multivariate LSTM for {ticker} for the first time…"):
        return train_multivariate_model(ticker, model_path, scaler_path, api_key)


# =======================================================================================
# PREDICTION — iteratively forecast N days ahead
# =======================================================================================
@st.cache_data
def prepare_data_for_prediction(ticker, sentiment_score):
    feat_df, raw = build_feature_df(ticker, sentiment_score)
    feature_values = feat_df[["Close", "Volume", "MA20", "RSI", "Sentiment"]].values
    return feat_df, feature_values, raw


def forecast_days(model, scaler, feature_values, days: int):
    """
    Rolls the last SEQ_LEN rows forward `days` times.
    Only the Close price (column 0) is varied; other features stay fixed
    at their last known values (reasonable assumption for short-term forecasts).
    """
    scaled_all    = scaler.transform(feature_values)
    current_input = scaled_all[-SEQ_LEN:].copy()   # shape (60, 5)
    predictions   = []

    for _ in range(days):
        X_input    = np.expand_dims(current_input, axis=0)   # (1, 60, 5)
        pred_scaled = model.predict(X_input, verbose=0)[0, 0]
        predictions.append(pred_scaled)

        # Build next row: replace Close with prediction, keep other features
        next_row        = current_input[-1].copy()
        next_row[0]     = pred_scaled                        # update Close only
        current_input   = np.vstack([current_input[1:], next_row])

    # Inverse-transform only the Close column
    dummy           = np.zeros((days, N_FEATURES))
    dummy[:, 0]     = predictions
    inv             = scaler.inverse_transform(dummy)
    return inv[:, 0]


# =======================================================================================
# SIDEBAR
# =======================================================================================
st.sidebar.markdown("### 🔍 Stock Selection")

# ── 50 popular stocks across US, India & Global markets ──────────────────────
stocks = [
    # 🇺🇸 US Tech
    "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META", "NFLX",
    "AMD", "INTC", "ORCL", "IBM", "ADBE", "CRM", "PYPL", "UBER",
    # 🇺🇸 US Finance / Other
    "JPM", "BAC", "GS", "V", "MA", "WMT", "KO", "PEP", "JNJ", "PFE",
    # 🇮🇳 India (NSE)
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "SBIN.NS", "WIPRO.NS", "HINDUNILVR.NS", "ITC.NS", "BAJFINANCE.NS",
    "ADANIENT.NS", "TATAMOTORS.NS", "MARUTI.NS", "SUNPHARMA.NS", "LT.NS",
    # 🌍 Global
    "BABA", "TSM", "SONY", "SAP", "ASML",
]

selected_stock = st.sidebar.selectbox(
    "Pick from popular stocks",
    options=stocks,
    index=0,
)
custom_ticker = st.sidebar.text_input(
    "Or type any ticker (e.g. HDFCBANK.NS, BABA, SHOP)"
).upper().strip()

ticker = custom_ticker if custom_ticker else selected_stock

# Show which ticker is currently active
st.sidebar.success(f"📌 Active: **{ticker}**")

# ---- MODEL CONTROLS ----
st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 Model Controls")
st.sidebar.caption(
    "The model uses **5 features**: Close price, Volume, MA20, RSI, and News Sentiment."
)

if st.sidebar.button(f"🔄 Retrain Model for {ticker}"):
    model_path, scaler_path = get_model_paths(ticker)

    # 1. Delete saved files so training starts from scratch
    for p in [model_path, scaler_path]:
        if os.path.exists(p):
            os.remove(p)

    # 2. Clear Streamlit's in-memory cache for this ticker
    load_or_create_model.clear()
    prepare_data_for_prediction.clear()

    st.sidebar.info(f"Old model cleared. Training fresh model for **{ticker}**…")

    # 3. Live progress widgets inside the sidebar
    progress_bar = st.sidebar.progress(0)
    status_text  = st.sidebar.empty()

    try:
        api_key = get_alphavantage_token()
        train_multivariate_model(
            ticker, model_path, scaler_path,
            api_key       = api_key,
            progress_bar  = progress_bar,
            status_text   = status_text,
        )
        st.sidebar.success(f"✅ Model retrained successfully for **{ticker}**!")
        st.sidebar.caption(
            "Switch to the **Prediction** tab to see updated forecasts."
        )
    except Exception as e:
        st.sidebar.error(f"❌ Retraining failed: {e}")


# =======================================================================================
# NAVIGATION MENU
# =======================================================================================
selected_option = option_menu(
    None,
    ["Home", "Visual Analysis", "News", "Prediction", "Portfolio"],
    icons=["house", "graph-up", "newspaper", "lightbulb", "briefcase"],
    orientation="horizontal",
    default_index=0,
)


# =======================================================================================
# HOME
# =======================================================================================
if selected_option == "Home":
    st.subheader(f"Displaying Daily Price Data for {ticker}")

    top_col, chart_col = st.columns([1, 2])
    with top_col:
        lottie_url  = "https://lottie.host/19ad9b6a-1882-4957-8216-bafa10a2ceaf/vA8lTy3DBm.json"
        lottie_anim = load_lottieurl(lottie_url)
        if lottie_anim:
            st_lottie(lottie_anim, height=230)

    with chart_col:
        date_cols  = st.columns(2)
        start_date = date_cols[0].date_input("Start Date", date.today() - timedelta(days=365))
        end_date   = date_cols[1].date_input("End Date",   date.today())

        if start_date > end_date:
            st.error("End date cannot be before start date.")
        else:
            with loader_context(f"Loading {ticker} data…"):
                data, err = get_stock_data(ticker, start_date, end_date)

            if err:
                st.error(err)
            else:
                data         = normalize_price_frame(data)
                latest       = data.iloc[-1]
                prev         = data.iloc[-2] if len(data) > 1 else latest
                latest_close = to_float(latest["Close"])
                prev_close   = to_float(prev["Close"])
                latest_vol   = to_float(latest["Volume"])
                delta_price  = latest_close - prev_close
                delta_pct    = (delta_price / prev_close * 100) if len(data) > 1 else 0

                metric_cols = st.columns(3)
                with metric_cols[0]: metric_card("Latest Close",    f"${latest_close:.2f}", delta_price)
                with metric_cols[1]: metric_card("Volume",          f"{latest_vol:,.0f}")
                with metric_cols[2]: metric_card("Daily Change %",  f"{delta_pct:+.2f}%",   delta_pct, "%")

                csv_data = data.reset_index(drop=True).to_csv(index=False).encode("utf-8")
                st.download_button("Download data as CSV", data=csv_data,
                                   file_name=f"{ticker}_history.csv", mime="text/csv")
                st.dataframe(data, height=420, use_container_width=True)


# =======================================================================================
# VISUAL ANALYSIS
# =======================================================================================
if selected_option == "Visual Analysis":
    st.subheader(f"📊 Visual Analysis for {ticker}")

    today         = date.today()
    default_start = today - timedelta(days=365)
    col_start, col_end = st.columns(2)
    start_date = col_start.date_input("Start Date (visuals)", default_start, key="viz_start")
    end_date   = col_end.date_input("End Date (visuals)",   today,          key="viz_end")

    err = None
    if start_date > end_date:
        st.error("End date cannot be before start date.")
    else:
        with loader_context(f"Preparing visuals for {ticker}…"):
            data, err = get_stock_data(ticker, start_date, end_date)

    if err:
        st.error(err)
    else:
        data = normalize_price_frame(data)
        data["MA20"]          = data["Close"].rolling(window=20).mean()
        data["MA50"]          = data["Close"].rolling(window=50).mean()
        data["Daily Return %"] = data["Close"].pct_change().fillna(0) * 100
        data["Daily Change"]   = data["Close"] - data["Open"]
        data["Direction"]      = np.where(data["Daily Change"] >= 0, "Up Day", "Down Day")
        rolling_std            = data["Close"].rolling(window=20).std()
        data["Upper BB"]       = data["MA20"] + 2 * rolling_std
        data["Lower BB"]       = data["MA20"] - 2 * rolling_std
        data["EMA12"]          = data["Close"].ewm(span=12, adjust=False).mean()
        data["EMA26"]          = data["Close"].ewm(span=26, adjust=False).mean()
        data["MACD"]           = data["EMA12"] - data["EMA26"]
        data["Signal"]         = data["MACD"].ewm(span=9, adjust=False).mean()

        # NEW: RSI tab
        data["RSI"] = compute_rsi(data["Close"])

        tabs = st.tabs([
            "Line & Moving Averages",
            "Candlestick",
            "Volume Histogram",
            "Daily Return Distribution",
            "Price vs Volume Scatter",
            "Up vs Down Days",
            "Bollinger Bands",
            "MACD",
            "RSI",                    # ← NEW tab
        ])

        with tabs[0]:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], mode="lines", name="Close Price"))
            fig.add_trace(go.Scatter(x=data["Date"], y=data["MA20"],  mode="lines", name="MA 20", line=dict(dash="dash")))
            fig.add_trace(go.Scatter(x=data["Date"], y=data["MA50"],  mode="lines", name="MA 50", line=dict(dash="dot")))
            fig.update_layout(title=f"{ticker} Closing Price & Moving Averages", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        with tabs[1]:
            cd = data.dropna(subset=["Open", "High", "Low", "Close"])
            fig = go.Figure(data=[go.Candlestick(
                x=cd["Date"], open=cd["Open"], high=cd["High"],
                low=cd["Low"], close=cd["Close"],
                increasing_line_color="green", decreasing_line_color="red"
            )])
            fig.update_layout(title=f"{ticker} Candlestick Chart", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        with tabs[2]:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=data["Date"], y=data["Volume"], marker_color="#636EFA", name="Volume"))
            fig.update_layout(title=f"{ticker} Daily Volume", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        with tabs[3]:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=data["Daily Return %"], nbinsx=40, marker_color="#EF553B"))
            fig.update_layout(title=f"{ticker} Daily Return Distribution (%)", template="plotly_white",
                              xaxis_title="Daily Return (%)", yaxis_title="Frequency")
            st.plotly_chart(fig, use_container_width=True)

        with tabs[4]:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data["Close"], y=data["Volume"], mode="markers",
                marker=dict(size=8, color=data["Daily Return %"],
                            colorscale="Viridis", showscale=True, colorbar_title="Daily Return (%)")
            ))
            fig.update_layout(title=f"{ticker} Price vs Volume", template="plotly_white",
                              xaxis_title="Close Price", yaxis_title="Volume")
            st.plotly_chart(fig, use_container_width=True)

        with tabs[5]:
            dc = data["Direction"].value_counts()
            fig = go.Figure(data=[go.Pie(labels=dc.index, values=dc.values, hole=0.4)])
            fig.update_layout(title=f"{ticker} Up vs Down Days", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        with tabs[6]:
            bb = data.dropna(subset=["Upper BB", "Lower BB"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=bb["Date"], y=bb["Close"],    mode="lines", name="Close Price"))
            fig.add_trace(go.Scatter(x=bb["Date"], y=bb["Upper BB"], mode="lines", name="Upper Band", line=dict(dash="dash")))
            fig.add_trace(go.Scatter(x=bb["Date"], y=bb["Lower BB"], mode="lines", name="Lower Band", line=dict(dash="dash")))
            fig.update_layout(title=f"{ticker} Bollinger Bands", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        with tabs[7]:
            md = data.dropna(subset=["MACD", "Signal"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=md["Date"], y=md["MACD"],   mode="lines", name="MACD"))
            fig.add_trace(go.Scatter(x=md["Date"], y=md["Signal"], mode="lines", name="Signal"))
            fig.add_trace(go.Bar(x=md["Date"], y=md["MACD"] - md["Signal"],
                                 name="Histogram", marker_color="#636EFA", opacity=0.3))
            fig.update_layout(title=f"{ticker} MACD Indicator", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        with tabs[8]:   # ← NEW RSI tab
            rsi_data = data.dropna(subset=["RSI"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=rsi_data["Date"], y=rsi_data["RSI"],
                                     mode="lines", name="RSI", line=dict(color="#a29bfe")))
            # Overbought / oversold reference lines
            fig.add_hline(y=70, line_dash="dash", line_color="red",
                          annotation_text="Overbought (70)", annotation_position="bottom right")
            fig.add_hline(y=30, line_dash="dash", line_color="green",
                          annotation_text="Oversold (30)",   annotation_position="top right")
            fig.update_layout(
                title=f"{ticker} Relative Strength Index (RSI)",
                template="plotly_white",
                yaxis=dict(range=[0, 100]),
                xaxis_title="Date",
                yaxis_title="RSI",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("RSI > 70 → potentially overbought (consider selling). RSI < 30 → potentially oversold (consider buying).")


# =======================================================================================
# NEWS
# =======================================================================================
if selected_option == "News":
    st.subheader(f"📰 Latest News for {ticker}")

    token_input = get_alphavantage_token().strip()
    if not token_input:
        st.error("Alpha Vantage API key missing. Please set ALPHA_VANTAGE_API_KEY in secrets or environment.")
        st.stop()

    url    = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers" : ticker,
        "sort"    : "LATEST",
        "limit"   : 10,
        "apikey"  : token_input,
    }

    try:
        with loader_context(f"Fetching latest news for {ticker}…"):
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
                # ── NEW: aggregate sentiment chart ──────────────────────────────
                sentiment_rows = []
                for item in articles[:10]:
                    for ts in item.get("ticker_sentiment", []):
                        if ts.get("ticker", "").upper() == ticker.upper():
                            try:
                                sentiment_rows.append({
                                    "title": item.get("title", "")[:40] + "…",
                                    "score": float(ts["ticker_sentiment_score"]),
                                    "label": ts.get("ticker_sentiment_label", "Neutral"),
                                })
                            except (ValueError, KeyError):
                                pass

                if sentiment_rows:
                    st.markdown("### 📊 News Sentiment Overview")
                    sent_df  = pd.DataFrame(sentiment_rows)
                    colors   = ["#37d67a" if s > 0.15 else "#ff6b81" if s < -0.15 else "#a1a7c4"
                                for s in sent_df["score"]]
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=sent_df["score"],
                        y=sent_df["title"],
                        orientation="h",
                        marker_color=colors,
                        text=[f"{s:+.3f}" for s in sent_df["score"]],
                        textposition="outside",
                    ))
                    fig.add_vline(x=0, line_width=1, line_color="white")
                    fig.update_layout(
                        title="Per-Article Sentiment Score (green = bullish, red = bearish)",
                        template="plotly_dark",
                        xaxis_title="Sentiment Score",
                        height=380,
                        margin=dict(l=10, r=10),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    avg = float(np.mean([r["score"] for r in sentiment_rows]))
                    label = "🟢 Bullish" if avg > 0.15 else "🔴 Bearish" if avg < -0.15 else "⚪ Neutral"
                    st.info(f"**Overall Sentiment for {ticker}:** {label} (avg score: {avg:+.4f})")
                # ── end sentiment chart ─────────────────────────────────────────

                st.markdown("### 📰 Articles")
                for item in articles[:10]:
                    title     = item.get("title", "No Title")
                    summary   = item.get("summary", "")
                    url_news  = item.get("url")
                    source    = item.get("source")
                    published = item.get("time_published")
                    st.markdown(f"### [{title}]({url_news})")
                    st.caption(f"📰 {source} | 📅 {published}")
                    st.write(summary)
                    st.write("---")

    except HTTPError as e:
        st.error(f"Failed to fetch news: {e}")
    except RequestException as e:
        st.error(f"Failed to fetch news: {e}")


# =======================================================================================
# PREDICTION  (multivariate LSTM)
# =======================================================================================
if selected_option == "Prediction":
    st.subheader(f"📈 AI Prediction for {selected_stock}")

    st.info(
        "⚠️ **Model Info:** This LSTM is trained on **5 features** — Close price, Volume, "
        "20-day Moving Average, RSI, and live News Sentiment from Alpha Vantage. "
        "Predictions are pattern-based and for **educational purposes only**."
    )

    try:
        api_key = get_alphavantage_token()

        with loader_context(f"Loading model for {selected_stock}…"):
            model, scaler, sentiment = load_or_create_model(selected_stock, api_key)

        # Show sentiment warning if Alpha Vantage had no data for this ticker
        if "sentiment_warning" in st.session_state:
            st.warning(st.session_state["sentiment_warning"])
            del st.session_state["sentiment_warning"]

        with loader_context(f"Preparing features for {selected_stock}…"):
            feat_df, feature_values, raw = prepare_data_for_prediction(selected_stock, sentiment)

        # Show what sentiment score the model is using
        sent_label = "🟢 Bullish" if sentiment > 0.15 else "🔴 Bearish" if sentiment < -0.15 else "⚪ Neutral"
        st.caption(f"Current news sentiment used as model input: **{sent_label}** ({sentiment:+.4f})")

        horizons = {
            "Today & Tomorrow": 2,
            "Next 7 Days"     : 7,
            "Next 30 Days"    : 30,
        }

        for label, days in horizons.items():
            preds        = forecast_days(model, scaler, feature_values, days)
            future_dates = pd.date_range(start=date.today(), periods=days)
            df_forecast  = pd.DataFrame({"Date": future_dates, "Predicted Close Price": preds})

            st.markdown(f"### 📅 {label}")
            st.dataframe(df_forecast)

            fig = go.Figure()
            # Plot last 60 actual days for context
            actual_tail = raw.iloc[-60:]
            actual_tail = normalize_price_frame(actual_tail) if "Date" not in actual_tail.columns else actual_tail
            fig.add_trace(go.Scatter(
                x=feat_df.index[-60:],
                y=feat_df["Close"].values[-60:],
                mode="lines", name="Actual (last 60 days)",
                line=dict(color="#74b9ff")
            ))
            fig.add_trace(go.Scatter(
                x=df_forecast["Date"],
                y=df_forecast["Predicted Close Price"],
                mode="lines+markers", name="Predicted",
                line=dict(color="#ff5f6d", dash="dash")
            ))
            fig.update_layout(
                title=f"{selected_stock} — {label}",
                template="plotly_white",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
            )
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"⚠️ Prediction error: {e}")
        st.caption("Try clicking **Retrain Model** in the sidebar to rebuild the model from scratch.")


# =======================================================================================
# PORTFOLIO TRACKER
# =======================================================================================
if selected_option == "Portfolio":
    st.subheader("💼 Portfolio Tracker")
    st.caption("Add your holdings below. Current prices are fetched live from Yahoo Finance.")

    # ── Session state initialisation ────────────────────────────────────────────────────
    if "portfolio" not in st.session_state:
        st.session_state["portfolio"] = []   # list of dicts: {ticker, qty, buy_price}

    # ── Helper: fetch current price for a ticker ─────────────────────────────────────
    @st.cache_data(ttl=60)   # cache 60 sec so repeated renders don't spam yfinance
    def get_current_price(ticker: str) -> float:
        try:
            info = yf.Ticker(ticker).fast_info
            price = getattr(info, "last_price", None) or getattr(info, "regular_market_price", None)
            if price:
                return float(price)
            # fallback: download last 2 days and take last close
            hist = yf.download(ticker, period="2d", progress=False, auto_adjust=True)
            if not hist.empty:
                return float(hist["Close"].iloc[-1])
        except Exception:
            pass
        return 0.0

    # ── ADD HOLDING FORM ─────────────────────────────────────────────────────────────
    st.markdown("### ➕ Add a Holding")
    with st.container():
        col1, col2, col3, col4 = st.columns([2, 1.2, 1.5, 1])
        with col1:
            new_ticker = st.text_input(
                "Ticker symbol",
                placeholder="e.g. AAPL, RELIANCE.NS",
                key="pt_ticker"
            ).upper().strip()
        with col2:
            new_qty = st.number_input(
                "Quantity (shares)",
                min_value=0.01,
                value=1.0,
                step=1.0,
                key="pt_qty"
            )
        with col3:
            new_buy = st.number_input(
                "Buy price (per share)",
                min_value=0.01,
                value=100.0,
                step=0.01,
                key="pt_buy"
            )
        with col4:
            st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
            add_clicked = st.button("Add ➕", use_container_width=True)

    if add_clicked:
        if not new_ticker:
            st.error("Please enter a ticker symbol.")
        else:
            # Check if ticker already exists → update qty and avg buy price
            existing = next(
                (h for h in st.session_state["portfolio"] if h["ticker"] == new_ticker), None
            )
            if existing:
                # Weighted average buy price
                total_qty   = existing["qty"] + new_qty
                avg_buy     = (existing["qty"] * existing["buy_price"] + new_qty * new_buy) / total_qty
                existing["qty"]       = total_qty
                existing["buy_price"] = round(avg_buy, 4)
                st.success(f"Updated **{new_ticker}** — new qty: {total_qty:.2f}, avg buy: {avg_buy:.2f}")
            else:
                st.session_state["portfolio"].append({
                    "ticker"   : new_ticker,
                    "qty"      : new_qty,
                    "buy_price": new_buy,
                })
                st.success(f"Added **{new_ticker}** to your portfolio!")

    # ── DISPLAY PORTFOLIO ────────────────────────────────────────────────────────────
    if not st.session_state["portfolio"]:
        st.info("Your portfolio is empty. Add a holding above to get started.")
    else:
        st.markdown("---")
        st.markdown("### 📊 Portfolio Overview")

        # Fetch live prices and build rows
        rows = []
        total_invested   = 0.0
        total_cur_value  = 0.0

        with st.spinner("Fetching live prices…"):
            for h in st.session_state["portfolio"]:
                cur_price   = get_current_price(h["ticker"])
                invested    = h["qty"] * h["buy_price"]
                cur_value   = h["qty"] * cur_price
                pnl         = cur_value - invested
                pnl_pct     = (pnl / invested * 100) if invested else 0.0
                total_invested  += invested
                total_cur_value += cur_value
                rows.append({
                    "Ticker"         : h["ticker"],
                    "Qty"            : h["qty"],
                    "Buy Price"      : h["buy_price"],
                    "Current Price"  : round(cur_price, 2),
                    "Invested"       : round(invested, 2),
                    "Current Value"  : round(cur_value, 2),
                    "P&L"            : round(pnl, 2),
                    "P&L %"          : round(pnl_pct, 2),
                })

        total_pnl     = total_cur_value - total_invested
        total_pnl_pct = (total_pnl / total_invested * 100) if total_invested else 0.0

        # ── SUMMARY METRIC CARDS ──────────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            metric_card("Total Invested",    f"₹{total_invested:,.2f}")
        with m2:
            metric_card("Current Value",     f"₹{total_cur_value:,.2f}")
        with m3:
            pnl_color = "#37d67a" if total_pnl >= 0 else "#ff6b81"
            pnl_icon  = "▲" if total_pnl >= 0 else "▼"
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Total P&amp;L</div>
                    <div class="metric-value" style="color:{pnl_color};">
                        {pnl_icon} ₹{abs(total_pnl):,.2f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m4:
            metric_card("Overall Return",    f"{total_pnl_pct:+.2f}%",
                        total_pnl_pct, "%")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── HOLDINGS TABLE with colour-coded P&L ─────────────────────────────────
        st.markdown("### 📋 Holdings")
        df_port = pd.DataFrame(rows)

        def color_pnl(val):
            color = "#37d67a" if val >= 0 else "#ff6b81"
            return f"color: {color}; font-weight: 600;"

        styled = (
            df_port.style
            .applymap(color_pnl, subset=["P&L", "P&L %"])
            .format({
                "Buy Price"     : "{:.2f}",
                "Current Price" : "{:.2f}",
                "Invested"      : "{:,.2f}",
                "Current Value" : "{:,.2f}",
                "P&L"           : "{:+,.2f}",
                "P&L %"         : "{:+.2f}%",
            })
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # ── REMOVE HOLDING ────────────────────────────────────────────────────────
        st.markdown("### 🗑️ Remove a Holding")
        ticker_to_remove = st.selectbox(
            "Select ticker to remove",
            options=[h["ticker"] for h in st.session_state["portfolio"]],
            key="pt_remove_select"
        )
        if st.button("Remove ❌"):
            st.session_state["portfolio"] = [
                h for h in st.session_state["portfolio"]
                if h["ticker"] != ticker_to_remove
            ]
            st.success(f"Removed **{ticker_to_remove}** from portfolio.")
            st.rerun()

        st.markdown("---")

        # ── CHARTS ────────────────────────────────────────────────────────────────
        chart_col1, chart_col2 = st.columns(2)

        # Pie chart — allocation by current value
        with chart_col1:
            st.markdown("### 🥧 Allocation by Value")
            fig_pie = go.Figure(data=[go.Pie(
                labels=[r["Ticker"] for r in rows],
                values=[r["Current Value"] for r in rows],
                hole=0.42,
                textinfo="label+percent",
                marker=dict(colors=[
                    "#ff5f6d","#ffc371","#74b9ff","#a29bfe",
                    "#55efc4","#fdcb6e","#e17055","#6c5ce7",
                    "#00cec9","#fd79a8",
                ][:len(rows)])
            )])
            fig_pie.update_layout(
                template="plotly_dark",
                margin=dict(t=10, b=10, l=10, r=10),
                height=340,
                showlegend=True,
                legend=dict(orientation="v", x=1.02, y=0.5),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # Bar chart — P&L per stock
        with chart_col2:
            st.markdown("### 📊 P&L per Stock")
            pnl_colors = ["#37d67a" if r["P&L"] >= 0 else "#ff6b81" for r in rows]
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=[r["Ticker"]  for r in rows],
                y=[r["P&L"]     for r in rows],
                marker_color=pnl_colors,
                text=[f"{r['P&L %']:+.1f}%" for r in rows],
                textposition="outside",
            ))
            fig_bar.add_hline(y=0, line_width=1, line_color="gray")
            fig_bar.update_layout(
                template="plotly_dark",
                yaxis_title="P&L (₹)",
                margin=dict(t=10, b=10),
                height=340,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # Invested vs Current Value — grouped bar
        st.markdown("### 💰 Invested vs Current Value per Stock")
        fig_grp = go.Figure()
        fig_grp.add_trace(go.Bar(
            name="Invested",
            x=[r["Ticker"]   for r in rows],
            y=[r["Invested"] for r in rows],
            marker_color="#74b9ff",
        ))
        fig_grp.add_trace(go.Bar(
            name="Current Value",
            x=[r["Ticker"]        for r in rows],
            y=[r["Current Value"] for r in rows],
            marker_color="#ffc371",
        ))
        fig_grp.update_layout(
            barmode="group",
            template="plotly_dark",
            yaxis_title="Value (₹)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=30, b=10),
            height=340,
        )
        st.plotly_chart(fig_grp, use_container_width=True)

        # ── EXPORT ────────────────────────────────────────────────────────────────
        st.markdown("---")
        csv_portfolio = df_port.to_csv(index=False).encode("utf-8")
        st.download_button(
            label     = "⬇️ Download Portfolio as CSV",
            data      = csv_portfolio,
            file_name = "portfolio.csv",
            mime      = "text/csv",
        )