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

/* ── Loading bar ── */
.sv-loading-bar {
    width: 100%;
    height: 3px;
    background: rgba(255,255,255,0.07);
    border-radius: 99px;
    overflow: hidden;
    margin-bottom: 0.6rem;
}
.sv-loading-bar-inner {
    height: 100%;
    width: 40%;
    background: linear-gradient(90deg, #6C63FF, #ff5f6d, #ffc371);
    border-radius: 99px;
    animation: svSlide 1.4s ease-in-out infinite;
}
@keyframes svSlide {
    0%   { transform: translateX(-100%); }
    100% { transform: translateX(350%);  }
}

/* ── Skeleton shimmer ── */
.sv-skeleton {
    background: linear-gradient(90deg,
        rgba(255,255,255,0.04) 25%,
        rgba(255,255,255,0.10) 50%,
        rgba(255,255,255,0.04) 75%);
    background-size: 200% 100%;
    animation: svShimmer 1.6s infinite;
    border-radius: 10px;
}
@keyframes svShimmer {
    0%   { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}

/* ── Headline card ── */
.headline-card {
    background: linear-gradient(135deg,rgba(108,99,255,0.08),rgba(255,95,109,0.05));
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.6rem;
    transition: border-color 0.2s;
}
.headline-card:hover { border-color: rgba(108,99,255,0.4); }
.headline-source {
    font-size: 0.72rem;
    color: #6C63FF;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 3px;
}
.headline-title {
    font-size: 0.88rem;
    color: #f3f6ff;
    font-weight: 500;
    line-height: 1.45;
    text-decoration: none;
}
.headline-title:hover { color: #a29bfe; }
.headline-time {
    font-size: 0.72rem;
    color: #a1a7c4;
    margin-top: 4px;
}
.sentiment-pill {
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 600;
    padding: 2px 9px;
    border-radius: 99px;
    margin-left: 6px;
}
.pill-bull { background: rgba(46,204,113,0.18); color: #2ecc71; }
.pill-bear { background: rgba(231,76,60,0.18);  color: #e74c3c; }
.pill-neut { background: rgba(161,167,196,0.18);color: #a1a7c4; }

/* ── Sidebar sliding ticker ── */
.sv-ticker-wrap {
    width: 100%;
    overflow: hidden;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 6px 0;
    margin-bottom: 10px;
}
.sv-ticker-track {
    display: inline-flex;
    gap: 28px;
    white-space: nowrap;
    animation: svTicker 28s linear infinite;
}
.sv-ticker-wrap:hover .sv-ticker-track { animation-play-state: paused; }
@keyframes svTicker { from { transform: translateX(0); } to { transform: translateX(-50%); } }
.sv-ticker-item { font-size: 0.72rem; font-family: "Space Grotesk", sans-serif; }
.sv-ticker-label { color: #a1a7c4; font-weight: 500; margin-right: 4px; }
.sv-ticker-price { color: #f3f6ff; font-weight: 600; margin-right: 3px; }
.sv-ticker-up   { color: #2ecc71; font-weight: 600; }
.sv-ticker-dn   { color: #e74c3c; font-weight: 600; }

/* ── Top performer card ── */
.sv-top-card {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 6px 10px;
    margin-bottom: 6px;
    font-family: "Space Grotesk", sans-serif;
}
.sv-top-rank { font-size: 0.72rem; color: #a1a7c4; width: 18px; }
.sv-top-name { font-size: 0.8rem;  color: #f3f6ff; font-weight: 600; flex: 1; margin-left: 6px; }
.sv-top-pct  { font-size: 0.8rem;  font-weight: 700; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
# ── Logo + Hero ──────────────────────────────────────────────────────────────
LOGO_URL = "https://cdn-icons-png.flaticon.com/512/6198/6198527.png"
st.markdown(
    f"""
    <div style="display:flex;align-items:center;justify-content:center;gap:14px;margin:0.5rem 0 0.2rem;">
        <img src="{LOGO_URL}" width="52" height="52"
             style="border-radius:14px;background:rgba(108,99,255,0.15);padding:6px;"
             onerror="this.style.display='none'"/>
        <div class="hero-title" style="margin:0;">Stock Vision</div>
    </div>
    <p class="hero-subtitle">Real-time insights, news, AI forecasts — powered by multivariate LSTM.</p>
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




# =======================================================================================
# INLINE LOADING HELPERS  — shown inside the page while data is being fetched
# =======================================================================================

def show_loading_bar(message: str = "Fetching data…"):
    """
    Renders a slim animated gradient bar + status message inline in the page.
    Call this BEFORE the slow operation, then replace with st.empty() pattern.
    """
    st.markdown(
        f"""
        <div style="margin:0.5rem 0 0.25rem 0;">
            <div class="sv-loading-bar"><div class="sv-loading-bar-inner"></div></div>
            <span style="font-size:0.82rem;color:#a1a7c4;">⏳ {message}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_skeleton_rows(n: int = 4):
    """
    Shows n skeleton placeholder rows while a table/chart is loading.
    Replace the container with st.empty() and call .empty() when done.
    """
    rows_html = "".join(
        f'<div class="sv-skeleton" style="height:38px;margin-bottom:8px;opacity:{0.9 - i*0.15:.2f};"></div>'
        for i in range(n)
    )
    st.markdown(rows_html, unsafe_allow_html=True)


# =======================================================================================
# LATEST HEADLINES FETCHER  — used on Home tab
# =======================================================================================

# =======================================================================================
# MARKET SNAPSHOT  — Gold, Silver, Oil, BTC, Indices (yfinance, no extra API)
# =======================================================================================

MARKET_WATCHLIST = [
    {"label": "Gold",       "ticker": "GC=F",   "icon": "🥇"},
    {"label": "Silver",     "ticker": "SI=F",   "icon": "🥈"},
    {"label": "Crude Oil",  "ticker": "CL=F",   "icon": "🛢️"},
    {"label": "Bitcoin",    "ticker": "BTC-USD", "icon": "₿"},
    {"label": "S&P 500",    "ticker": "^GSPC",  "icon": "🇺🇸"},
    {"label": "NIFTY 50",   "ticker": "^NSEI",  "icon": "🇮🇳"},
    {"label": "NASDAQ",     "ticker": "^IXIC",  "icon": "💻"},
    {"label": "SENSEX",     "ticker": "^BSESN", "icon": "📈"},
]

@st.cache_data(ttl=120)   # refresh every 2 min
def fetch_market_snapshot():
    """Fetch current price + daily change for Gold, Silver, Oil, BTC, indices."""
    results = []
    for item in MARKET_WATCHLIST:
        try:
            hist = yf.download(item["ticker"], period="2d", progress=False, auto_adjust=True)
            if hist.empty or len(hist) < 2:
                continue
            close = hist["Close"].dropna()
            cur   = float(close.iloc[-1])
            prev  = float(close.iloc[-2])
            chg   = cur - prev
            pct   = (chg / prev * 100) if prev else 0.0
            results.append({
                "label"  : item["label"],
                "icon"   : item["icon"],
                "price"  : cur,
                "change" : chg,
                "pct"    : pct,
            })
        except Exception:
            pass
    return results


def render_market_snapshot(snapshot: list):
    """Renders compact market ticker cards in a 2-column grid."""
    if not snapshot:
        st.caption("Market data unavailable.")
        return
    cards_html = '<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:1rem;">'
    for s in snapshot:
        color  = "#2ecc71" if s["change"] >= 0 else "#e74c3c"
        arrow  = "▲" if s["change"] >= 0 else "▼"
        price_str = f"{s['price']:,.2f}"
        cards_html += f"""
        <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
                    border-radius:12px;padding:0.6rem 0.75rem;">
            <div style="font-size:0.72rem;color:#a1a7c4;margin-bottom:2px;">
                {s['icon']} {s['label']}
            </div>
            <div style="font-size:1rem;font-weight:600;color:#f3f6ff;">{price_str}</div>
            <div style="font-size:0.75rem;color:{color};font-weight:500;">
                {arrow} {abs(s['pct']):.2f}%
            </div>
        </div>"""
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)


# =======================================================================================
# HEADLINES  — Alpha Vantage general market news (no ticker filter)
# =======================================================================================

# RSS feeds — no API key needed, always available on free tier
_RSS_FEEDS = [
    ("Reuters Markets",   "https://feeds.reuters.com/reuters/businessNews"),
    ("CNBC Markets",      "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839135"),
    ("Yahoo Finance",     "https://finance.yahoo.com/news/rssindex"),
    ("Investing.com",     "https://www.investing.com/rss/news.rss"),
    ("MarketWatch",       "https://feeds.marketwatch.com/marketwatch/topstories/"),
]

# Keywords that map article titles to a category pill
_CATEGORY_KEYWORDS = {
    "Gold"       : ["gold","silver","metal","commodity","commodities","oil","crude"],
    "Crypto"     : ["bitcoin","crypto","ethereum","btc","eth","blockchain"],
    "Economy"    : ["fed","inflation","gdp","rate","interest","economy","economic","recession"],
    "Earnings"   : ["earnings","revenue","profit","quarterly","results","eps"],
    "Markets"    : ["market","stock","index","nifty","nasdaq","sensex","s&p","dow"],
}

def _classify(title: str) -> str:
    t = title.lower()
    for cat, kws in _CATEGORY_KEYWORDS.items():
        if any(k in t for k in kws):
            return cat
    return "Finance"

def _parse_rss_date(date_str: str) -> str:
    """Parse RSS pubDate into friendly string."""
    for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S %Z"):
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            return dt.strftime("%b %d  %H:%M")
        except Exception:
            pass
    return date_str[:16] if date_str else ""

@st.cache_data(ttl=300)
def fetch_market_headlines(api_key: str = "", limit: int = 10):
    """
    Fetches latest financial headlines from multiple RSS feeds.
    No API key required — works on Alpha Vantage free tier too.
    Returns list of dicts: title, url, source, time, category, score.
    """
    import xml.etree.ElementTree as ET
    out = []
    for source_name, rss_url in _RSS_FEEDS:
        if len(out) >= limit:
            break
        try:
            resp = requests.get(
                rss_url,
                timeout=6,
                headers={"User-Agent": "Mozilla/5.0 StockVision/1.0"},
            )
            if resp.status_code != 200:
                continue
            root = ET.fromstring(resp.content)
            ns   = {"content": "http://purl.org/rss/1.0/modules/content/"}
            items = root.findall(".//item")
            for item in items[:3]:   # max 3 per source for variety
                title   = (item.findtext("title") or "").strip()
                url     = (item.findtext("link")  or "#").strip()
                pub     = (item.findtext("pubDate") or "").strip()
                if not title or not url:
                    continue
                out.append({
                    "title"   : title,
                    "url"     : url,
                    "source"  : source_name,
                    "time"    : _parse_rss_date(pub),
                    "category": _classify(title),
                    "score"   : 0.0,   # RSS has no sentiment — shown as Neutral
                })
                if len(out) >= limit:
                    break
        except Exception:
            continue
    return out[:limit]


def render_headline_cards(headlines: list):
    """Renders headline cards — one st.markdown call per card for reliable rendering."""
    if not headlines:
        st.caption("No headlines available right now.")
        return
    for h in headlines:
        score = h.get("score", 0.0)
        if score > 0.15:
            pill = '<span class="sentiment-pill pill-bull">▲ Bullish</span>'
        elif score < -0.15:
            pill = '<span class="sentiment-pill pill-bear">▼ Bearish</span>'
        else:
            pill = '<span class="sentiment-pill pill-neut">● Neutral</span>'
        cat   = h.get("category", "Finance")
        src_  = h.get("source", "")
        title = h.get("title", "")
        url   = h.get("url", "#")
        time_ = h.get("time", "")
        st.markdown(
            f"""<div class="headline-card">
                <div class="headline-source">
                    {src_}
                    <span style="font-size:0.68rem;color:#a1a7c4;margin-left:6px;
                                 background:rgba(255,255,255,0.06);padding:1px 7px;
                                 border-radius:99px;">{cat}</span>
                    {pill}
                </div>
                <a class="headline-title" href="{url}" target="_blank">{title}</a>
                <div class="headline-time">🕐 {time_}</div>
            </div>""",
            unsafe_allow_html=True,
        )


# kept for backward compat
@st.cache_data(ttl=300)
def fetch_home_headlines(ticker: str, api_key: str, limit: int = 6):
    return fetch_market_headlines(limit=limit)


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
# SIDEBAR DATA HELPERS
# =======================================================================================

TICKER_TAPE = [
    ("Gold",    "GC=F"),  ("Silver",  "SI=F"),  ("Oil",     "CL=F"),
    ("BTC",     "BTC-USD"),("ETH",    "ETH-USD"),
    ("S&P500",  "^GSPC"), ("NASDAQ",  "^IXIC"),  ("NIFTY",   "^NSEI"),
    ("SENSEX",  "^BSESN"),
    ("AAPL",    "AAPL"),  ("MSFT",    "MSFT"),   ("NVDA",    "NVDA"),
    ("TSLA",    "TSLA"),  ("GOOGL",   "GOOGL"),  ("AMZN",    "AMZN"),
    ("RELIANCE","RELIANCE.NS"),("TCS", "TCS.NS"), ("INFY",    "INFY.NS"),
]

TOP_PERFORMERS_POOL = [
    "AAPL","MSFT","NVDA","TSLA","GOOGL","AMZN","META","NFLX","AMD","ADBE",
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
    "WIPRO.NS","BAJFINANCE.NS","TATAMOTORS.NS",
]

@st.cache_data(ttl=180)   # refresh every 3 min
def fetch_ticker_tape():
    """Returns list of (label, price_str, pct, up:bool) for the sliding tape."""
    items = []
    for label, sym in TICKER_TAPE:
        try:
            hist = yf.download(sym, period="2d", progress=False, auto_adjust=True)
            if hist.empty or len(hist) < 2:
                continue
            close = hist["Close"].dropna()
            cur   = float(close.iloc[-1])
            prev  = float(close.iloc[-2])
            pct   = (cur - prev) / prev * 100 if prev else 0.0
            price_str = f"{cur:,.2f}" if cur < 10000 else f"{cur:,.0f}"
            items.append((label, price_str, pct, pct >= 0))
        except Exception:
            pass
    return items


@st.cache_data(ttl=180)
def fetch_top_performers(n: int = 5):
    """Returns top n gainers from TOP_PERFORMERS_POOL sorted by daily % change."""
    results = []
    for sym in TOP_PERFORMERS_POOL:
        try:
            hist = yf.download(sym, period="2d", progress=False, auto_adjust=True)
            if hist.empty or len(hist) < 2:
                continue
            close = hist["Close"].dropna()
            cur   = float(close.iloc[-1])
            prev  = float(close.iloc[-2])
            pct   = (cur - prev) / prev * 100 if prev else 0.0
            # strip .NS suffix for display
            display = sym.replace(".NS","").replace(".BO","")
            results.append((display, pct, pct >= 0))
        except Exception:
            pass
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:n]


def render_sidebar_ticker(tape: list):
    """Renders a sliding marquee ticker strip in the sidebar."""
    if not tape:
        return
    # Build items HTML — duplicate for seamless loop
    def item_html(label, price, pct, up):
        cls   = "sv-ticker-up" if up else "sv-ticker-dn"
        arrow = "▲" if up else "▼"
        return (f'<span class="sv-ticker-item">'
                f'<span class="sv-ticker-label">{label}</span>'
                f'<span class="sv-ticker-price">{price}</span>'
                f'<span class="{cls}">{arrow}{abs(pct):.2f}%</span>'
                f'</span>')
    items_html = "".join(item_html(*t) for t in tape)
    # duplicate for seamless loop
    track = items_html + items_html
    st.sidebar.markdown(
        f'<div class="sv-ticker-wrap"><div class="sv-ticker-track">{track}</div></div>',
        unsafe_allow_html=True,
    )


def render_top_performers(performers: list):
    """Renders ranked top gainer cards in the sidebar."""
    if not performers:
        st.sidebar.caption("Data unavailable.")
        return
    medals = ["🥇","🥈","🥉","4️⃣","5️⃣"]
    cards  = ""
    for i, (name, pct, up) in enumerate(performers):
        color = "#2ecc71" if up else "#e74c3c"
        arrow = "▲" if up else "▼"
        cards += (f'<div class="sv-top-card">'
                  f'<span class="sv-top-rank">{medals[i]}</span>'
                  f'<span class="sv-top-name">{name}</span>'
                  f'<span class="sv-top-pct" style="color:{color};">'
                  f'{arrow}{abs(pct):.2f}%</span>'
                  f'</div>')
    st.sidebar.markdown(cards, unsafe_allow_html=True)


# =======================================================================================
# SIDEBAR
# =======================================================================================

# ── Logo in sidebar ──────────────────────────────────────────────────────────
st.sidebar.markdown(
    """
    <div style="display:flex;align-items:center;gap:10px;padding:4px 0 10px;">
        <img src="https://cdn-icons-png.flaticon.com/512/6198/6198527.png"
             width="36" height="36"
             style="border-radius:10px;background:rgba(108,99,255,0.15);padding:4px;"
             onerror="this.style.display='none'"/>
        <span style="font-size:1.1rem;font-weight:700;color:#f3f6ff;">Stock Vision</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Sliding market ticker at the very top ────────────────────────────────────
_tape = fetch_ticker_tape()
render_sidebar_ticker(_tape)

# ── Top performers ────────────────────────────────────────────────────────────
st.sidebar.markdown(
    "<div style='font-size:0.75rem;font-weight:600;color:#a1a7c4;"
    "text-transform:uppercase;letter-spacing:0.06em;"
    "margin:4px 0 6px;'>🔥 Top Performers Today</div>",
    unsafe_allow_html=True,
)
_performers = fetch_top_performers(5)
render_top_performers(_performers)

st.sidebar.markdown("---")
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

    # ── LEFT COLUMN: animation + market snapshot + headlines ────────────────────────────
    with top_col:
        lottie_url  = "https://lottie.host/19ad9b6a-1882-4957-8216-bafa10a2ceaf/vA8lTy3DBm.json"
        lottie_anim = load_lottieurl(lottie_url)
        if lottie_anim:
            st_lottie(lottie_anim, height=180)

        # ── Market Snapshot: Gold, Silver, Oil, BTC, Indices ─────────────────────────
        st.markdown(
            "<div style='font-size:0.82rem;font-weight:600;color:#a1a7c4;"
            "text-transform:uppercase;letter-spacing:0.06em;"
            "margin:0.4rem 0 0.5rem;'>📊 Live Market Snapshot</div>",
            unsafe_allow_html=True,
        )
        snap_placeholder = st.empty()
        with snap_placeholder:
            show_loading_bar("Fetching market prices…")
        snapshot = fetch_market_snapshot()
        snap_placeholder.empty()
        render_market_snapshot(snapshot)

        # ── Latest Market Headlines ───────────────────────────────────────────────────
        st.markdown(
            "<div style='font-size:0.82rem;font-weight:600;color:#a1a7c4;"
            "text-transform:uppercase;letter-spacing:0.06em;"
            "margin:0.6rem 0 0.5rem;'>📰 Latest Market Headlines</div>",
            unsafe_allow_html=True,
        )
        hl_placeholder = st.empty()
        with hl_placeholder:
            show_loading_bar("Fetching headlines…")
        headlines = fetch_market_headlines(limit=10)
        hl_placeholder.empty()
        render_headline_cards(headlines)

    # ── RIGHT COLUMN: date pickers + metrics + table ──────────────────────────────────
    with chart_col:
        date_cols  = st.columns(2)
        start_date = date_cols[0].date_input("Start Date", date.today() - timedelta(days=365))
        end_date   = date_cols[1].date_input("End Date",   date.today())

        if start_date > end_date:
            st.error("End date cannot be before start date.")
        else:
            # ── Inline loading state: bar + skeleton rows ─────────────────────────
            load_placeholder = st.empty()
            with load_placeholder.container():
                show_loading_bar(f"Loading {ticker} price data…")
                show_skeleton_rows(4)

                data, err = get_stock_data(ticker, start_date, end_date)
            load_placeholder.empty()   # wipe skeleton once data is ready

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
                with metric_cols[0]: metric_card("Latest Close",   f"${latest_close:.2f}", delta_price)
                with metric_cols[1]: metric_card("Volume",         f"{latest_vol:,.0f}")
                with metric_cols[2]: metric_card("Daily Change %", f"{delta_pct:+.2f}%",  delta_pct, "%")

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
        viz_placeholder = st.empty()
        with viz_placeholder.container():
            show_loading_bar(f"Preparing charts for {ticker}…")
            show_skeleton_rows(5)
        data, err = get_stock_data(ticker, start_date, end_date)
        viz_placeholder.empty()

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
        news_placeholder = st.empty()
        with news_placeholder.container():
            show_loading_bar(f"Fetching latest news for {ticker}…")
            show_skeleton_rows(3)
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        news_placeholder.empty()

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

        pred_placeholder = st.empty()
        with pred_placeholder.container():
            show_loading_bar(f"Loading AI model for {selected_stock}…")
            show_skeleton_rows(3)
        model, scaler, sentiment = load_or_create_model(selected_stock, api_key)
        pred_placeholder.empty()

        # Show sentiment warning if Alpha Vantage had no data for this ticker
        if "sentiment_warning" in st.session_state:
            st.warning(st.session_state["sentiment_warning"])
            del st.session_state["sentiment_warning"]

        feat_placeholder = st.empty()
        with feat_placeholder.container():
            show_loading_bar(f"Computing features for {selected_stock}…")
            show_skeleton_rows(2)
        feat_df, feature_values, raw = prepare_data_for_prediction(selected_stock, sentiment)
        feat_placeholder.empty()

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

# ── Shared color palette — used across ALL portfolio charts ──────────────────────────
PORTFOLIO_COLORS = [
    "#6C63FF",   # indigo
    "#FF6584",   # rose
    "#43C6AC",   # teal
    "#F7971E",   # amber
    "#A78BFA",   # violet
    "#34D399",   # emerald
    "#FB923C",   # orange
    "#38BDF8",   # sky blue
    "#F472B6",   # pink
    "#FACC15",   # yellow
]

# ── Shared chart layout defaults — call .update_layout(**CHART_LAYOUT) on every fig ─
CHART_LAYOUT = dict(
    template    = "plotly_dark",
    paper_bgcolor = "rgba(19,23,34,0)",    # fully transparent → blends with app bg
    plot_bgcolor  = "rgba(28,34,51,0.55)", # faint dark surface for plot area
    font        = dict(family="Space Grotesk, sans-serif", size=13, color="#a1a7c4"),
    title_font  = dict(family="Space Grotesk, sans-serif", size=15, color="#f3f6ff"),
    legend      = dict(
        bgcolor     = "rgba(19,23,34,0.6)",
        bordercolor = "rgba(255,255,255,0.08)",
        borderwidth = 1,
        font        = dict(size=12, color="#f3f6ff"),
    ),
    xaxis = dict(
        gridcolor   = "rgba(255,255,255,0.05)",
        linecolor   = "rgba(255,255,255,0.08)",
        tickfont    = dict(size=12, color="#a1a7c4"),
        showgrid    = True,
    ),
    yaxis = dict(
        gridcolor   = "rgba(255,255,255,0.05)",
        linecolor   = "rgba(255,255,255,0.08)",
        tickfont    = dict(size=12, color="#a1a7c4"),
        showgrid    = True,
        zeroline    = False,
    ),
    margin  = dict(t=50, b=50, l=60, r=30),
    height  = 420,    # uniform height for ALL charts
    hoverlabel = dict(
        bgcolor    = "#1c2233",
        bordercolor= "rgba(255,255,255,0.15)",
        font_size  = 13,
        font_color = "#f3f6ff",
    ),
)

if selected_option == "Portfolio":
    st.subheader("💼 Portfolio Tracker")
    st.caption("Add your holdings below. Current prices are fetched live from Yahoo Finance.")

    # ── Session state initialisation ─────────────────────────────────────────────────
    if "portfolio" not in st.session_state:
        st.session_state["portfolio"] = []

    # ── Helper: fetch current price ───────────────────────────────────────────────────
    @st.cache_data(ttl=60)
    def get_current_price(tkr: str) -> float:
        try:
            info  = yf.Ticker(tkr).fast_info
            price = getattr(info, "last_price", None) or getattr(info, "regular_market_price", None)
            if price:
                return float(price)
            hist = yf.download(tkr, period="2d", progress=False, auto_adjust=True)
            if not hist.empty:
                return float(hist["Close"].iloc[-1])
        except Exception:
            pass
        return 0.0

    # ── Currency symbol helper ────────────────────────────────────────────────────────
    def currency_symbol(tkr: str) -> str:
        return "₹" if tkr.upper().endswith(".NS") or tkr.upper().endswith(".BO") else "$"

    # ── ADD HOLDING FORM ──────────────────────────────────────────────────────────────
    st.markdown("### ➕ Add a Holding")
    col1, col2, col3, col4 = st.columns([2, 1.3, 1.5, 1])
    with col1:
        new_ticker = st.text_input(
            "Ticker symbol",
            placeholder="e.g. AAPL, RELIANCE.NS",
            key="pt_ticker",
        ).upper().strip()
    with col2:
        new_qty = st.number_input(
            "Quantity", min_value=0.01, value=1.0, step=1.0, key="pt_qty"
        )
    with col3:
        new_buy = st.number_input(
            "Buy price / share", min_value=0.01, value=100.0, step=0.01, key="pt_buy"
        )
    with col4:
        st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
        add_clicked = st.button("Add ➕", use_container_width=True)

    if add_clicked:
        if not new_ticker:
            st.error("Please enter a ticker symbol.")
        else:
            existing = next(
                (h for h in st.session_state["portfolio"] if h["ticker"] == new_ticker), None
            )
            if existing:
                total_qty = existing["qty"] + new_qty
                avg_buy   = (existing["qty"] * existing["buy_price"] + new_qty * new_buy) / total_qty
                existing["qty"]       = total_qty
                existing["buy_price"] = round(avg_buy, 4)
                st.success(f"Updated **{new_ticker}** — qty: {total_qty:.2f}, avg buy: {avg_buy:.2f}")
            else:
                st.session_state["portfolio"].append(
                    {"ticker": new_ticker, "qty": new_qty, "buy_price": new_buy}
                )
                st.success(f"Added **{new_ticker}** to your portfolio!")

    # ── PORTFOLIO DISPLAY ─────────────────────────────────────────────────────────────
    if not st.session_state["portfolio"]:
        st.info("Your portfolio is empty. Add a holding above to get started.")
    else:
        st.markdown("---")

        # Build rows with live prices
        rows            = []
        total_invested  = 0.0
        total_cur_value = 0.0

        with st.spinner("Fetching live prices…"):
            for h in st.session_state["portfolio"]:
                cur_price = get_current_price(h["ticker"])
                sym       = currency_symbol(h["ticker"])
                invested  = h["qty"] * h["buy_price"]
                cur_value = h["qty"] * cur_price
                pnl       = cur_value - invested
                pnl_pct   = (pnl / invested * 100) if invested else 0.0
                total_invested  += invested
                total_cur_value += cur_value
                rows.append({
                    "Ticker"        : h["ticker"],
                    "Qty"           : h["qty"],
                    "Buy Price"     : round(h["buy_price"], 2),
                    "Current Price" : round(cur_price, 2),
                    "Invested"      : round(invested, 2),
                    "Current Value" : round(cur_value, 2),
                    "P&L"           : round(pnl, 2),
                    "P&L %"         : round(pnl_pct, 2),
                    "_sym"          : sym,
                })

        total_pnl     = total_cur_value - total_invested
        total_pnl_pct = (total_pnl / total_invested * 100) if total_invested else 0.0

        # ── SUMMARY METRIC CARDS ──────────────────────────────────────────────────────
        st.markdown("### 📊 Portfolio Overview")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            metric_card("Total Invested",  f"${total_invested:,.2f}")
        with m2:
            metric_card("Current Value",   f"${total_cur_value:,.2f}")
        with m3:
            pnl_color = "#2ecc71" if total_pnl >= 0 else "#e74c3c"
            pnl_icon  = "▲" if total_pnl >= 0 else "▼"
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-label">Total P&amp;L</div>
                    <div class="metric-value" style="color:{pnl_color};">
                        {pnl_icon} ${abs(total_pnl):,.2f}
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )
        with m4:
            metric_card("Overall Return", f"{total_pnl_pct:+.2f}%", total_pnl_pct, "%")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── HOLDINGS TABLE ────────────────────────────────────────────────────────────
        st.markdown("### 📋 Holdings")
        df_port = pd.DataFrame([{k: v for k, v in r.items() if k != "_sym"} for r in rows])

        def color_pnl(val):
            return "color: #2ecc71; font-weight:600;" if val >= 0 else "color: #e74c3c; font-weight:600;"

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
            .set_properties(**{
                "background-color": "#1c2233",
                "color"           : "#f3f6ff",
                "border-color"    : "rgba(255,255,255,0.06)",
            })
            .set_table_styles([{
                "selector": "th",
                "props"   : [
                    ("background-color", "#131722"),
                    ("color", "#a1a7c4"),
                    ("font-size", "13px"),
                    ("font-weight", "600"),
                    ("border-bottom", "1px solid rgba(255,255,255,0.08)"),
                ],
            }])
        )
        st.dataframe(styled, use_container_width=True, hide_index=True, height=240)

        # ── REMOVE HOLDING ────────────────────────────────────────────────────────────
        rem_col, _ = st.columns([2, 3])
        with rem_col:
            ticker_to_remove = st.selectbox(
                "🗑️ Remove a holding",
                options=[h["ticker"] for h in st.session_state["portfolio"]],
                key="pt_remove_select",
            )
            if st.button("Remove ❌", use_container_width=True):
                st.session_state["portfolio"] = [
                    h for h in st.session_state["portfolio"]
                    if h["ticker"] != ticker_to_remove
                ]
                st.success(f"Removed **{ticker_to_remove}** from portfolio.")
                st.rerun()

        st.markdown("---")

        # ── CHART 1 + 2  (side by side, equal height) ────────────────────────────────
        chart_col1, chart_col2 = st.columns(2, gap="medium")

        # --- Donut pie — portfolio allocation ---
        with chart_col1:
            st.markdown("#### 🥧 Allocation by Current Value")
            pie_colors = PORTFOLIO_COLORS[:len(rows)]
            fig_pie = go.Figure(go.Pie(
                labels      = [r["Ticker"] for r in rows],
                values      = [r["Current Value"] for r in rows],
                hole        = 0.5,
                textinfo    = "label+percent",
                textfont    = dict(size=12, color="#f3f6ff"),
                hovertemplate = "<b>%{label}</b><br>Value: $%{value:,.2f}<br>Share: %{percent}<extra></extra>",
                marker      = dict(
                    colors = pie_colors,
                    line   = dict(color="#131722", width=2),   # gap between slices
                ),
                pull        = [0.03] * len(rows),              # slight explode effect
            ))
            fig_pie.update_layout(
                template      = "plotly_dark",
                paper_bgcolor = "rgba(19,23,34,0)",
                plot_bgcolor  = "rgba(28,34,51,0.55)",
                height        = 420,
                showlegend    = True,
                legend        = dict(
                    orientation = "v",
                    x=1.02, y=0.5,
                    bgcolor     = "rgba(19,23,34,0.6)",
                    bordercolor = "rgba(255,255,255,0.08)",
                    borderwidth = 1,
                    font        = dict(size=12, color="#f3f6ff"),
                ),
                margin      = dict(t=40, b=40, l=20, r=120),
                font        = dict(family="Space Grotesk, sans-serif", size=13, color="#a1a7c4"),
                hoverlabel  = dict(bgcolor="#1c2233", bordercolor="rgba(255,255,255,0.15)",
                                   font_size=13, font_color="#f3f6ff"),
                annotations = [dict(
                    text      = f"<b>{len(rows)}<br>stocks</b>",
                    x=0.5, y=0.5,
                    font_size = 14,
                    font_color= "#f3f6ff",
                    showarrow = False,
                )],
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # --- Horizontal bar — P&L per stock ---
        with chart_col2:
            st.markdown("#### 📊 P&L per Stock")
            sorted_rows = sorted(rows, key=lambda r: r["P&L"])
            bar_colors  = ["#2ecc71" if r["P&L"] >= 0 else "#e74c3c" for r in sorted_rows]
            fig_pnl = go.Figure(go.Bar(
                x             = [r["P&L"] for r in sorted_rows],
                y             = [r["Ticker"] for r in sorted_rows],
                orientation   = "h",
                marker_color  = bar_colors,
                marker_line   = dict(width=0),
                text          = [f"{r['P&L %']:+.1f}%" for r in sorted_rows],
                textposition  = "outside",
                textfont      = dict(size=12, color="#f3f6ff"),
                hovertemplate = "<b>%{y}</b><br>P&L: $%{x:+,.2f}<extra></extra>",
            ))
            fig_pnl.add_vline(
                x=0,
                line_width = 1.5,
                line_color = "rgba(255,255,255,0.25)",
            )
            fig_pnl.update_layout(
                template      = "plotly_dark",
                paper_bgcolor = "rgba(19,23,34,0)",
                plot_bgcolor  = "rgba(28,34,51,0.55)",
                height        = 420,
                xaxis_title   = "Profit / Loss ($)",
                yaxis_title   = "",
                margin        = dict(t=40, b=50, l=90, r=70),
                font          = dict(family="Space Grotesk, sans-serif", size=13, color="#a1a7c4"),
                hoverlabel    = dict(bgcolor="#1c2233", bordercolor="rgba(255,255,255,0.15)",
                                     font_size=13, font_color="#f3f6ff"),
                xaxis = dict(
                    gridcolor  = "rgba(255,255,255,0.05)",
                    linecolor  = "rgba(255,255,255,0.08)",
                    tickfont   = dict(size=12, color="#a1a7c4"),
                    showgrid   = True,
                    tickprefix = "$",
                    zeroline   = False,
                ),
                yaxis = dict(
                    gridcolor = "rgba(255,255,255,0.05)",
                    linecolor = "rgba(255,255,255,0.08)",
                    tickfont  = dict(size=12, color="#a1a7c4"),
                    showgrid  = True,
                    zeroline  = False,
                ),
                legend = dict(
                    bgcolor     = "rgba(19,23,34,0.6)",
                    bordercolor = "rgba(255,255,255,0.08)",
                    borderwidth = 1,
                    font        = dict(size=12, color="#f3f6ff"),
                ),
            )
            st.plotly_chart(fig_pnl, use_container_width=True)

        # ── CHART 3 — Invested vs Current Value grouped bar (full width) ──────────────
        st.markdown("#### 💰 Invested vs Current Value")
        ticker_labels = [r["Ticker"] for r in rows]
        fig_grp = go.Figure()
        fig_grp.add_trace(go.Bar(
            name          = "Invested",
            x             = ticker_labels,
            y             = [r["Invested"] for r in rows],
            marker_color  = "#6C63FF",
            marker_line   = dict(width=0),
            hovertemplate = "<b>%{x}</b><br>Invested: $%{y:,.2f}<extra></extra>",
        ))
        fig_grp.add_trace(go.Bar(
            name          = "Current Value",
            x             = ticker_labels,
            y             = [r["Current Value"] for r in rows],
            marker_color  = "#43C6AC",
            marker_line   = dict(width=0),
            hovertemplate = "<b>%{x}</b><br>Current: $%{y:,.2f}<extra></extra>",
        ))
        fig_grp.update_layout(
            template      = "plotly_dark",
            paper_bgcolor = "rgba(19,23,34,0)",
            plot_bgcolor  = "rgba(28,34,51,0.55)",
            height        = 400,
            barmode       = "group",
            bargap        = 0.22,
            bargroupgap   = 0.08,
            yaxis_title   = "Value ($)",
            xaxis_title   = "Ticker",
            font          = dict(family="Space Grotesk, sans-serif", size=13, color="#a1a7c4"),
            hoverlabel    = dict(bgcolor="#1c2233", bordercolor="rgba(255,255,255,0.15)",
                                 font_size=13, font_color="#f3f6ff"),
            legend        = dict(
                orientation = "h",
                yanchor     = "bottom",
                y           = 1.02,
                xanchor     = "right",
                x           = 1,
                bgcolor     = "rgba(19,23,34,0.6)",
                bordercolor = "rgba(255,255,255,0.08)",
                borderwidth = 1,
                font        = dict(size=12, color="#f3f6ff"),
            ),
            margin = dict(t=60, b=50, l=70, r=30),
        )
        st.plotly_chart(fig_grp, use_container_width=True)

        # ── EXPORT ────────────────────────────────────────────────────────────────────
        st.markdown("---")
        csv_portfolio = df_port.to_csv(index=False).encode("utf-8")
        st.download_button(
            label     = "⬇️ Download Portfolio as CSV",
            data      = csv_portfolio,
            file_name = "portfolio.csv",
            mime      = "text/csv",
        )
        