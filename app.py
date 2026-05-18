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
# ── Logo embedded as base64 — works offline, no CDN needed ──────────────────
_LOGO_B64 = "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"
_LOGO_SRC  = f"data:image/png;base64,{_LOGO_B64}"

st.markdown(
    f"""
    <div style="display:flex;align-items:center;justify-content:center;gap:14px;margin:0.5rem 0 0.2rem;">
        <img src="{_LOGO_SRC}" width="90" height="90"
             style="border-radius:18px;padding:4px;"/>
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
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        return r.json()
    except (requests.exceptions.RequestException, requests.exceptions.ConnectionError) as e:
        # Handle network errors gracefully - animation will be skipped
        return None
    except Exception as e:
        # Handle JSON parsing errors or other unexpected issues
        return None




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
    from keras.layers import Input
    model = Sequential([
        Input(shape=(seq_len, n_features)),          # explicit Input layer — Keras 3 compatible
        LSTM(units=100, return_sequences=True),
        Dropout(0.2),
        LSTM(units=100, return_sequences=True),
        Dropout(0.2),
        LSTM(units=100),
        Dropout(0.2),
        Dense(units=1),
    ])
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
        try:
            model  = load_model(model_path)
            scaler = joblib.load(scaler_path)
            # sentiment is not stored — fetch fresh for prediction
            sentiment = fetch_avg_sentiment(ticker, api_key) if api_key else 0.0
            return model, scaler, sentiment
        except Exception as load_err:
            # Saved model was built with an older Keras/TF that used 'batch_shape'
            # in InputLayer config (incompatible with Keras 3.x which uses 'shape').
            # Delete stale files and fall through to retrain automatically.
            st.warning(
                f"⚠️ Saved model for **{ticker}** is incompatible with the current "
                f"Keras version (`{load_err}`). "
                "Deleting stale files and retraining automatically — this may take a minute…"
            )
            for p in [model_path, scaler_path]:
                try:
                    os.remove(p)
                except OSError:
                    pass

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
# TRENDING STOCKS — Top Gainers & Losers
# =======================================================================================

TRENDING_POOL_US = [
    ("AAPL","Apple"),("MSFT","Microsoft"),("NVDA","NVIDIA"),("TSLA","Tesla"),
    ("GOOGL","Alphabet"),("AMZN","Amazon"),("META","Meta"),("NFLX","Netflix"),
    ("AMD","AMD"),("ADBE","Adobe"),("CRM","Salesforce"),("PYPL","PayPal"),
    ("UBER","Uber"),("JPM","JP Morgan"),("V","Visa"),("MA","Mastercard"),
]
TRENDING_POOL_IN = [
    ("RELIANCE.NS","Reliance"),("TCS.NS","TCS"),("INFY.NS","Infosys"),
    ("HDFCBANK.NS","HDFC Bank"),("ICICIBANK.NS","ICICI Bank"),
    ("SBIN.NS","SBI"),("WIPRO.NS","Wipro"),("BAJFINANCE.NS","Bajaj Finance"),
    ("TATAMOTORS.NS","Tata Motors"),("MARUTI.NS","Maruti"),
    ("ADANIENT.NS","Adani Ent."),("SUNPHARMA.NS","Sun Pharma"),
    ("LT.NS","L&T"),("ITC.NS","ITC"),("HINDUNILVR.NS","HUL"),
]

@st.cache_data(ttl=180)
def fetch_trending_stocks():
    """
    Returns (gainers, losers) — each a list of dicts with
    ticker, name, price, change, pct — sorted by % move.
    Pulls from both US and Indian pools.
    """
    pool = TRENDING_POOL_US + TRENDING_POOL_IN
    results = []
    for sym, name in pool:
        try:
            hist = yf.download(sym, period="2d", progress=False, auto_adjust=True)
            if hist.empty or len(hist) < 2:
                continue
            close = hist["Close"].dropna()
            cur   = float(close.iloc[-1])
            prev  = float(close.iloc[-2])
            chg   = cur - prev
            pct   = (chg / prev * 100) if prev else 0.0
            results.append({
                "ticker" : sym.replace(".NS","").replace(".BO",""),
                "name"   : name,
                "price"  : cur,
                "change" : chg,
                "pct"    : pct,
            })
        except Exception:
            pass
    results.sort(key=lambda x: x["pct"], reverse=True)
    gainers = results[:5]
    losers  = sorted(results, key=lambda x: x["pct"])[:5]
    return gainers, losers


def render_trending_table(stocks: list, label: str, color: str):
    """Renders a styled trending stocks table like NSE/Moneycontrol style."""
    badge_bg  = "rgba(46,204,113,0.18)"  if color == "green" else "rgba(231,76,60,0.18)"
    badge_col = "#2ecc71"                 if color == "green" else "#e74c3c"
    arrow     = "▲" if color == "green" else "▼"

    rows_html = ""
    for i, s in enumerate(stocks):
        bg = "rgba(255,255,255,0.02)" if i % 2 == 0 else "transparent"
        price_str = f"{s['price']:,.2f}"
        rows_html += f"""
        <tr style="background:{bg};border-bottom:1px solid rgba(255,255,255,0.05);">
            <td style="padding:9px 10px;color:#f3f6ff;font-weight:600;font-size:0.82rem;">
                {s['name']}
                <span style="display:block;font-size:0.68rem;color:#a1a7c4;font-weight:400;">
                    {s['ticker']}
                </span>
            </td>
            <td style="padding:9px 10px;color:#f3f6ff;font-size:0.82rem;text-align:right;">
                {price_str}
            </td>
            <td style="padding:9px 10px;text-align:right;">
                <span style="background:{badge_bg};color:{badge_col};font-weight:700;
                             font-size:0.8rem;padding:3px 9px;border-radius:99px;">
                    {arrow}{abs(s['pct']):.1f}%
                </span>
            </td>
        </tr>"""

    table_html = f"""
    <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.07);
                border-radius:16px;overflow:hidden;margin-bottom:1rem;">
        <div style="padding:12px 14px 8px;border-bottom:1px solid rgba(255,255,255,0.06);
                    display:flex;align-items:center;justify-content:space-between;">
            <span style="background:{badge_bg};color:{badge_col};font-weight:700;
                         font-size:0.78rem;padding:4px 12px;border-radius:6px;
                         letter-spacing:0.04em;">{label}</span>
            <span style="font-size:0.72rem;color:#a1a7c4;">Live · Today</span>
        </div>
        <table style="width:100%;border-collapse:collapse;font-family:'Space Grotesk',sans-serif;">
            <thead>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.07);">
                    <th style="padding:7px 10px;color:#a1a7c4;font-size:0.72rem;
                               font-weight:600;text-align:left;text-transform:uppercase;
                               letter-spacing:0.05em;">Company</th>
                    <th style="padding:7px 10px;color:#a1a7c4;font-size:0.72rem;
                               font-weight:600;text-align:right;text-transform:uppercase;
                               letter-spacing:0.05em;">Price</th>
                    <th style="padding:7px 10px;color:#a1a7c4;font-size:0.72rem;
                               font-weight:600;text-align:right;text-transform:uppercase;
                               letter-spacing:0.05em;">% Chg</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
    </div>"""
    st.markdown(table_html, unsafe_allow_html=True)


# =======================================================================================
# MARKET STATISTICS  — NSE/BSE style dashboard
# =======================================================================================

@st.cache_data(ttl=300)
def fetch_market_statistics():
    """
    Computes market-wide statistics from a pool of NSE stocks:
    - Stocks traded, Advances, Declines, Unchanged
    - 52-week highs & lows
    - Upper & Lower circuit hitters
    - Approximate Market Cap
    Returns a dict.
    """
    STAT_POOL = [
        "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
        "SBIN.NS","WIPRO.NS","BAJFINANCE.NS","TATAMOTORS.NS","MARUTI.NS",
        "ADANIENT.NS","SUNPHARMA.NS","LT.NS","ITC.NS","HINDUNILVR.NS",
        "AAPL","MSFT","NVDA","GOOGL","AMZN","TSLA","META","JPM","V","MA",
        "NFLX","AMD","ADBE","CRM","PYPL","UBER","BAC","GS","WMT","PFE",
        "NTPC.NS","POWERGRID.NS","COALINDIA.NS","ONGC.NS","BPCL.NS",
        "GRASIM.NS","TITAN.NS","ULTRACEMCO.NS","ASIANPAINT.NS","NESTLEIND.NS",
    ]
    advances   = 0
    declines   = 0
    unchanged  = 0
    high_52w   = 0
    low_52w    = 0
    upper_ckt  = 0
    lower_ckt  = 0
    total_mcap = 0.0

    for sym in STAT_POOL:
        try:
            tk   = yf.Ticker(sym)
            hist = yf.download(sym, period="2d", progress=False, auto_adjust=True)
            if hist.empty or len(hist) < 2:
                continue
            close   = hist["Close"].dropna()
            cur     = float(close.iloc[-1])
            prev    = float(close.iloc[-2])
            chg_pct = (cur - prev) / prev * 100 if prev else 0

            if   chg_pct >  0.05: advances  += 1
            elif chg_pct < -0.05: declines  += 1
            else:                 unchanged += 1

            # 52-week high/low from fast_info
            info = tk.fast_info
            wk52h = getattr(info, "fifty_two_week_high", None)
            wk52l = getattr(info, "fifty_two_week_low",  None)
            if wk52h and cur >= wk52h * 0.995: high_52w += 1
            if wk52l and cur <= wk52l * 1.005: low_52w  += 1

            # Circuit: >9% move flags as circuit hitter
            if chg_pct >=  9: upper_ckt += 1
            if chg_pct <= -9: lower_ckt += 1

            # Market cap
            mcap = getattr(info, "market_cap", None)
            if mcap:
                total_mcap += float(mcap)

        except Exception:
            pass

    traded    = advances + declines + unchanged
    mcap_lakh = total_mcap / 1e7   # convert to Lakh Crores (approx)

    # Timestamp IST
    from datetime import timezone, timedelta
    ist = timezone(timedelta(hours=5, minutes=30))
    ts  = datetime.now(ist).strftime("%d-%b-%Y %H:%M IST")

    return {
        "traded"    : traded,
        "advances"  : advances,
        "declines"  : declines,
        "unchanged" : unchanged,
        "high_52w"  : high_52w,
        "low_52w"   : low_52w,
        "upper_ckt" : upper_ckt,
        "lower_ckt" : lower_ckt,
        "mcap_lakh" : mcap_lakh,
        "timestamp" : ts,
    }


def render_market_statistics(stats: dict):
    """Renders NSE/BSE-style Market Statistics dashboard."""
    ts = stats.get("timestamp", "")

    # ── Header row ──────────────────────────────────────────────────────────
    st.markdown(
        f"""<div style="display:flex;align-items:center;justify-content:space-between;
                        margin-bottom:14px;">
            <span style="font-size:1.05rem;font-weight:700;color:#f3f6ff;">
                📊 Market Statistics
            </span>
            <span style="font-size:0.75rem;color:#a1a7c4;">As on {ts}</span>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Row 1: 4 metric cards using st.columns ──────────────────────────────────
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    def _mcard(col, label, val, color, border):
        col.markdown(
            f"""<div style="background:rgba(255,255,255,0.03);
                            border:1px solid rgba(255,255,255,0.07);
                            border-top:4px solid {border};
                            border-radius:14px;padding:14px 16px;">
                <div style="font-size:0.72rem;color:#a1a7c4;margin-bottom:6px;">{label}</div>
                <div style="font-size:1.5rem;font-weight:700;color:{color};">{val}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    _mcard(r1c1, "Stock Traded", f"{stats['traded']:,}",    "#f3f6ff", "#6C63FF")
    _mcard(r1c2, "Advances",     f"{stats['advances']:,}",  "#2ecc71", "#2ecc71")
    _mcard(r1c3, "Declines",     f"{stats['declines']:,}",  "#e74c3c", "#e74c3c")
    _mcard(r1c4, "Unchanged",    f"{stats['unchanged']:,}", "#F7971E", "#F7971E")

    st.markdown("<div style='margin:10px 0;'></div>", unsafe_allow_html=True)

    # ── Row 2: 52-week + Circuit cards using st.columns ──────────────────────
    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    def _mcard2(col, top_label, label, val, color, arrow=""):
        arrow_span = f"<span style='color:{color};'>{arrow}</span> " if arrow else ""
        col.markdown(
            f"""<div style="background:rgba(255,255,255,0.03);
                            border:1px solid rgba(255,255,255,0.07);
                            border-radius:14px;padding:12px 14px;text-align:center;">
                <div style="font-size:0.65rem;color:#a1a7c4;line-height:1.4;margin-bottom:6px;">
                    {top_label}<br><b style="color:#f3f6ff;">{label}</b>
                </div>
                <div style="font-size:1.35rem;font-weight:700;color:{color};">
                    {arrow_span}{val}
                </div>
            </div>""",
            unsafe_allow_html=True,
        )
    _mcard2(r2c1, "No. of Stocks at", "52 Week High", stats["high_52w"], "#2ecc71", "▲")
    _mcard2(r2c2, "No. of Stocks at", "52 Week Low",  stats["low_52w"],  "#e74c3c", "▼")
    _mcard2(r2c3, "No. of Stocks in", "Upper Circuit",stats["upper_ckt"],"#2ecc71")
    _mcard2(r2c4, "No. of Stocks in", "Lower Circuit",stats["lower_ckt"],"#e74c3c")

    st.markdown("<div style='margin:10px 0;'></div>", unsafe_allow_html=True)

    # ── Row 3: Market Cap card ────────────────────────────────────────────────
    mcap_str = f"$ {stats['mcap_lakh']:,.2f} Tn" if stats["mcap_lakh"] > 0 else "N/A"
    st.markdown(
        f"""<div style="background:rgba(255,255,255,0.03);
                        border:1px solid rgba(255,255,255,0.07);
                        border-radius:14px;padding:14px 20px;">
            <div style="font-size:0.75rem;color:#6C63FF;font-weight:600;
                        margin-bottom:4px;">Market Capitalization (Sample Pool)</div>
            <div style="font-size:1.4rem;font-weight:700;color:#f3f6ff;">{mcap_str}</div>
            <div style="font-size:0.7rem;color:#a1a7c4;margin-top:2px;">
                Based on tracked stocks · {ts}
            </div>
        </div>""",
        unsafe_allow_html=True,
    )


# =======================================================================================
# MARKET TURNOVER
# =======================================================================================

@st.cache_data(ttl=300)
def fetch_market_turnover():
    """
    Computes approximate market turnover by segment using yfinance volume * price data.
    Returns a list of row dicts matching the NSE turnover table format.
    """
    from datetime import timezone, timedelta

    # Representative tickers per segment
    SEGMENTS = {
        "Equity"                  : ["RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
                                     "SBIN.NS","WIPRO.NS","TATAMOTORS.NS","MARUTI.NS","ITC.NS",
                                     "AAPL","MSFT","NVDA","GOOGL","AMZN","TSLA","META","JPM","V","MA"],
        "Equity Derivatives"      : ["NIFTY=F","^NSEI"],
        "Currency Derivatives"    : ["EURUSD=X","GBPUSD=X","USDINR=X","JPYUSD=X"],
        "Commodity Derivatives"   : ["GC=F","SI=F","CL=F","NG=F","HG=F"],
        "Debt"                    : ["^TNX","^TYX","^IRX"],
        "Mutual Fund"             : ["HDFC.NS","ICICIGI.NS"],
    }

    ist      = timezone(timedelta(hours=5, minutes=30))
    now_ist  = datetime.now(ist)
    rows     = []
    tot_vol  = 0.0
    tot_val  = 0.0
    tot_oi   = 0.0

    for segment, tickers in SEGMENTS.items():
        seg_vol = 0.0
        seg_val = 0.0
        updated = now_ist.strftime("%H:%M")

        for sym in tickers:
            try:
                hist = yf.download(sym, period="1d", interval="1d",
                                   progress=False, auto_adjust=True)
                if hist.empty:
                    continue
                vol   = float(hist["Volume"].iloc[-1])  if "Volume" in hist.columns else 0
                close = float(hist["Close"].iloc[-1])   if "Close"  in hist.columns else 0
                seg_vol += vol
                seg_val += vol * close
            except Exception:
                pass

        # Scale to Crores (1 Cr = 10M)
        vol_cr = seg_vol / 1e7
        val_cr = seg_val / 1e7

        # Open interest only meaningful for derivatives
        if "Derivatives" in segment:
            oi_cr = vol_cr * 0.25   # approx 25% OI ratio
        else:
            oi_cr = None

        tot_vol += vol_cr
        tot_val += val_cr
        if oi_cr:
            tot_oi += oi_cr

        rows.append({
            "product"  : segment,
            "volume"   : f"{vol_cr:,.2f} Cr" if vol_cr >= 1 else f"{vol_cr*100:.2f} L",
            "value"    : f"{val_cr:,.2f}",
            "oi"       : f"{oi_cr:,.2f} Cr" if oi_cr else "—",
            "updated"  : updated,
        })

    # Total row
    rows.append({
        "product" : "Total",
        "volume"  : f"{tot_vol:,.2f} Cr",
        "value"   : f"{tot_val:,.2f}",
        "oi"      : f"{tot_oi:,.2f} Cr",
        "updated" : "",
        "is_total": True,
    })

    ist_ts = now_ist.strftime("%d-%b-%Y")
    return rows, ist_ts


def render_market_turnover(rows: list, date_str: str):
    """Renders NSE-style Market Turnover table using st.dataframe — no raw HTML."""

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        f"""<div style="display:flex;align-items:center;justify-content:space-between;
                        margin:1.2rem 0 10px;">
            <span style="font-size:1.05rem;font-weight:700;color:#f3f6ff;">
                💹 Market Turnover
            </span>
            <span style="font-size:0.78rem;color:#a1a7c4;">As on {date_str}</span>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Build DataFrame and style it ─────────────────────────────────────────
    import pandas as pd

    table_rows = []
    for r in rows:
        table_rows.append({
            "Products"              : r["product"],
            "Volume (Shares/Cont.)" : r["volume"],
            "Value (₹ Crores)"      : r["value"],
            "Open Interest"         : r["oi"],
            "Updated At"            : r["updated"],
        })

    df_turn = pd.DataFrame(table_rows)

    def style_row(row):
        if row["Products"] == "Total":
            return ["background-color:#1c1a4a;color:#f3f6ff;font-weight:700;"] * len(row)
        return [""] * len(row)

    styled = (
        df_turn.style
        .apply(style_row, axis=1)
        .set_table_styles([
            {"selector": "thead th",
             "props": [
                 ("background", "linear-gradient(90deg,#3d2fa0,#6C63FF)"),
                 ("color", "#ffffff"),
                 ("font-size", "0.75rem"),
                 ("font-weight", "600"),
                 ("text-align", "center"),
                 ("padding", "10px 12px"),
             ]},
            {"selector": "tbody td",
             "props": [
                 ("font-size", "0.8rem"),
                 ("padding", "9px 12px"),
                 ("border-bottom", "1px solid rgba(255,255,255,0.05)"),
                 ("color", "#f3f6ff"),
             ]},
            {"selector": "tbody tr:nth-child(even)",
             "props": [("background-color", "rgba(255,255,255,0.02)")]},
        ])
        .hide(axis="index")
    )

    st.dataframe(
        df_turn,
        use_container_width=True,
        hide_index=True,
        height=300,
    )


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
def fetch_top_performers(n: int = 10):
    """
    Returns ALL results from TOP_PERFORMERS_POOL sorted by daily % change desc.
    Caller slices for gainers (top n) or losers (bottom n).
    """
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
            display = sym.replace(".NS","").replace(".BO","")
            results.append((display, pct, pct >= 0))
        except Exception:
            pass
    results.sort(key=lambda x: x[1], reverse=True)
    return results   # return all — caller decides how many


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


def render_top_performers(performers: list, mode: str = "gain"):
    """
    Renders ranked cards in the sidebar.
    mode = 'gain' → green arrows | mode = 'loss' → red arrows
    """
    if not performers:
        st.sidebar.caption("Data unavailable.")
        return
    medals = ["🥇","🥈","🥉","4️⃣","5️⃣"]
    cards  = ""
    for i, (name, pct, up) in enumerate(performers):
        color = "#2ecc71" if mode == "gain" else "#e74c3c"
        arrow = "▲"       if mode == "gain" else "▼"
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
    f"""
    <div style="display:flex;align-items:center;gap:10px;padding:4px 0 10px;">
        <img src="{_LOGO_SRC}" width="36" height="36"
             style="border-radius:10px;padding:3px;"/>
        <span style="font-size:1.1rem;font-weight:700;color:#f3f6ff;">Stock Vision</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Sliding market ticker at the very top ────────────────────────────────────
_tape = fetch_ticker_tape()
render_sidebar_ticker(_tape)

# ── Top Gainers + Top Losers in sidebar ──────────────────────────────────────
_all_performers = fetch_top_performers(10)   # fetch 10 — top 5 gainers + bottom 5 losers
_sb_gainers = _all_performers[:5]
_sb_losers  = sorted(_all_performers, key=lambda x: x[1])[:5]  # sort asc by pct

# Gainers
st.sidebar.markdown(
    "<div style='font-size:0.75rem;font-weight:600;color:#2ecc71;"
    "text-transform:uppercase;letter-spacing:0.06em;"
    "margin:6px 0 5px;'>▲ Top Gainers</div>",
    unsafe_allow_html=True,
)
render_top_performers(_sb_gainers, mode="gain")

st.sidebar.markdown("<div style='margin:6px 0 0;'></div>", unsafe_allow_html=True)

# Losers
st.sidebar.markdown(
    "<div style='font-size:0.75rem;font-weight:600;color:#e74c3c;"
    "text-transform:uppercase;letter-spacing:0.06em;"
    "margin:4px 0 5px;'>▼ Top Losers</div>",
    unsafe_allow_html=True,
)
render_top_performers(_sb_losers, mode="loss")

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

    # ── RIGHT COLUMN: date pickers + metrics + table + market stats ─────────────────────
    with chart_col:
        date_cols  = st.columns(2)
        start_date = date_cols[0].date_input("Start Date", date.today() - timedelta(days=365))
        end_date   = date_cols[1].date_input("End Date",   date.today())

        if start_date > end_date:
            st.error("End date cannot be before start date.")
        else:
            load_placeholder = st.empty()
            with load_placeholder.container():
                show_loading_bar(f"Loading {ticker} price data…")
                show_skeleton_rows(4)
                data, err = get_stock_data(ticker, start_date, end_date)
            load_placeholder.empty()

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
                st.dataframe(data, height=360, use_container_width=True)

        # ── MARKET STATISTICS — right side, below stock table ────────────────────────
        st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
        mstat_ph = st.empty()
        with mstat_ph:
            show_loading_bar("Loading market statistics…")
        _mstats = fetch_market_statistics()
        mstat_ph.empty()
        render_market_statistics(_mstats)

        # ── MARKET TURNOVER — below Market Statistics ─────────────────────────────
        turn_ph = st.empty()
        with turn_ph:
            show_loading_bar("Loading market turnover…")
        _turnover_rows, _turn_date = fetch_market_turnover()
        turn_ph.empty()
        render_market_turnover(_turnover_rows, _turn_date)


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

        # Detect Alpha Vantage rate-limit / premium-only responses
        _av_limit_msg = news_data.get("Note", "") or news_data.get("Information", "")
        _av_rate_limited = bool(_av_limit_msg)

        if _av_rate_limited:
            st.warning(
                "📰 Alpha Vantage free-tier daily limit reached (25 requests/day). "
                "Showing general market headlines from RSS feeds instead. "
                "Upgrade at alphavantage.co/premium for unlimited access."
            )
            # ── Fallback: render RSS-based market headlines ──────────────────
            rss_headlines = fetch_market_headlines(limit=12)
            render_headline_cards(rss_headlines)
            if not rss_headlines:
                st.info("No RSS headlines available right now either. Try again shortly.")
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