from io import StringIO
from contextlib import nullcontext
from typing import List, Optional, Dict
import logging
import pandas as pd
import requests
from requests.exceptions import RequestException, Timeout, HTTPError
import time
from datetime import datetime, timedelta
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
# Ensure .database module is available for init_db, etc.
from .database import init_db, get_latest_date, save_to_db, load_from_db
from .vendor_api import _rate_limiter, fetch_latest_ltp

# Configure logging for data_io module
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Initialize DB on first import
init_db()

def _is_nepse_trading_day(day) -> bool:
    """
    Prefer deriving tradable dates from benchmark (NEPSE) history to avoid hardcoding
    calendar assumptions. Falls back to the Sun–Thu weekmask used in data_quality.
    """
    try:
        d = pd.Timestamp(day).normalize()
    except Exception:
        return False
    try:
        bench = load_from_db("NEPSE")
        if bench is not None and not bench.empty:
            idx = pd.DatetimeIndex(bench.index).normalize()
            return bool(d in idx)
    except Exception:
        pass
    # Fallback: Sunday(6) ... Thursday(3) on pandas dayofweek scale (Mon=0).
    return int(d.dayofweek) in {6, 0, 1, 2, 3}


# Mapping from sector name to Merolagani index symbol (extend as needed)
SECTOR_INDEX_SYMBOLS: Dict[str, str] = {
    "Commercial Banks": "BANKING",
    "Development Banks": "DEVELOPMENT BANK",
    "Finance": "FINANCE",
    "Hydropower": "HYDROPOWER",
    "Hotels & Tourism": "HOTELS",
    "Hotels": "HOTELS",
    "Manufacturing & Processing": "MANUFACTURING",
    "Investment": "INVESTMENT",
    "Trading": "TRADING",
    "Others": "OTHERS",
    "Microfinance": "MICROFINANCE",
    "Life Insurance": "LIFE INSURANCE",
    "Non-Life Insurance": "NON LIFE INSURANCE",
    "Mutual Fund": "MUTUAL FUND",
    "NEPSE": "NEPSE",
    "Sensitive": "SENSITIVE",
}


def _is_streamlit_active() -> bool:
    """Returns True when running inside a Streamlit session."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def _spinner_context(message: str):
    """Return Streamlit spinner context in UI mode, otherwise no-op context."""
    if _is_streamlit_active():
        import streamlit as st
        return st.spinner(message)
    return nullcontext()


def _report_error(message: str):
    """Send errors to Streamlit in UI mode, otherwise log."""
    if _is_streamlit_active():
        import streamlit as st
        st.error(message)
    else:
        logger.error(message)

# --- OLD CSV LOGIC (Preserved) ---
def load_csv(file) -> Optional[pd.DataFrame]:
    try:
        if isinstance(file, str):
            return pd.read_csv(file)
        stringio = StringIO(file.getvalue().decode("utf-8"))
        return pd.read_csv(stringio)
    except Exception as exc:
        if _is_streamlit_active():
            import streamlit as st
            st.error(f"Error loading CSV: {exc}")
        else:
            logger.error(f"Error loading CSV: {exc}")
        return None

def extract_price_matrix(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        raise ValueError("Input DataFrame is empty.")
    df = raw_df.copy()
    df.columns = [c.strip() for c in df.columns]
    
    date_candidates = [c for c in df.columns if "date" in c.lower()]
    date_col = "Date" if "Date" in df.columns else (date_candidates[0] if date_candidates else df.columns[0])
    
    df["Date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.drop_duplicates(subset=["Date"], keep="last")
    df = df.sort_values("Date").set_index("Date")
    
    price_data = pd.DataFrame(index=df.index)
    for col in df.columns:
        if col == date_col: continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notnull().sum() > 0:
            price_data[col] = numeric
            
    # [ROBUSTNESS FIX] Fill Volume NaNs with 0 before final dropna
    if 'Volume' in price_data.columns:
        price_data['Volume'] = price_data['Volume'].fillna(0)
            
    return price_data.dropna(how="all").ffill().dropna()

def load_macro_series(file) -> Optional[pd.Series]:
    if file is None: return None
    macro_df = load_csv(file)
    if macro_df is None or macro_df.empty: return None
    date_col = macro_df.columns[0]
    val_col = macro_df.columns[1]
    macro_df[date_col] = pd.to_datetime(macro_df[date_col])
    return macro_df.set_index(date_col)[val_col]

# --- NEW SMART FETCH LOGIC (Corrected for Continuity) ---

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    retry=retry_if_exception_type((RequestException, Timeout, HTTPError, ValueError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _fetch_chunk_with_retry(symbol: str, start_ts: int, end_ts: int) -> pd.DataFrame:
    """Fetches a single chunk from the API with retry/backoff."""
    _rate_limiter.wait()
    url = "https://merolagani.com/handlers/TechnicalChartHandler.ashx"
    params = {
        "type": "get_advanced_chart", "symbol": symbol, "resolution": "1D",
        "rangeStartDate": int(start_ts), "rangeEndDate": int(end_ts),
        "isAdjust": "1", "currencyCode": "NPR"
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": f"https://merolagani.com/CompanyDetail.aspx?symbol={symbol}",
        "X-Requested-With": "XMLHttpRequest"
    }
    response = requests.get(url, params=params, headers=headers, timeout=15)
    response.raise_for_status()  # Raise HTTPError for bad status codes

    data = response.json()

    # Validate schema
    required_keys = ["t", "o", "h", "l", "c", "v"]
    if not all(k in data for k in required_keys):
        logger.warning(f"{symbol}: Invalid API response schema - missing keys")
        return pd.DataFrame()

    if len(data["t"]) == 0:
        logger.debug(f"{symbol}: Empty data returned for range")
        return pd.DataFrame()

    return pd.DataFrame({
        "Date": pd.to_datetime(data["t"], unit="s"),
        "Open": data["o"], "High": data["h"], "Low": data["l"],
        "Close": data["c"], "Volume": data["v"]
    })


def fetch_chunk(symbol: str, start_ts, end_ts) -> pd.DataFrame:
    """Fetches a single chunk from the API with error handling."""
    try:
        return _fetch_chunk_with_retry(symbol, start_ts, end_ts)
    except RequestException as e:
        logger.error(f"{symbol}: Network error after retries: {e}")
        return pd.DataFrame()
    except ValueError as e:
        logger.error(f"{symbol}: JSON parsing error: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"{symbol}: Unexpected error in fetch_chunk: {type(e).__name__}: {e}")
        return pd.DataFrame()

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((RequestException, Timeout)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _fetch_live_candle_with_retry(symbol: str) -> pd.DataFrame:
    """Fetch the latest price through the shared intraday provider."""
    price = fetch_latest_ltp(symbol)
    if price is not None and price > 0:
        return pd.DataFrame([{
            "Date": pd.to_datetime(datetime.now().date()),
            "Open": price, "High": price, "Low": price, "Close": price, "Volume": 0
        }])
    return pd.DataFrame()


def fetch_live_candle(symbol: str) -> pd.DataFrame:
    """Scrapes the live price (LTP) from the main page."""
    # Never synthesize candles on non-trading days (Fri/Sat) to avoid polluting history/DB.
    if not _is_nepse_trading_day(datetime.now().date()):
        return pd.DataFrame()
    try:
        return _fetch_live_candle_with_retry(symbol)
    except RequestException as e:
        logger.warning(f"{symbol}: Failed to fetch live candle: {e}")
        return pd.DataFrame()
    except (ValueError, AttributeError) as e:
        logger.warning(f"{symbol}: Error parsing live candle: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"{symbol}: Unexpected error in fetch_live_candle: {type(e).__name__}: {e}")
        return pd.DataFrame()

def fetch_sector_history(sector_symbol: str, duration_days: int = 3650) -> pd.DataFrame:
    """
    Fetch OHLCV history for a sector index symbol (e.g., BANKING, HYDROPOWER).
    Uses the same retry logic as fetch_chunk.
    """
    end_ts = int(time.time())
    start_ts = int(time.time() - (duration_days * 24 * 60 * 60))
    try:
        return _fetch_chunk_with_retry(sector_symbol, start_ts, end_ts)
    except RequestException as e:
        logger.error(f"Sector {sector_symbol}: Network error after retries: {e}")
        return pd.DataFrame()
    except ValueError as e:
        logger.error(f"Sector {sector_symbol}: JSON parsing error: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Sector {sector_symbol}: Unexpected error: {type(e).__name__}: {e}")
        return pd.DataFrame()

def _fetch_dynamic_data(symbol: str) -> pd.DataFrame:
    """
    ROBUST DATA SYNCHRONIZATION LOGIC.
    Performs a single, maximum-range fetch for stability.
    """
    symbol = symbol.upper().strip()
    last_date = get_latest_date(symbol)
    today = datetime.now().date()
    new_data = pd.DataFrame()
    
    # --- STEP 1: Initial Full Fetch (The FIX for Fragmentation) ---
    if last_date is None:
        # Fetch up to ~10 years of history to cover multiple regimes
        start_of_time = datetime.now() - timedelta(days=365 * 10)
        
        # Use a single, large range fetch for maximum stability
        with _spinner_context("Fetching full 10-year history..."):
            new_data = fetch_chunk(symbol, start_of_time.timestamp(), time.time())
        
    # --- STEP 2: Incremental Update ---
    elif last_date < today:
        with _spinner_context(f"Fetching updates since {last_date}..."):
            start_ts = pd.Timestamp(last_date).timestamp()
            new_data = fetch_chunk(symbol, start_ts, time.time())
        
    # --- STEP 3: Live Candle (Today's Price) ---
    # Live LTP is useful for UI/inference, but should not be persisted into the historical DB
    # (it is not an official EOD bar and can fall on non-trading days).
    include_live = _is_streamlit_active()
    live_df = fetch_live_candle(symbol) if include_live else pd.DataFrame()

    if not new_data.empty:
        new_data = new_data.drop_duplicates(subset=["Date"], keep="last")
        save_to_db(new_data, symbol)
    elif live_df.empty and last_date is None:
        _report_error(f"Could not fetch any data for {symbol}.")
        return pd.DataFrame()
    
    # Return the entire history from the database (guaranteed alignment)
    df = load_from_db(symbol)

    # Overlay live candle for Streamlit UI only (never persisted).
    if include_live and live_df is not None and not live_df.empty:
        try:
            live = live_df.copy()
            live["Date"] = pd.to_datetime(live["Date"], errors="coerce")
            live = live.dropna(subset=["Date"]).set_index("Date").sort_index()
            if df is None or df.empty:
                df = live
            else:
                df_out = df.copy()
                for col in ["Open", "High", "Low", "Close", "Volume"]:
                    if col in live.columns:
                        df_out.loc[live.index, col] = live[col].iloc[-1]
                df = df_out.sort_index()
        except (KeyError, IndexError, ValueError) as e:
            logger.debug(f"{symbol}: Error overlaying live candle: {e}")

    return df


def _get_dynamic_data_cached(symbol: str) -> pd.DataFrame:
    """Cached wrapper for Streamlit sessions."""
    return _fetch_dynamic_data(symbol)


# Apply Streamlit cache decorator only when available
try:
    import streamlit as _st
    _get_dynamic_data_cached = _st.cache_data(ttl=60, show_spinner="Syncing Market Data...")(_get_dynamic_data_cached)
except Exception:
    pass


def get_dynamic_data(symbol: str, force_update: bool = False) -> pd.DataFrame:
    """
    Fetch market data for a symbol.

    - Uses Streamlit cache when running inside the app.
    - If force_update is True, clears cache and re-syncs from the vendor/database.
    - Falls back to a direct fetch when not in a Streamlit runtime (CLI jobs).
    """
    cache_available = _is_streamlit_active()

    if force_update and cache_available:
        try:
            _get_dynamic_data_cached.clear()
        except (AttributeError, RuntimeError) as e:
            # Cache clearing may fail outside Streamlit; log and proceed.
            logger.debug(f"Cache clear failed (non-critical): {e}")

    if force_update or not cache_available:
        return _fetch_dynamic_data(symbol)

    # Default path: cached fetch inside Streamlit
    try:
        return _get_dynamic_data_cached(symbol)
    except (RuntimeError, AttributeError) as e:
        # Defensive fallback if cache is unavailable or misconfigured
        logger.debug(f"Cache unavailable, falling back to direct fetch: {e}")
        return _fetch_dynamic_data(symbol)


__all__ = ["load_csv", "extract_price_matrix", "load_macro_series", "get_dynamic_data"]

# --- SECTOR INDEX ACCESSORS ---

def get_sector_index_symbol(sector_name: str) -> Optional[str]:
    return SECTOR_INDEX_SYMBOLS.get(sector_name)


def get_sector_index_series(sector_name: str, force_update: bool = False) -> Optional[pd.Series]:
    """
    Returns sector index Close series aligned by date.
    """
    sym = get_sector_index_symbol(sector_name)
    if sym is None:
        return None
    cache_key = f"SECTOR::{sym}"
    if not force_update:
        try:
            df_cached = load_from_db(cache_key)
            if df_cached is not None and not df_cached.empty:
                return df_cached["Close"].astype(float)
        except (KeyError, ValueError) as e:
            logger.debug(f"Sector {sector_name}: Cache load failed: {e}")
    df_hist = fetch_sector_history(sym)
    if df_hist is None or df_hist.empty:
        return None
    df_hist = df_hist.drop_duplicates(subset=["Date"]).sort_values("Date")
    try:
        save_to_db(df_hist, cache_key)
    except (IOError, ValueError) as e:
        logger.warning(f"Sector {sector_name}: Failed to cache sector data: {e}")
    return df_hist.set_index("Date")["Close"].astype(float)


__all__.extend(["get_sector_index_symbol", "get_sector_index_series"])
