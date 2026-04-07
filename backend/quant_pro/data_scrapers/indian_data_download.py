"""
Download Indian market data for MAML pre-training.

Uses yfinance to download daily OHLCV for top Nifty 50 stocks from NSE India.
This data is used as the source-domain for First-Order MAML (FOMAML) regime
detection, which is then fast-adapted to NEPSE.

Usage:
    python3 -m backend.quant_pro.data_scrapers.indian_data_download
    python3 -m backend.quant_pro.data_scrapers.indian_data_download --years 10 --output data/indian_markets

Output:
    data/indian_markets/<TICKER>.csv  — One CSV per stock
    data/indian_markets/_index_nifty50.csv  — Nifty 50 index
    data/indian_markets/_metadata.json  — Download metadata
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from backend.quant_pro.paths import get_project_root

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = get_project_root(__file__)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "indian_markets"

# Top Nifty 50 stocks — diversified across sectors
# These mirror NEPSE's sector structure: banks, energy, insurance, manufacturing
NIFTY50_TICKERS = [
    # Banks (similar to NEPSE banking sector)
    "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS",
    "INDUSINDBK.NS", "BANKBARODA.NS", "PNB.NS",
    # IT/Tech
    "TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS",
    # Energy (similar to NEPSE hydropower)
    "RELIANCE.NS", "NTPC.NS", "POWERGRID.NS", "ONGC.NS", "ADANIGREEN.NS",
    # FMCG
    "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS",
    # Financials (similar to NEPSE insurance/microfinance)
    "BAJFINANCE.NS", "BAJAJFINSV.NS", "HDFCLIFE.NS", "SBILIFE.NS",
    # Pharma
    "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS",
    # Auto
    "MARUTI.NS", "TATAMOTORS.NS", "M&M.NS",
    # Metals & Mining
    "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS",
    # Infrastructure
    "LT.NS", "ULTRACEMCO.NS", "GRASIM.NS",
    # Others
    "TITAN.NS", "ASIANPAINT.NS", "BHARTIARTL.NS",
    "ADANIENT.NS", "ADANIPORTS.NS", "COALINDIA.NS",
    "DIVISLAB.NS", "EICHERMOT.NS", "HEROMOTOCO.NS",
    "APOLLOHOSP.NS", "TATACONSUM.NS", "BAJAJ-AUTO.NS",
    "LTIM.NS", "SHRIRAMFIN.NS",
]

# Nifty 50 index
NIFTY_INDEX = "^NSEI"

# Sector mapping for Indian stocks (parallels NEPSE sectors)
INDIAN_SECTOR_MAP = {
    "HDFCBANK.NS": "banking", "ICICIBANK.NS": "banking", "SBIN.NS": "banking",
    "KOTAKBANK.NS": "banking", "AXISBANK.NS": "banking", "INDUSINDBK.NS": "banking",
    "BANKBARODA.NS": "banking", "PNB.NS": "banking",
    "TCS.NS": "tech", "INFY.NS": "tech", "WIPRO.NS": "tech",
    "HCLTECH.NS": "tech", "TECHM.NS": "tech",
    "RELIANCE.NS": "energy", "NTPC.NS": "energy", "POWERGRID.NS": "energy",
    "ONGC.NS": "energy", "ADANIGREEN.NS": "energy",
    "HINDUNILVR.NS": "fmcg", "ITC.NS": "fmcg", "NESTLEIND.NS": "fmcg",
    "BRITANNIA.NS": "fmcg",
    "BAJFINANCE.NS": "finance", "BAJAJFINSV.NS": "finance",
    "HDFCLIFE.NS": "insurance", "SBILIFE.NS": "insurance",
    "SUNPHARMA.NS": "pharma", "DRREDDY.NS": "pharma", "CIPLA.NS": "pharma",
    "MARUTI.NS": "auto", "TATAMOTORS.NS": "auto", "M&M.NS": "auto",
    "TATASTEEL.NS": "metals", "JSWSTEEL.NS": "metals", "HINDALCO.NS": "metals",
    "LT.NS": "infra", "ULTRACEMCO.NS": "infra", "GRASIM.NS": "infra",
}


def download_indian_data(
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    years: int = 5,
    tickers: Optional[List[str]] = None,
    include_index: bool = True,
) -> Dict:
    """
    Download OHLCV data for Nifty 50 stocks using yfinance.

    Args:
        output_dir: Directory to save CSV files
        years: Number of years of history to download
        tickers: Optional custom ticker list (default: NIFTY50_TICKERS)
        include_index: Whether to download Nifty 50 index data

    Returns:
        Dict with download statistics: {successful, failed, total_rows, tickers}
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed. Run: pip install yfinance")
        return {"successful": 0, "failed": 0, "total_rows": 0, "tickers": []}

    if tickers is None:
        tickers = NIFTY50_TICKERS

    # Create output directory
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    logger.info(f"Downloading {len(tickers)} tickers, {start_str} to {end_str}")
    logger.info(f"Output directory: {out_path}")

    successful = []
    failed = []
    total_rows = 0

    # Download index first
    if include_index:
        logger.info(f"Downloading Nifty 50 index ({NIFTY_INDEX})...")
        try:
            idx_data = yf.download(NIFTY_INDEX, start=start_str, end=end_str, progress=False)
            if idx_data is not None and len(idx_data) > 0:
                # Handle multi-level columns from yfinance
                if isinstance(idx_data.columns, __import__('pandas').MultiIndex):
                    idx_data.columns = idx_data.columns.get_level_values(0)
                idx_path = out_path / "_index_nifty50.csv"
                idx_data.to_csv(idx_path)
                logger.info(f"  Nifty 50 index: {len(idx_data)} rows saved")
                total_rows += len(idx_data)
            else:
                logger.warning("  Nifty 50 index: no data returned")
        except Exception as e:
            logger.warning(f"  Nifty 50 index download failed: {e}")

    # Download individual stocks
    for i, ticker in enumerate(tickers):
        logger.info(f"  [{i+1}/{len(tickers)}] {ticker}...")
        try:
            data = yf.download(ticker, start=start_str, end=end_str, progress=False)
            if data is not None and len(data) > 0:
                # Handle multi-level columns
                if hasattr(data.columns, 'nlevels') and data.columns.nlevels > 1:
                    data.columns = data.columns.get_level_values(0)
                # Clean ticker name for filename
                clean_name = ticker.replace(".NS", "").replace("&", "_AND_")
                csv_path = out_path / f"{clean_name}.csv"
                data.to_csv(csv_path)
                successful.append(ticker)
                total_rows += len(data)
                logger.info(f"    {len(data)} rows saved")
            else:
                failed.append(ticker)
                logger.warning(f"    No data returned")
        except Exception as e:
            failed.append(ticker)
            logger.warning(f"    Failed: {e}")

        # Brief pause to be nice to Yahoo's servers
        time.sleep(0.3)

    # Save metadata
    metadata = {
        "download_date": datetime.now(timezone.utc).isoformat(),
        "start_date": start_str,
        "end_date": end_str,
        "years": years,
        "successful": successful,
        "failed": failed,
        "total_rows": total_rows,
        "sector_map": INDIAN_SECTOR_MAP,
    }
    meta_path = out_path / "_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"\nDownload complete:")
    logger.info(f"  Successful: {len(successful)}/{len(tickers)}")
    logger.info(f"  Failed: {len(failed)}")
    logger.info(f"  Total rows: {total_rows:,}")
    logger.info(f"  Output: {out_path}")

    if failed:
        logger.warning(f"  Failed tickers: {failed}")

    return {
        "successful": len(successful),
        "failed": len(failed),
        "total_rows": total_rows,
        "tickers": successful,
    }


def load_indian_returns(
    data_dir: str = str(DEFAULT_OUTPUT_DIR),
    min_rows: int = 500,
) -> Dict:
    """
    Load downloaded Indian market data and compute returns.

    Args:
        data_dir: Directory containing CSV files
        min_rows: Minimum rows required to include a stock

    Returns:
        Dict mapping ticker -> DataFrame with columns [Close, Return]
    """
    import pandas as pd

    data_path = Path(data_dir)
    if not data_path.exists():
        logger.warning(f"Indian data directory not found: {data_path}")
        return {}

    results = {}
    csv_files = list(data_path.glob("*.csv"))
    csv_files = [f for f in csv_files if not f.name.startswith("_")]

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            if len(df) < min_rows:
                continue
            if "Close" in df.columns:
                df["Return"] = df["Close"].pct_change()
                results[csv_file.stem] = df
        except Exception as e:
            logger.warning(f"Failed to load {csv_file.name}: {e}")

    logger.info(f"Loaded {len(results)} Indian stocks with >= {min_rows} rows")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Download Indian market data for MAML pre-training"
    )
    parser.add_argument(
        "--output", type=str, default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--years", type=int, default=5,
        help="Years of history to download (default: 5)"
    )
    parser.add_argument(
        "--top-n", type=int, default=None,
        help="Download only top N tickers (default: all 50)"
    )
    parser.add_argument(
        "--no-index", action="store_true",
        help="Skip downloading Nifty 50 index"
    )

    args = parser.parse_args()

    tickers = NIFTY50_TICKERS
    if args.top_n:
        tickers = tickers[:args.top_n]

    stats = download_indian_data(
        output_dir=args.output,
        years=args.years,
        tickers=tickers,
        include_index=not args.no_index,
    )

    if stats["successful"] == 0:
        print("\nERROR: No data downloaded!")
        return 1

    print(f"\nSuccess: {stats['successful']} stocks, {stats['total_rows']:,} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
