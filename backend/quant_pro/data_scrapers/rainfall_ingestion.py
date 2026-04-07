"""
Rainfall data ingestion from Open-Meteo Archive API.

Fetches daily precipitation and temperature for Nepal's 3 major
hydropower basins (Koshi, Gandaki, Karnali) plus a national average.

Usage:
    python3 -m backend.quant_pro.data_scrapers.rainfall_ingestion
    python3 -m backend.quant_pro.data_scrapers.rainfall_ingestion --start 2020-01-01 --end 2026-02-12

Source: Open-Meteo Archive API (free, no API key, no rate limits)
    https://archive-api.open-meteo.com/v1/archive
"""

import argparse
import json
import logging
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from backend.quant_pro.paths import get_data_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Nepal's 3 major hydropower basins — representative center points
HYDRO_BASINS = {
    "koshi": {"lat": 27.7, "lon": 87.2, "desc": "Koshi basin (Upper Tamakoshi, NHPC, AKPL, AHPC)"},
    "gandaki": {"lat": 28.2, "lon": 84.5, "desc": "Gandaki basin (Kaligandaki, BARUN, Chilime, API)"},
    "karnali": {"lat": 29.2, "lon": 81.7, "desc": "Karnali basin (Sanjen, UPCL)"},
}

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

DEFAULT_DB_PATH = get_data_dir(__file__) / "nepse_market_data.db"


def fetch_basin_weather(
    basin_name: str,
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    retry_count: int = 3,
    retry_delay: float = 5.0,
) -> list[dict]:
    """Fetch daily precipitation + temperature from Open-Meteo for a basin.

    Returns list of dicts with keys: date, basin, rainfall_mm, temperature_c, source.
    """
    params = (
        f"latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&daily=precipitation_sum,temperature_2m_mean"
        f"&timezone=Asia%2FKathmandu"
    )
    url = f"{OPEN_METEO_ARCHIVE_URL}?{params}"

    for attempt in range(1, retry_count + 1):
        try:
            logger.info(f"  Fetching {basin_name} (attempt {attempt}): {url}")
            req = Request(url, headers={"User-Agent": "NepseQuant/1.0"})
            with urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            break
        except (URLError, HTTPError, TimeoutError) as e:
            logger.warning(f"  Attempt {attempt} failed for {basin_name}: {e}")
            if attempt < retry_count:
                time.sleep(retry_delay * attempt)
            else:
                logger.error(f"  All {retry_count} attempts failed for {basin_name}. Skipping.")
                return []

    daily = data.get("daily", {})
    dates = daily.get("time", [])
    precip = daily.get("precipitation_sum", [])
    temp = daily.get("temperature_2m_mean", [])

    if not dates:
        logger.warning(f"  No data returned for {basin_name}")
        return []

    now_utc = datetime.now(timezone.utc).isoformat()
    rows = []
    for i, d in enumerate(dates):
        rows.append({
            "date": d,
            "basin": basin_name,
            "rainfall_mm": precip[i] if i < len(precip) else None,
            "temperature_c": temp[i] if i < len(temp) else None,
            "source": "open_meteo",
            "scraped_at_utc": now_utc,
        })

    logger.info(f"  {basin_name}: {len(rows)} days fetched ({dates[0]} to {dates[-1]})")
    return rows


def compute_national_average(all_basin_rows: list[dict]) -> list[dict]:
    """Compute national average across all basins for each date.

    Returns rows with basin='national_avg'.
    """
    from collections import defaultdict

    by_date = defaultdict(lambda: {"rainfall": [], "temperature": []})
    for row in all_basin_rows:
        d = row["date"]
        if row["rainfall_mm"] is not None:
            by_date[d]["rainfall"].append(row["rainfall_mm"])
        if row["temperature_c"] is not None:
            by_date[d]["temperature"].append(row["temperature_c"])

    now_utc = datetime.now(timezone.utc).isoformat()
    avg_rows = []
    for d, vals in sorted(by_date.items()):
        r_list = vals["rainfall"]
        t_list = vals["temperature"]
        avg_rows.append({
            "date": d,
            "basin": "national_avg",
            "rainfall_mm": sum(r_list) / len(r_list) if r_list else None,
            "temperature_c": sum(t_list) / len(t_list) if t_list else None,
            "source": "open_meteo",
            "scraped_at_utc": now_utc,
        })

    logger.info(f"  national_avg: {len(avg_rows)} days computed")
    return avg_rows


def store_weather_data(rows: list[dict], db_path: str) -> int:
    """Insert or replace weather data into SQLite.

    Returns number of rows upserted.
    """
    if not rows:
        return 0

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    inserted = 0

    for row in rows:
        try:
            c.execute(
                """INSERT OR REPLACE INTO weather_data
                   (date, basin, rainfall_mm, temperature_c, source, scraped_at_utc)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    row["date"],
                    row["basin"],
                    row["rainfall_mm"],
                    row["temperature_c"],
                    row["source"],
                    row["scraped_at_utc"],
                ),
            )
            inserted += 1
        except sqlite3.Error as e:
            logger.warning(f"  DB error for {row['date']}/{row['basin']}: {e}")

    conn.commit()
    conn.close()
    return inserted


def run_ingestion(
    start_date: str = "2020-01-01",
    end_date: str | None = None,
    db_path: str | None = None,
) -> dict:
    """Run full rainfall ingestion pipeline.

    Args:
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format (defaults to yesterday).
        db_path: Path to SQLite database.

    Returns:
        Summary dict with counts per basin.
    """
    if end_date is None:
        from datetime import date as _date
        end_date = _date.today().isoformat()

    if db_path is None:
        db_path = str(DEFAULT_DB_PATH)

    logger.info(f"Starting rainfall ingestion: {start_date} to {end_date}")
    logger.info(f"Database: {db_path}")

    all_basin_rows = []
    summary = {}

    for basin_name, info in HYDRO_BASINS.items():
        rows = fetch_basin_weather(
            basin_name=basin_name,
            lat=info["lat"],
            lon=info["lon"],
            start_date=start_date,
            end_date=end_date,
        )
        all_basin_rows.extend(rows)
        count = store_weather_data(rows, db_path)
        summary[basin_name] = count
        # Small delay between basin requests to be polite
        time.sleep(1.0)

    # Compute and store national average
    avg_rows = compute_national_average(all_basin_rows)
    count = store_weather_data(avg_rows, db_path)
    summary["national_avg"] = count

    # Summary stats
    total = sum(summary.values())
    logger.info(f"Ingestion complete: {total} total rows stored")
    for basin, cnt in summary.items():
        logger.info(f"  {basin}: {cnt} rows")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Ingest rainfall data from Open-Meteo into nepse_market_data.db"
    )
    parser.add_argument(
        "--start", type=str, default="2020-01-01",
        help="Start date (YYYY-MM-DD). Default: 2020-01-01"
    )
    parser.add_argument(
        "--end", type=str, default=None,
        help="End date (YYYY-MM-DD). Default: today"
    )
    parser.add_argument(
        "--db", type=str, default=None,
        help="Path to SQLite database. Default: nepse_market_data.db in project root"
    )

    args = parser.parse_args()
    summary = run_ingestion(
        start_date=args.start,
        end_date=args.end,
        db_path=args.db,
    )

    print(f"\nDone. Summary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()


__all__ = ["run_ingestion", "fetch_basin_weather", "HYDRO_BASINS"]
