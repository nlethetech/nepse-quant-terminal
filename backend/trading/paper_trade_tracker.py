import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

try:
    import fcntl
except ImportError:
    fcntl = None
    try:
        import msvcrt
    except ImportError:
        msvcrt = None
else:
    msvcrt = None

import pandas as pd

from backend.quant_pro.data_io import get_dynamic_data
from backend.quant_pro.config import (
    TRAILING_STOP_PCT,
    HARD_STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    PORTFOLIO_DRAWDOWN_LIMIT,
    RECOMMENDED_HOLDING_DAYS,
)
from backend.quant_pro.nepse_calendar import is_trading_day
from backend.quant_pro.database import get_db_path
from backend.quant_pro.paths import ensure_dir, get_project_root, get_trading_runtime_dir, migrate_legacy_path

logger = logging.getLogger(__name__)


def _lock_file_exclusive(handle) -> None:
    if fcntl is not None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        return
    if msvcrt is not None:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        return
    raise RuntimeError("No file-lock implementation available on this platform")


def _unlock_file(handle) -> None:
    if fcntl is not None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return
    if msvcrt is not None:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return
    raise RuntimeError("No file-lock implementation available on this platform")

# --- CONFIGURATION ---
# Resolve paths at call-time so Streamlit profile switches work.
PROJECT_ROOT = get_project_root(__file__)
TRADING_RUNTIME_DIR = ensure_dir(get_trading_runtime_dir(__file__))
DEFAULT_PORTFOLIO_FILE = str(migrate_legacy_path(TRADING_RUNTIME_DIR / "paper_portfolio.csv", [PROJECT_ROOT / "paper_portfolio.csv"]))
DEFAULT_PNL_LOG_FILE = str(migrate_legacy_path(TRADING_RUNTIME_DIR / "paper_pnl_log.csv", [PROJECT_ROOT / "paper_pnl_log.csv"]))
DEFAULT_INPUT_ORDERS_FILE = str(migrate_legacy_path(TRADING_RUNTIME_DIR / "buy_orders.csv", [PROJECT_ROOT / "buy_orders.csv"]))
DEFAULT_SELL_ORDERS_FILE = str(migrate_legacy_path(TRADING_RUNTIME_DIR / "sell_orders.csv", [PROJECT_ROOT / "sell_orders.csv"]))
DEFAULT_HIGH_WATERMARK_FILE = str(
    migrate_legacy_path(TRADING_RUNTIME_DIR / "portfolio_watermark.csv", [PROJECT_ROOT / "portfolio_watermark.csv"])
)


def _resolve_paths():
    return {
        "portfolio": os.environ.get("PAPER_PORTFOLIO_FILE", DEFAULT_PORTFOLIO_FILE),
        "pnl_log": os.environ.get("PAPER_PNL_LOG_FILE", DEFAULT_PNL_LOG_FILE),
        "orders": os.environ.get("PAPER_INPUT_ORDERS_FILE", DEFAULT_INPUT_ORDERS_FILE),
        "sell_orders": os.environ.get("PAPER_SELL_ORDERS_FILE", DEFAULT_SELL_ORDERS_FILE),
        "watermark": os.environ.get("PAPER_WATERMARK_FILE", DEFAULT_HIGH_WATERMARK_FILE),
    }

# --- NEPSE FEE CALCULATOR — unified source in validation.transaction_costs ---
from validation.transaction_costs import TransactionCostModel as NepseFees  # noqa: E402


# --- RISK MANAGEMENT ---

class RiskManager:
    """
    Risk management for paper trading portfolio.

    Features:
    1. Trailing stops: Exit when price drops 10% from peak
    2. Hard stop loss: Exit when price drops 8% from entry
    3. Portfolio drawdown: Pause trading when portfolio down 15%
    4. Take profit: Optional exit at 20% gain
    """

    @staticmethod
    def load_watermarks(watermark_file: str) -> pd.DataFrame:
        """Load high watermark data for positions."""
        if os.path.exists(watermark_file):
            return pd.read_csv(watermark_file)
        return pd.DataFrame(columns=["Symbol", "High_Watermark", "Last_Updated"])

    @staticmethod
    def save_watermarks(watermarks: pd.DataFrame, watermark_file: str):
        """Save high watermark data with file locking."""
        with open(watermark_file, 'w') as f:
            _lock_file_exclusive(f)
            try:
                watermarks.to_csv(f, index=False)
            finally:
                _unlock_file(f)

    @staticmethod
    def update_watermark(watermarks: pd.DataFrame, symbol: str, current_price: float) -> pd.DataFrame:
        """Update high watermark for a symbol."""
        today = datetime.now().strftime("%Y-%m-%d")

        if symbol in watermarks["Symbol"].values:
            idx = watermarks[watermarks["Symbol"] == symbol].index[0]
            current_hw = watermarks.loc[idx, "High_Watermark"]
            if current_price > current_hw:
                watermarks.loc[idx, "High_Watermark"] = current_price
                watermarks.loc[idx, "Last_Updated"] = today
        else:
            new_row = pd.DataFrame([{
                "Symbol": symbol,
                "High_Watermark": current_price,
                "Last_Updated": today
            }])
            watermarks = pd.concat([watermarks, new_row], ignore_index=True)

        return watermarks

    @staticmethod
    def check_trailing_stop(current_price: float, high_watermark: float) -> bool:
        """Check if trailing stop is triggered."""
        if high_watermark <= 0:
            return False
        drawdown = (high_watermark - current_price) / high_watermark
        return drawdown >= TRAILING_STOP_PCT

    @staticmethod
    def check_hard_stop(current_price: float, entry_price: float) -> bool:
        """Check if hard stop loss is triggered."""
        if entry_price <= 0:
            return False
        loss = (entry_price - current_price) / entry_price
        return loss >= HARD_STOP_LOSS_PCT

    @staticmethod
    def check_take_profit(current_price: float, entry_price: float) -> bool:
        """Check if take profit level is reached."""
        if entry_price <= 0:
            return False
        gain = (current_price - entry_price) / entry_price
        return gain >= TAKE_PROFIT_PCT

    @staticmethod
    def check_portfolio_drawdown(current_value: float, peak_value: float) -> bool:
        """Check if portfolio drawdown limit is breached."""
        if peak_value <= 0:
            return False
        drawdown = (peak_value - current_value) / peak_value
        return drawdown >= PORTFOLIO_DRAWDOWN_LIMIT


def _count_trading_days_since(entry_date_str: str) -> int:
    """Count NEPSE trading days from entry to the current Nepal session date."""
    from datetime import datetime, timedelta
    from backend.quant_pro.nepse_calendar import current_nepal_datetime
    entry = datetime.strptime(entry_date_str, "%Y-%m-%d").date()
    today = current_nepal_datetime().date()
    count = 0
    current = entry + timedelta(days=1)  # Don't count entry day
    while current <= today:
        if is_trading_day(current):
            count += 1
        current += timedelta(days=1)
    return count


def check_risk_exits():
    """
    Check all positions for risk-based exits.
    Generates sell_orders.csv with positions that should be exited.
    """
    paths = _resolve_paths()
    portfolio_file = paths["portfolio"]
    watermark_file = paths["watermark"]
    sell_orders_file = paths["sell_orders"]

    if not os.path.exists(portfolio_file):
        print(f"❌ Error: {portfolio_file} not found. No positions to check.")
        return

    portfolio = pd.read_csv(portfolio_file)
    watermarks = RiskManager.load_watermarks(watermark_file)

    today = datetime.now().strftime("%Y-%m-%d")

    print(f"\n--- RISK CHECK ({today}) ---")
    print(f"{'Symbol':<10} {'Entry':<10} {'LTP':<10} {'High':<10} {'Action':<15} {'Reason'}")
    print("-" * 75)

    sell_orders = []
    total_current_value = 0.0
    total_cost_basis = 0.0

    for _, row in portfolio.iterrows():
        symbol = row['Symbol']
        qty = row['Quantity']
        entry_price = row['Buy_Price']
        cost_basis = row['Total_Cost_Basis']

        # Fetch current price
        try:
            df_data = get_dynamic_data(symbol)
            if df_data is None or df_data.empty:
                print(f"{symbol:<10} {'ERROR':<10} - Could not fetch price")
                continue
            current_price = float(df_data['Close'].iloc[-1])
        except Exception as e:
            print(f"{symbol:<10} {'ERROR':<10} - {e}")
            continue

        # Update watermark
        watermarks = RiskManager.update_watermark(watermarks, symbol, current_price)
        hw_row = watermarks[watermarks["Symbol"] == symbol]
        high_watermark = hw_row["High_Watermark"].iloc[0] if not hw_row.empty else current_price

        # Track portfolio value
        total_current_value += qty * current_price
        total_cost_basis += cost_basis

        # Count trading days held
        entry_date_str = row.get('Entry_Date', '')
        trading_days_held = _count_trading_days_since(entry_date_str) if entry_date_str else 0

        # Check exit conditions
        action = "HOLD"
        reason = ""

        if RiskManager.check_hard_stop(current_price, entry_price):
            action = "SELL"
            reason = f"HARD STOP: -{HARD_STOP_LOSS_PCT*100:.0f}% from entry"

        if action == "HOLD" and RiskManager.check_trailing_stop(current_price, high_watermark):
            action = "SELL"
            reason = f"TRAILING STOP: -{TRAILING_STOP_PCT*100:.0f}% from peak {high_watermark:.2f}"

        if action == "HOLD" and RiskManager.check_take_profit(current_price, entry_price):
            action = "SELL"
            reason = f"TAKE PROFIT: +{TAKE_PROFIT_PCT*100:.0f}% reached"

        if action == "HOLD" and trading_days_held >= RECOMMENDED_HOLDING_DAYS:
            action = "SELL"
            reason = f"HOLDING PERIOD: {trading_days_held}/{RECOMMENDED_HOLDING_DAYS} trading days"

        print(f"{symbol:<10} {entry_price:<10.2f} {current_price:<10.2f} {high_watermark:<10.2f} {action:<15} {reason}")

        if action == "SELL":
            sell_orders.append({
                "Symbol": symbol,
                "Quantity": qty,
                "Entry_Price": entry_price,
                "Current_Price": current_price,
                "High_Watermark": high_watermark,
                "Reason": reason,
                "Date": today
            })

    # Save watermarks
    RiskManager.save_watermarks(watermarks, watermark_file)

    # Check portfolio-level drawdown
    print("-" * 75)
    portfolio_return = (total_current_value - total_cost_basis) / total_cost_basis if total_cost_basis > 0 else 0
    print(f"📊 Portfolio Value: Rs. {total_current_value:,.2f}")
    print(f"💰 Cost Basis: Rs. {total_cost_basis:,.2f}")
    print(f"📈 Portfolio Return: {portfolio_return*100:.2f}%")

    # Load portfolio peak from watermark file or calculate
    portfolio_peak = total_cost_basis  # Start with cost basis as minimum peak
    peak_file = watermark_file.replace(".csv", "_portfolio_peak.txt")
    if os.path.exists(peak_file):
        with open(peak_file, 'r') as f:
            portfolio_peak = max(float(f.read().strip()), total_current_value)
    else:
        portfolio_peak = max(total_cost_basis, total_current_value)

    # Update peak if current value is higher
    if total_current_value > portfolio_peak:
        portfolio_peak = total_current_value

    with open(peak_file, 'w') as f:
        f.write(str(portfolio_peak))

    portfolio_drawdown = (portfolio_peak - total_current_value) / portfolio_peak if portfolio_peak > 0 else 0

    if RiskManager.check_portfolio_drawdown(total_current_value, portfolio_peak):
        print(f"\n🚨 PORTFOLIO DRAWDOWN ALERT: {portfolio_drawdown*100:.1f}% from peak!")
        print(f"⚠️  RECOMMENDATION: PAUSE TRADING - Drawdown exceeds {PORTFOLIO_DRAWDOWN_LIMIT*100:.0f}% limit")
        print(f"   Peak Value: Rs. {portfolio_peak:,.2f}")
        print(f"   Current Value: Rs. {total_current_value:,.2f}")
    else:
        print(f"✅ Portfolio drawdown: {portfolio_drawdown*100:.1f}% (limit: {PORTFOLIO_DRAWDOWN_LIMIT*100:.0f}%)")

    # Generate sell orders file
    if sell_orders:
        sell_df = pd.DataFrame(sell_orders)
        sell_df.to_csv(sell_orders_file, index=False)
        print(f"\n🔴 SELL SIGNALS GENERATED: {len(sell_orders)} positions")
        print(f"   Output: {sell_orders_file}")
        for order in sell_orders:
            print(f"   - {order['Symbol']}: {order['Reason']}")
    else:
        print(f"\n✅ No sell signals triggered. All positions within risk limits.")
        # Remove old sell orders file if exists
        if os.path.exists(sell_orders_file):
            os.remove(sell_orders_file)


# --- ACTIONS ---

def initialize_portfolio():
    """Reads buy orders and creates a new portfolio file without overwriting the old one."""
    paths = _resolve_paths()
    input_orders = paths["orders"]
    portfolio_file = paths["portfolio"]

    if not os.path.exists(input_orders):
        print(f"❌ Error: {input_orders} not found. Run the strategy first.")
        return

    # If a portfolio already exists, back it up with a timestamped copy.
    if os.path.exists(portfolio_file):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{portfolio_file}.backup_{ts}"
        os.rename(portfolio_file, backup_path)
        print(f"📦 Existing portfolio backed up to {backup_path}")

    print(f"📥 Loading orders from {input_orders}...")
    orders = pd.read_csv(input_orders)
    
    portfolio_records = []
    total_investment = 0.0
    total_buy_fees = 0.0

    today = datetime.now().strftime("%Y-%m-%d")

    print("\n--- EXECUTING BUY ORDERS (PAPER) ---")
    print(f"{'Symbol':<10} {'Qty':<8} {'Price':<10} {'Amount':<15} {'Fees':<10}")
    print("-" * 60)

    for _, row in orders.iterrows():
        symbol = row['Symbol']
        
        # Parse Qty and Price (Handle string formatting if present)
        try:
            qty = int(str(row['Shares']).replace(',', ''))
            # We assume the 'Value (NPR)' was based on the latest price
            # Let's fetch the real latest price to be accurate
            df_data = get_dynamic_data(symbol)
            if df_data is None or df_data.empty:
                print(f"⚠️ Warning: Could not fetch price for {symbol}. Skipping.")
                continue
            
            buy_price = float(df_data['Close'].iloc[-1])
        except Exception as e:
            print(f"Error parsing data for {symbol}: {e}")
            continue

        amount = qty * buy_price
        fees = NepseFees.calculate_total_fees(qty, buy_price, is_buy=True)
        
        total_cost = amount + fees['total']
        
        portfolio_records.append({
            "Entry_Date": today,
            "Symbol": symbol,
            "Quantity": qty,
            "Buy_Price": buy_price,
            "Buy_Amount": amount,
            "Buy_Fees": fees['total'],
            "Total_Cost_Basis": total_cost # Includes fees
        })
        
        total_investment += total_cost
        total_buy_fees += fees['total']
        
        print(f"{symbol:<10} {qty:<8} {buy_price:<10.2f} {amount:<15,.2f} {fees['total']:<10.2f}")

    # Save
    pd.DataFrame(portfolio_records).to_csv(portfolio_file, index=False)
    print("-" * 60)
    print(f"✅ Portfolio initialized in {portfolio_file}")
    print(f"💰 Total Capital Deployment: Rs. {total_investment:,.2f}")
    print(f"💸 Total Buy Fees Paid:      Rs. {total_buy_fees:,.2f}")


def update_pnl():
    """Fetches live prices and calculates Unrealized P&L (Pre-Tax)."""
    paths = _resolve_paths()
    portfolio_file = paths["portfolio"]
    pnl_log_file = paths["pnl_log"]

    if not os.path.exists(portfolio_file):
        print(f"❌ Error: {portfolio_file} not found. Run --action buy first.")
        return

    portfolio = pd.read_csv(portfolio_file)
    today = datetime.now().strftime("%Y-%m-%d")
    
    print(f"\n--- DAILY P&L UPDATE ({today}) ---")
    print(f"{'Symbol':<10} {'Buy Price':<10} {'LTP':<10} {'Chg%':<8} {'Unrealized P&L':<15} {'Est. Tax':<10}")
    print("-" * 75)

    daily_stats = []
    total_current_value = 0.0
    total_unrealized_pnl = 0.0
    total_est_tax = 0.0

    for idx, row in portfolio.iterrows():
        symbol = row['Symbol']
        qty = row['Quantity']
        buy_price = row['Buy_Price']
        cost_basis = row['Total_Cost_Basis'] # Includes Buy Fees
        
        # 1. Fetch Live Price
        try:
            df_data = get_dynamic_data(symbol)
            if df_data is None or df_data.empty:
                print(f"{symbol:<10} {'ERROR':<10}")
                continue
            ltp = float(df_data['Close'].iloc[-1])
        except (KeyError, IndexError, ValueError, TypeError) as e:
            logger.warning("Failed to fetch price for %s, using buy price: %s", symbol, e)
            ltp = buy_price
        
        # 2. Calculate Market Value & Sell Fees
        current_mkt_value = qty * ltp
        # We estimate sell fees to give a realistic "Net Liquidation" view
        est_sell_fees = NepseFees.calculate_total_fees(qty, ltp, is_buy=False)['total']
        
        # 3. Calculate Unrealized P&L (Pre-Tax)
        # Unrealized P&L = (Current Value - Est Sell Fees) - (Buy Cost Basis)
        unrealized_pnl = current_mkt_value - est_sell_fees - cost_basis
        
        # 4. Calculate Est. Tax (Shadow Calculation)
        # Tax is only calculated on the GAIN component
        taxable_gain = (current_mkt_value - est_sell_fees) - cost_basis
        est_tax = 0.0
        if taxable_gain > 0:
            est_tax = NepseFees.calculate_tax(taxable_gain, holding_period_days=0)
            
        change_pct = (ltp - buy_price) / buy_price * 100
        
        total_current_value += current_mkt_value
        total_unrealized_pnl += unrealized_pnl
        total_est_tax += est_tax
        
        print(f"{symbol:<10} {buy_price:<10.2f} {ltp:<10.2f} {change_pct:^8.2f}% {unrealized_pnl:<15.2f} {est_tax:<10.2f}")
        
        daily_stats.append({
            "Date": today,
            "Symbol": symbol,
            "LTP": ltp,
            "Unrealized_PnL": unrealized_pnl,
            "Est_Tax_Liability": est_tax
        })

    # Log to history
    log_df = pd.DataFrame(daily_stats)
    # Check if file exists/headers match, safer to append if exists, or create new
    write_header = not os.path.exists(pnl_log_file)
    log_df.to_csv(pnl_log_file, mode='a', header=write_header, index=False)

    print("-" * 75)
    print(f"📊 Portfolio Value:   Rs. {total_current_value:,.2f}")
    print(f"📈 Total Unrealized:  Rs. {total_unrealized_pnl:,.2f} (Pre-Tax)")
    print(f"🏛️ Est. Tax Liability: Rs. {total_est_tax:,.2f} (If sold today)")

def main():
    parser = argparse.ArgumentParser(description="NEPSE Paper Trading Tracker")
    parser.add_argument("--action", choices=["buy", "update", "risk"], required=True,
                        help="Use 'buy' to initialize, 'update' to track P&L, or 'risk' to check stop losses.")
    args = parser.parse_args()

    if args.action == "buy":
        initialize_portfolio()
    elif args.action == "update":
        update_pnl()
    elif args.action == "risk":
        check_risk_exits()

if __name__ == "__main__":
    main()
