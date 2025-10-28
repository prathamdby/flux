"""
FLUX ALGORITHMIC TRADING SYSTEM
================================
Market Data Collection Module

Connects to Binance futures WebSocket streams to collect real-time order book
and trade data. Computes high-frequency trading metrics including:
- Order Flow Imbalance (OFI) with multi-level weighting
- Volume-Synchronized Probability of Informed Trading (VPIN)
- Microprice and spread metrics
- Trade tick momentum and directional flow
- Forward returns for ML model training (5s, 10s, 30s timeframes)

Part of the Flux end-to-end algorithmic trading system.
"""

import time
import json
import os
from datetime import datetime
from collections import deque
import websocket
import threading
import numpy as np
import pandas as pd

# ===== CONFIGURATION =====

# Data Storage
CSV_FILENAME = "flux_data.csv"

# OFI Parameters
OFI_DEPTH_LEVELS = 15
OFI_HISTORY_SIZE = 500
OFI_ZSCORE_WINDOW = 200

# Return Calculation Timeframes (focused on short-term)
RETURN_TIMEFRAMES = [5, 10, 30]  # 5s, 10s, 30s - where signal is strongest

# Binance WebSocket
WS_URL = "wss://fstream.binance.com/ws/btcusdt@depth20@100ms"

# Data Collection Settings
SAVE_INTERVAL_SECONDS = 60
PRINT_INTERVAL_SECONDS = 300

# ===== ENHANCED OFI COLLECTOR =====


class EnhancedOFICollector:
    """
    Collects comprehensive OFI data with ML-focused features:
    - Microprice
    - Quote churn
    - VPIN toxicity
    - Spread percentiles
    - Tick direction
    """

    def __init__(self):
        self.ws_url = WS_URL
        self.ws = None
        self.ws_thread = None
        self.running = False

        # Orderbook state
        self.prev_bids = []
        self.prev_asks = []
        self.current_bids = []
        self.current_asks = []

        # OFI tracking
        self.ofi_history = deque(maxlen=OFI_HISTORY_SIZE)
        self.volume_history = deque(maxlen=OFI_HISTORY_SIZE)
        self.timestamp_history = deque(maxlen=OFI_HISTORY_SIZE)

        # Price tracking
        self.price_history = deque(maxlen=5000)
        self.price_timestamps = deque(maxlen=5000)
        self.microprice_history = deque(maxlen=500)

        # Tick direction tracking
        self.tick_direction_history = deque(maxlen=100)

        # Quote churn tracking (for toxicity)
        self.bid_updates = deque(maxlen=100)
        self.ask_updates = deque(maxlen=100)
        self.trade_imbalance_history = deque(maxlen=100)

        # Current metrics
        self.latest_ofi = 0
        self.latest_ofi_velocity = 0
        self.latest_ofi_acceleration = 0
        self.latest_ofi_zscore = 0
        self.latest_volume = 0
        self.latest_mid_price = 0
        self.latest_microprice = 0
        self.latest_spread = 0
        self.latest_spread_bps = 0
        self.latest_depth_imbalance = 0
        self.latest_volatility = 0
        self.latest_quote_churn = 0
        self.latest_vpin = 0
        self.latest_tick_direction = 0

        # Data collection buffer
        self.collected_data = []
        self.last_save_time = time.time()
        self.last_print_time = time.time()
        self.total_records = 0

    def start(self):
        """Start WebSocket connection"""
        self.running = True

        def on_open(ws):
            print("WebSocket connected - Flux data collection active")

        def on_message(ws, message):
            try:
                data = json.loads(message)
                bids = data.get("b", [])
                asks = data.get("a", [])

                if not bids or not asks:
                    return

                self.current_bids = [
                    (float(p), float(q)) for p, q in bids[:OFI_DEPTH_LEVELS]
                ]
                self.current_asks = [
                    (float(p), float(q)) for p, q in asks[:OFI_DEPTH_LEVELS]
                ]

                # Calculate prices
                if self.current_bids and self.current_asks:
                    current_time = time.time()

                    # Mid price
                    mid_price = (self.current_bids[0][0] + self.current_asks[0][0]) / 2
                    self.price_history.append(mid_price)
                    self.price_timestamps.append(current_time)
                    self.latest_mid_price = mid_price

                    # Spread
                    self.latest_spread = (
                        self.current_asks[0][0] - self.current_bids[0][0]
                    )
                    self.latest_spread_bps = (self.latest_spread / mid_price) * 10000

                    # Microprice (volume-weighted mid)
                    bid_vol = self.current_bids[0][1]
                    ask_vol = self.current_asks[0][1]
                    microprice = (
                        self.current_bids[0][0] * ask_vol
                        + self.current_asks[0][0] * bid_vol
                    ) / (bid_vol + ask_vol)
                    self.latest_microprice = microprice
                    self.microprice_history.append(microprice)

                    # Tick direction (microprice vs mid)
                    if len(self.microprice_history) >= 2:
                        tick_dir = (
                            1
                            if self.microprice_history[-1] > self.microprice_history[-2]
                            else -1
                        )
                        self.tick_direction_history.append(tick_dir)
                        self.latest_tick_direction = tick_dir

                # Calculate OFI metrics if we have previous data
                if self.prev_bids and self.prev_asks:
                    ofi = self.calculate_multi_level_ofi()
                    volume = self.calculate_total_volume()

                    self.ofi_history.append(ofi)
                    self.volume_history.append(volume)
                    self.timestamp_history.append(time.time())

                    # Track quote updates for churn
                    self.track_quote_updates()

                    if len(self.ofi_history) >= 3:
                        self.latest_ofi = ofi
                        self.latest_ofi_velocity = self.calculate_ofi_velocity()
                        self.latest_ofi_acceleration = self.calculate_ofi_acceleration()
                        self.latest_ofi_zscore = self.calculate_ofi_zscore()
                        self.latest_volume = volume
                        self.latest_depth_imbalance = self.calculate_depth_imbalance()
                        self.latest_volatility = self.calculate_volatility()
                        self.latest_quote_churn = self.calculate_quote_churn()
                        self.latest_vpin = self.calculate_vpin()

                        # Collect data point
                        self.collect_data_point()

                self.prev_bids = self.current_bids.copy()
                self.prev_asks = self.current_asks.copy()

            except Exception as e:
                print(f"Calculation error: {e}")

        def on_error(ws, error):
            pass

        def on_close(ws, code, msg):
            if self.running and code != 1000:
                time.sleep(2)
                self.start()

        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )

        self.ws_thread = threading.Thread(
            target=lambda: self.ws.run_forever(ping_interval=30, ping_timeout=20),
            daemon=True,
        )
        self.ws_thread.start()

    def calculate_multi_level_ofi(self):
        """Multi-level weighted OFI"""
        weighted_ofi = 0.0
        for level in range(
            min(len(self.current_bids), len(self.current_asks), OFI_DEPTH_LEVELS)
        ):
            bid_qty = (
                self.current_bids[level][1] if level < len(self.current_bids) else 0
            )
            ask_qty = (
                self.current_asks[level][1] if level < len(self.current_asks) else 0
            )
            weight = 1.0 / (level + 1)
            level_ofi = bid_qty - ask_qty
            weighted_ofi += level_ofi * weight
        return weighted_ofi

    def calculate_total_volume(self):
        """Calculate total volume"""
        bid_volume = sum(qty for _, qty in self.current_bids)
        ask_volume = sum(qty for _, qty in self.current_asks)
        return bid_volume + ask_volume

    def calculate_ofi_velocity(self):
        """1st derivative of OFI"""
        if len(self.ofi_history) < 2:
            return 0
        return self.ofi_history[-1] - self.ofi_history[-2]

    def calculate_ofi_acceleration(self):
        """2nd derivative of OFI"""
        if len(self.ofi_history) < 3:
            return 0
        velocity_now = self.ofi_history[-1] - self.ofi_history[-2]
        velocity_prev = self.ofi_history[-2] - self.ofi_history[-3]
        return velocity_now - velocity_prev

    def calculate_ofi_zscore(self):
        """Z-score of OFI"""
        lookback = min(OFI_ZSCORE_WINDOW, len(self.ofi_history))
        if lookback < 10:
            return 0
        recent_ofi = list(self.ofi_history)[-lookback:]
        mean = np.mean(recent_ofi)
        std = np.std(recent_ofi)
        return (self.latest_ofi - mean) / std if std > 0 else 0

    def calculate_depth_imbalance(self):
        """Bid/ask depth imbalance"""
        if not self.current_bids or not self.current_asks:
            return 0
        bid_depth = sum(qty for _, qty in self.current_bids)
        ask_depth = sum(qty for _, qty in self.current_asks)
        return (
            (bid_depth - ask_depth) / (bid_depth + ask_depth)
            if (bid_depth + ask_depth) > 0
            else 0
        )

    def calculate_volatility(self):
        """Realized volatility"""
        if len(self.price_history) < 60:
            return 0
        prices = list(self.price_history)[-60:]
        returns = [
            (prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, len(prices))
        ]
        return np.std(returns) * np.sqrt(len(returns))

    def track_quote_updates(self):
        """Track quote updates for churn calculation"""
        # Count how many levels changed
        bid_changes = 0
        ask_changes = 0

        for i in range(min(5, len(self.current_bids), len(self.prev_bids))):
            if i < len(self.current_bids) and i < len(self.prev_bids):
                if (
                    self.current_bids[i][0] != self.prev_bids[i][0]
                    or abs(self.current_bids[i][1] - self.prev_bids[i][1]) > 0.001
                ):
                    bid_changes += 1

        for i in range(min(5, len(self.current_asks), len(self.prev_asks))):
            if i < len(self.current_asks) and i < len(self.prev_asks):
                if (
                    self.current_asks[i][0] != self.prev_asks[i][0]
                    or abs(self.current_asks[i][1] - self.prev_asks[i][1]) > 0.001
                ):
                    ask_changes += 1

        self.bid_updates.append(bid_changes)
        self.ask_updates.append(ask_changes)

    def calculate_quote_churn(self):
        """Quote churn rate (cancellation rate proxy)"""
        if len(self.bid_updates) < 10:
            return 0
        recent_updates = list(self.bid_updates)[-20:] + list(self.ask_updates)[-20:]
        return np.mean(recent_updates)

    def calculate_vpin(self):
        """Volume-synchronized Probability of Informed Trading (VPIN)"""
        if len(self.ofi_history) < 50:
            return 0

        # Use OFI as proxy for buy/sell volume imbalance
        recent_ofi = list(self.ofi_history)[-50:]
        buy_volume = sum(abs(x) for x in recent_ofi if x > 0)
        sell_volume = sum(abs(x) for x in recent_ofi if x < 0)
        total_volume = buy_volume + sell_volume

        if total_volume == 0:
            return 0

        vpin = abs(buy_volume - sell_volume) / total_volume
        return vpin

    def calculate_forward_returns(self, current_time, current_price):
        """Calculate forward returns for multiple timeframes"""
        returns = {}
        for timeframe in RETURN_TIMEFRAMES:
            future_time = current_time + timeframe
            future_price = None
            min_time_diff = float("inf")

            for i, ts in enumerate(self.price_timestamps):
                if ts >= future_time:
                    time_diff = abs(ts - future_time)
                    if time_diff < min_time_diff:
                        min_time_diff = time_diff
                        future_price = self.price_history[i]
                    break

            if future_price is not None and min_time_diff < 2.0:
                price_return = (future_price - current_price) / current_price
                returns[f"return_{timeframe}s"] = price_return
            else:
                returns[f"return_{timeframe}s"] = None

        return returns

    def collect_data_point(self):
        """Collect enhanced data point"""
        current_time = time.time()

        # Calculate tick direction momentum
        tick_momentum = (
            sum(self.tick_direction_history) / len(self.tick_direction_history)
            if len(self.tick_direction_history) > 0
            else 0
        )

        record = {
            "timestamp": current_time,
            "datetime": datetime.now().isoformat(),
            # Price features
            "mid_price": self.latest_mid_price,
            "microprice": self.latest_microprice,
            "microprice_edge": self.latest_microprice - self.latest_mid_price,
            "spread": self.latest_spread,
            "spread_bps": self.latest_spread_bps,
            # OFI features
            "ofi": self.latest_ofi,
            "ofi_zscore": self.latest_ofi_zscore,
            "ofi_velocity": self.latest_ofi_velocity,
            "ofi_acceleration": self.latest_ofi_acceleration,
            # Depth features
            "depth_imbalance": self.latest_depth_imbalance,
            "volume": self.latest_volume,
            # Market microstructure features
            "quote_churn": self.latest_quote_churn,
            "vpin_toxicity": self.latest_vpin,
            "tick_direction": self.latest_tick_direction,
            "tick_momentum": tick_momentum,
            # Market regime
            "volatility": self.latest_volatility,
        }

        # Add forward returns
        forward_returns = self.calculate_forward_returns(
            current_time, self.latest_mid_price
        )
        record.update(forward_returns)

        self.collected_data.append(record)

        # Periodically save
        if time.time() - self.last_save_time >= SAVE_INTERVAL_SECONDS:
            self.save_data()
            self.last_save_time = time.time()

        # Periodically print status
        if time.time() - self.last_print_time >= PRINT_INTERVAL_SECONDS:
            self.print_status()
            self.last_print_time = time.time()

    def save_data(self):
        """Save data to CSV"""
        if not self.collected_data:
            return

        df = pd.DataFrame(self.collected_data)
        self.backfill_forward_returns(df)

        if os.path.exists(CSV_FILENAME):
            df.to_csv(CSV_FILENAME, mode="a", header=False, index=False)
        else:
            df.to_csv(CSV_FILENAME, mode="w", header=True, index=False)

        self.total_records += len(self.collected_data)
        print(
            f"FLUX - Saved {len(self.collected_data)} records | Total: {self.total_records:,}"
        )

        self.collected_data = []

    def backfill_forward_returns(self, df):
        """Backfill forward returns"""
        for idx, row in df.iterrows():
            if pd.notna(row["return_5s"]):
                continue

            record_time = row["timestamp"]
            record_price = row["mid_price"]
            forward_returns = self.calculate_forward_returns(record_time, record_price)

            for key, value in forward_returns.items():
                if value is not None:
                    df.at[idx, key] = value

    def print_status(self):
        """Print collection status"""
        print("\n" + "=" * 70)
        print("  FLUX ALGORITHMIC TRADING SYSTEM")
        print("  Market Data Collector - BTCUSDT")
        print("=" * 70)
        print(f"  Total records: {self.total_records:,}")
        print(f"  Buffer: {len(self.collected_data)}")
        print(
            f"  Mid: ${self.latest_mid_price:,.2f} | Spread: {self.latest_spread_bps:.2f}bps"
        )
        print(f"  OFI: {self.latest_ofi:.2f} | Z: {self.latest_ofi_zscore:.2f}")
        print(f"  Depth Imbalance: {self.latest_depth_imbalance:.4f}")
        print(
            f"  VPIN: {self.latest_vpin:.4f} | Quote Churn: {self.latest_quote_churn:.2f}"
        )

        hours_collected = self.total_records / 36000
        print(f"  Coverage: {hours_collected:.2f} hours")
        print("=" * 70 + "\n")

    def stop(self):
        """Stop and save"""
        self.running = False
        if self.ws:
            self.ws.close()
        if self.ws_thread:
            self.ws_thread.join(timeout=2)
        self.save_data()


def print_banner():
    """Print Flux startup banner"""
    print("\n" + "=" * 70)
    print("  FLUX ALGORITHMIC TRADING SYSTEM")
    print("  Market Data Collector - BTCUSDT")
    print("=" * 70)
    print(f"  OFI, VPIN, microprice, spread, tick momentum")
    print(f"  Output: {CSV_FILENAME}")
    print(f"  Forward labels: 5s, 10s, 30s")
    print("=" * 70 + "\n")


def main():
    print_banner()
    print("FLUX DATA COLLECTION ACTIVE (Ctrl+C to stop)\n")

    collector = EnhancedOFICollector()
    collector.start()
    time.sleep(5)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("FLUX SHUTDOWN REQUESTED - Saving data...")
        print("=" * 80)
        collector.stop()
        print("\nFLUX DATA SAVED SUCCESSFULLY")
        print(f"   File: {CSV_FILENAME}")
        print(f"   Ready for Flux ML model training")
        print(f"   Next: Run feature engineering and model training modules\n")
        print("=" * 80)
        print("Flux data collection complete. System ready for next stage.")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
