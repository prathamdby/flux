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
- Forward returns for ML model training

Part of the Flux end-to-end algorithmic trading system.
"""

import asyncio
import csv
import io
import json
import math
import time
from collections import deque

import aiofiles
import numpy as np
import requests
import websockets
from sortedcontainers import SortedDict

SYMBOL = "BTCUSDT"
SYMBOL_BASE = SYMBOL[:3]  # Extract base currency (SOL, BTC, etc.)
DEPTH_WS_URL = f"wss://fstream.binance.com/ws/{SYMBOL.lower()}@depth20@100ms"
TRADE_WS_URL = f"wss://fstream.binance.com/ws/{SYMBOL.lower()}@aggTrade"
SNAPSHOT_URL = f"https://fapi.binance.com/fapi/v1/depth?symbol={SYMBOL}&limit=1000"
OUT_CSV = "flux_data.csv"
CSV_FIELDNAMES = [
    "ts",
    "mid",
    "spread",
    "spread_bps",
    "micro",
    "microprice_edge",
    "ofi",
    "ofi_vel",
    "ofi_acc",
    "ofi_z",
    "bid_depth",
    "ask_depth",
    "balance",
    "trade_vol",
    "trade_count",
    "tick_up",
    "tick_dn",
    "tick_momentum",
    "churn",
    "cancel_rate",
    "vpin",
    "volatility",
    "forward_return_5s",
    "target_direction",
]
WINDOW_SEC = 1.0
FORWARD_WINDOW_SEC = 5.0  # For forward return labels
OFI_DEPTH_LEVELS = 15  # Multi-level OFI calculation depth

# Retry & resilience config
RETRY_BACKOFF_BASE = 1.0  # Start with 1 second
RETRY_BACKOFF_MAX = 30.0  # Cap at 30 seconds
STALE_CONNECTION_TIMEOUT = 60.0  # Reconnect if no data for 60s

# Logging config
VERBOSE_LOGGING = True
last_log_time = 0
trade_message_count = 0
depth_message_count = 0
start_time = 0  # Track when script started

# Health monitoring
last_depth_msg_time = 0
last_trade_msg_time = 0

# rolling buffers
ROLL_N = 100
ofi_roll = deque(maxlen=ROLL_N)
velocity_roll = deque(maxlen=ROLL_N)
mid_roll = deque(maxlen=ROLL_N)
trade_tick_history = deque(maxlen=50)
microprice_history = deque(maxlen=100)

# forward return buffer - stores recent rows with timestamp and mid price
forward_buffer = deque(maxlen=1000)

# window accumulators
trade_vol_window = 0.0
trade_count_window = 0
trade_dir_counts = {"+1": 0, "0": 0, "-1": 0}
quote_churn = 0.0
cancel_count = 0
update_count = 0
prev_ofi = 0.0
prev_velocity = 0.0
prev_mid = None
prev_trade_price = None

# simple VPIN
VPIN_BUCKETS = 10
vpin_bucket_vol = [0.0] * VPIN_BUCKETS
vpin_bucket_imb = [0.0] * VPIN_BUCKETS
vpin_cur_bucket = 0
vpin_bucket_target = 1.0

# shared state between websockets
bids = SortedDict()
asks = SortedDict()
lastUpdateId = 0
csv_file = None
csv_queue = None
csv_writer_task = None


def get_elapsed_time():
    """Get formatted elapsed time since script start (human-readable)"""
    if start_time == 0:
        return "0s"
    elapsed = int(time.time() - start_time)
    hours = elapsed // 3600
    minutes = (elapsed % 3600) // 60
    seconds = elapsed % 60

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 0 or not parts:
        parts.append(f"{seconds}s")

    return " ".join(parts)


async def retry_with_backoff(coro_func, name, *args, **kwargs):
    """Retry a coroutine with exponential backoff on failure"""
    attempt = 0
    while True:
        try:
            await coro_func(*args, **kwargs)
            # If we get here, connection ended normally (shouldn't happen in infinite loop)
            print(f"\n⚠️  {name} ended unexpectedly, reconnecting...")
        except asyncio.CancelledError:
            print(f"\n🛑 {name} cancelled, shutting down...")
            raise  # Propagate cancellation
        except Exception as e:
            attempt += 1
            backoff = min(RETRY_BACKOFF_BASE * (2 ** (attempt - 1)), RETRY_BACKOFF_MAX)
            print(f"\n❌ {name} failed: {e}")
            print(f"🔄 Reconnecting in {backoff:.1f}s (attempt #{attempt})...")
            await asyncio.sleep(backoff)
            # Reset counter on successful connection (will be set by next iteration)
            if attempt > 5:
                attempt = 0  # Reset after sustained failures to avoid huge backoffs


def calculate_weighted_ofi(bids: SortedDict, asks: SortedDict):
    """
    Calculate multi-level weighted OFI from order book depth.
    Weights decay by 1/(level+1) to emphasize top-of-book while capturing depth.
    """
    weighted_ofi = 0.0
    level = 0

    bid_iter = bids.irange(reverse=True)
    ask_iter = asks.irange()

    for bid_price, ask_price in zip(bid_iter, ask_iter):
        if level >= OFI_DEPTH_LEVELS:
            break
        weight = 1.0 / (level + 1)
        bid_qty = bids.get(bid_price, 0.0)
        ask_qty = asks.get(ask_price, 0.0)
        weighted_ofi += (bid_qty - ask_qty) * weight
        level += 1

    return weighted_ofi


def _csv_line_from_values(values):
    buffer = io.StringIO()
    csv.writer(buffer).writerow(values)
    return buffer.getvalue()


def _serialize_row(row):
    values = [
        "" if (val := row.get(field)) is None else val for field in CSV_FIELDNAMES
    ]
    return _csv_line_from_values(values)


async def csv_writer_worker(queue: asyncio.Queue, file):
    header_line = _csv_line_from_values(CSV_FIELDNAMES)
    await file.write(header_line)
    await file.flush()

    while True:
        rows = await queue.get()
        if rows is None:
            queue.task_done()
            break
        for row in rows:
            await file.write(_serialize_row(row))
        await file.flush()
        queue.task_done()


def print_banner():
    """Print Flux startup banner"""
    print("\n" + "=" * 70)
    print("  FLUX ALGORITHMIC TRADING SYSTEM")
    print("  Market Data Collector - " + SYMBOL)
    print("=" * 70)
    print(f"  📊 OFI, VPIN, microprice, spread, tick momentum")
    print(f"  📝 Output: {OUT_CSV}")
    print(f"  ⏱️  1s metrics | 5s forward labels")
    print("=" * 70 + "\n")


def log_stream_stats():
    """Log stream health"""
    global last_log_time, trade_message_count, depth_message_count
    current_time = time.time()

    if current_time - last_log_time >= 10:
        print(
            f"📡 Depth: {depth_message_count}/10s | Trades: {trade_message_count}/10s"
        )
        last_log_time = current_time
        trade_message_count = 0
        depth_message_count = 0


def ingest_update(delta, is_decrease):
    global quote_churn, cancel_count, update_count
    quote_churn += abs(delta)
    update_count += 1
    if is_decrease:
        cancel_count += 1


def ingest_trade(price, size):
    global trade_vol_window, trade_count_window, prev_trade_price, trade_tick_history
    trade_vol_window += size
    trade_count_window += 1
    if prev_trade_price is None:
        d = 0
        d_key = "0"
    else:
        if price > prev_trade_price:
            d = 1
            d_key = "+1"
        elif price < prev_trade_price:
            d = -1
            d_key = "-1"
        else:
            d = 0
            d_key = "0"
    trade_dir_counts[d_key] += 1
    prev_trade_price = price
    trade_tick_history.append(d)
    global vpin_cur_bucket, vpin_bucket_vol, vpin_bucket_imb, vpin_bucket_target
    vpin_bucket_vol[vpin_cur_bucket] += size
    vpin_bucket_imb[vpin_cur_bucket] += d * size
    if vpin_bucket_vol[vpin_cur_bucket] >= vpin_bucket_target:
        vpin_cur_bucket = (vpin_cur_bucket + 1) % VPIN_BUCKETS


def reset_window():
    global trade_vol_window, trade_count_window, trade_dir_counts, quote_churn, cancel_count, update_count
    trade_vol_window = 0.0
    trade_count_window = 0
    trade_dir_counts = {"+1": 0, "0": 0, "-1": 0}
    quote_churn = 0.0
    cancel_count = 0
    update_count = 0


def collect_completed_rows():
    global forward_buffer

    if len(forward_buffer) < 2:
        return []

    current_time = time.time()
    completed = []

    for row in forward_buffer:
        if row.get("_written"):
            continue

        row_time = row["ts"] / 1000.0
        if current_time - row_time < FORWARD_WINDOW_SEC:
            continue

        if row["mid"] > 0 and row.get("forward_return_5s") is None:
            target_time = row["ts"] + (FORWARD_WINDOW_SEC * 1000)
            future_mid = None
            for future_row in forward_buffer:
                if future_row["ts"] >= target_time:
                    future_mid = future_row["mid"]
                    break

            if future_mid and future_mid > 0:
                forward_ret = math.log(future_mid / row["mid"])
                row["forward_return_5s"] = forward_ret
                row["target_direction"] = 1 if forward_ret > 0 else 0

        completed.append(row)
        row["_written"] = True

    if len(completed) > 10:
        cutoff_time = current_time - (FORWARD_WINDOW_SEC * 2)
        forward_buffer[:] = [
            row
            for row in forward_buffer
            if not row.get("_written") or (row["ts"] / 1000.0) > cutoff_time
        ]

    return completed


def compute_metrics(ofi_acc):
    global prev_ofi, prev_velocity, prev_mid, bids, asks

    best_bid = bids.peekitem(-1)[0] if bids else 0
    best_ask = asks.peekitem(0)[0] if asks else 0

    if best_bid == 0 or best_ask == 0:
        return None

    weighted_ofi = calculate_weighted_ofi(bids, asks)
    ofi_to_use = weighted_ofi

    bid_size_top = bids.get(best_bid, 0.0)
    ask_size_top = asks.get(best_ask, 0.0)
    mid = (best_bid + best_ask) / 2
    spread = abs(best_ask - best_bid)

    spread_bps = (spread / mid) * 10000 if mid > 0 else 0

    micro = (
        (best_bid * ask_size_top + best_ask * bid_size_top)
        / (bid_size_top + ask_size_top)
        if (bid_size_top + ask_size_top) > 0
        else mid
    )

    microprice_edge = micro - mid

    microprice_history.append(micro)

    tick_momentum = (
        sum(trade_tick_history) / len(trade_tick_history)
        if len(trade_tick_history) > 0
        else 0
    )

    velocity = ofi_to_use - prev_ofi
    acceleration = velocity - prev_velocity
    ofi_roll.append(ofi_to_use)
    velocity_roll.append(velocity)
    prev_velocity = velocity
    prev_ofi = ofi_to_use

    ofi_z = None
    if len(ofi_roll) >= 5:
        ofi_array = np.fromiter(ofi_roll, dtype=np.float64)
        mu = np.mean(ofi_array)
        sd = np.std(ofi_array)
        sd = sd if sd > 0 else 1.0
        ofi_z = (ofi_to_use - mu) / sd

    topN = 20
    bid_depth = 0.0
    ask_depth = 0.0

    for level, price in enumerate(bids.irange(reverse=True)):
        if level >= topN:
            break
        bid_depth += bids[price]

    for level, price in enumerate(asks.irange()):
        if level >= topN:
            break
        ask_depth += asks[price]
    balance = (
        (bid_depth - ask_depth) / (bid_depth + ask_depth)
        if (bid_depth + ask_depth) > 0
        else 0
    )

    churn = quote_churn
    cancel_rate = cancel_count / update_count if update_count > 0 else 0

    total_bucket_vol = sum(vpin_bucket_vol)
    vpin = (
        sum(abs(x) for x in vpin_bucket_imb) / total_bucket_vol
        if total_bucket_vol > 0
        else 0
    )

    mid_roll.append(mid)
    vol = None
    if len(mid_roll) >= 2:
        mid_array = np.fromiter((m for m in mid_roll if m > 0), dtype=np.float64)
        if len(mid_array) >= 2:
            rets = np.log(mid_array[1:] / mid_array[:-1])
            if rets.size > 1:
                vol = float(np.std(rets))
            elif rets.size == 1:
                vol = float(abs(rets[0]))

    ts = int(time.time() * 1000)
    metrics = {
        "ts": ts,
        "mid": mid,
        "spread": spread,
        "spread_bps": spread_bps,
        "micro": micro,
        "microprice_edge": microprice_edge,
        "ofi": ofi_to_use,
        "ofi_vel": velocity,
        "ofi_acc": acceleration,
        "ofi_z": ofi_z,
        "bid_depth": bid_depth,
        "ask_depth": ask_depth,
        "balance": balance,
        "trade_vol": trade_vol_window,
        "trade_count": trade_count_window,
        "tick_up": trade_dir_counts["+1"],
        "tick_dn": trade_dir_counts["-1"],
        "tick_momentum": tick_momentum,
        "churn": churn,
        "cancel_rate": cancel_rate,
        "vpin": vpin,
        "volatility": vol,
        "forward_return_5s": None,
        "target_direction": None,
    }
    reset_window()
    prev_mid = mid

    # Add to forward buffer
    forward_buffer.append(metrics.copy())

    return metrics


async def handle_depth_stream(ofi_acc_container, win_start_container):
    """Handle order book depth updates"""
    global bids, asks, lastUpdateId, depth_message_count, last_depth_msg_time

    print("📖 Depth stream connecting...")

    async with websockets.connect(
        DEPTH_WS_URL, close_timeout=1, ping_interval=30, ping_timeout=10
    ) as ws:
        print("✅ Depth stream live")
        last_depth_msg_time = time.time()

        async for msg in ws:
            depth_message_count += 1
            last_depth_msg_time = time.time()
            data = json.loads(msg)
            U, u = data.get("U"), data.get("u")
            if U and u and u < lastUpdateId:
                continue
            if not (U and u):
                continue

            for price_s, qty_s in data.get("b", []):
                price, qty = float(price_s), float(qty_s)

                if qty == 0.0:
                    old = bids.pop(price, 0.0)
                    delta = -old
                    is_decrease = old > 0
                else:
                    old = bids.get(price, 0.0)
                    delta = qty - old
                    is_decrease = qty < old
                    if delta != 0:
                        bids[price] = qty

                ofi_acc_container[0] += delta
                ingest_update(delta, is_decrease)

            for price_s, qty_s in data.get("a", []):
                price, qty = float(price_s), float(qty_s)

                if qty == 0.0:
                    old = asks.pop(price, 0.0)
                    delta = -old
                    is_decrease = old > 0
                else:
                    old = asks.get(price, 0.0)
                    delta = qty - old
                    is_decrease = qty < old
                    if delta != 0:
                        asks[price] = qty

                ofi_acc_container[0] -= delta
                ingest_update(delta, is_decrease)

            lastUpdateId = max(lastUpdateId, u)
            log_stream_stats()


async def handle_trade_stream():
    """Handle aggregate trade updates"""
    global trade_message_count, last_trade_msg_time

    print("📊 Trade stream connecting...")

    async with websockets.connect(
        TRADE_WS_URL, close_timeout=1, ping_interval=30, ping_timeout=10
    ) as ws:
        print("✅ Trade stream live")
        last_trade_msg_time = time.time()

        async for msg in ws:
            data = json.loads(msg)
            if data.get("e") == "aggTrade":
                trade_message_count += 1
                last_trade_msg_time = time.time()
                price = float(data["p"])
                quantity = float(data["q"])
                ingest_trade(price, quantity)


async def periodic_metrics_writer(ofi_acc_container, win_start_container):
    """Periodically compute and write metrics"""
    global csv_queue
    row_count = 0
    total_rows_computed = 0

    while True:
        await asyncio.sleep(0.1)  # Check every 100ms

        now = time.time()
        if now - win_start_container[0] >= WINDOW_SEC:
            metrics = compute_metrics(ofi_acc_container[0])
            if metrics:
                total_rows_computed += 1
                # Note: metrics are added to forward_buffer in compute_metrics()
                # They'll be written once forward returns can be calculated

                completed_rows = collect_completed_rows()
                if completed_rows:
                    await csv_queue.put(completed_rows)
                    row_count += len(completed_rows)

                if total_rows_computed % 10 == 0:
                    buffered = len(forward_buffer)
                    print(
                        f"\n📊 #{total_rows_computed} | Saved: {row_count} | Buffered: {buffered}"
                    )
                    print(
                        f"   Mid: ${metrics['mid']:,.2f} | Spread: {metrics['spread_bps']:.1f}bps"
                    )
                    print(
                        f"   OFI: {metrics['ofi']:+.2f} | VPIN: {metrics['vpin']:.2%} | Momentum: {metrics['tick_momentum']:+.2f}"
                    )
                    print(f"   Elapsed: {get_elapsed_time()}\n")

                ofi_acc_container[0] = 0.0
                win_start_container[0] = now


async def run():
    global bids, asks, lastUpdateId, csv_file, start_time, csv_queue, csv_writer_task

    start_time = time.time()

    print_banner()

    resp = requests.get(SNAPSHOT_URL, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    if resp.status_code != 200:
        raise Exception(f"Snapshot failed: {resp.status_code}")

    snap = resp.json()
    lastUpdateId = snap["lastUpdateId"]
    bids = SortedDict({float(p): float(q) for p, q in snap["bids"]})
    asks = SortedDict({float(p): float(q) for p, q in snap["asks"]})

    best_bid = bids.peekitem(-1)[0] if bids else 0
    best_ask = asks.peekitem(0)[0] if asks else 0
    mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else 0
    spread = best_ask - best_bid if best_bid and best_ask else 0

    print(f"📸 Snapshot: {len(bids)} bid | {len(asks)} ask | Mid: ${mid_price:,.2f}")

    csv_queue = asyncio.Queue()
    csv_file = await aiofiles.open(OUT_CSV, "w")
    csv_writer_task = asyncio.create_task(csv_writer_worker(csv_queue, csv_file))
    print(f"💾 CSV ready: {OUT_CSV}\n")

    ofi_acc_container = [0.0]
    win_start_container = [time.time()]

    print("🚀 FLUX DATA COLLECTION ACTIVE (Ctrl+C to stop)\n")

    try:
        depth_task = asyncio.create_task(
            retry_with_backoff(
                handle_depth_stream,
                "Depth Stream",
                ofi_acc_container,
                win_start_container,
            )
        )
        trade_task = asyncio.create_task(
            retry_with_backoff(handle_trade_stream, "Trade Stream")
        )
        metrics_task = asyncio.create_task(
            periodic_metrics_writer(ofi_acc_container, win_start_container)
        )

        await asyncio.gather(depth_task, trade_task, metrics_task)
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n\n" + "=" * 80)
        print("🛑 FLUX SHUTDOWN REQUESTED - Saving data...")
        print("=" * 80)
    except Exception as e:
        print(f"\n⚠️  Error during collection: {e}")
        print("Saving data before exit...")
    finally:
        buffered_count = len(forward_buffer)
        unwritten_count = sum(1 for row in forward_buffer if not row.get("_written"))

        print(f"\n💾 FLUX - FINALIZING DATA:")
        print(f"   Buffered rows: {buffered_count}")
        print(f"   Need to write: {unwritten_count}")

        try:
            if unwritten_count > 0:
                print(f"   Writing remaining data...")
                remaining_rows = [
                    row for row in forward_buffer if not row.get("_written")
                ]
                for row in remaining_rows:
                    row.setdefault("forward_return_5s", None)
                    row.setdefault("target_direction", None)
                if remaining_rows:
                    await csv_queue.put(remaining_rows)
        finally:
            if csv_queue:
                await csv_queue.put(None)
            tasks = [t for t in (csv_writer_task,) if t and not t.done()]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            if csv_file:
                await csv_file.flush()
                await csv_file.close()

        print(f"\n✅ FLUX DATA SAVED SUCCESSFULLY!")
        print(f"   File: {OUT_CSV}")
        print(f"   Ready for Flux ML model training")
        print(f"   Next: Run feature engineering and model training modules\n")
        print("=" * 80)
        print("👋 Flux data collection complete. System ready for next stage.")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass  # Handled in run() function
