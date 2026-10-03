import logging
import json
import os
import sqlite3
import asyncio
from filelock import FileLock
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

TIMEFRAMES = {
    "M1": "TIMEFRAME_M1", "M5": "TIMEFRAME_M5", "M15": "TIMEFRAME_M15",
    "M30": "TIMEFRAME_M30", "H1": "TIMEFRAME_H1", "H4": "TIMEFRAME_H4",
    "D1": "TIMEFRAME_D1",
}
TIMEFRAME_SECONDS = {"M1": 60, "M5": 300, "M15": 900, "M30": 1800, "H1": 3600, "H4": 14400, "D1": 86400}
_cache = {}
os.makedirs("data/forecast", exist_ok=True)
# ponytail: serialize terminal access; use a separate MT5 bridge for multiple hosts.
_terminal_lock = FileLock("data/forecast/terminal.lock", timeout=30)


@contextmanager
def _connect():
    os.makedirs("data/forecast", exist_ok=True)
    conn = sqlite3.connect("data/forecast/history.db", timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE IF NOT EXISTS forecasts (symbol TEXT, timeframe TEXT, anchor INTEGER, payload TEXT, PRIMARY KEY(symbol,timeframe,anchor))")
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def _record_and_evaluate(symbol, timeframe, rates, forecast):
    # Evaluate by candle count so weekends and trading pauses do not shift the target.
    times = [int(row["time"]) for row in rates]
    with _connect() as conn:
        rows = conn.execute("SELECT anchor,payload FROM forecasts WHERE symbol=? AND timeframe=?", (symbol, timeframe)).fetchall()
        for row in rows:
            item = json.loads(row["payload"])
            if item["status"] != "pending" or row["anchor"] not in times:
                continue
            target_index = times.index(row["anchor"]) + 30
            if target_index >= len(rates):
                continue
            actual = float(rates[target_index]["close"])
            item.update(actual_price=actual, target_timestamp=times[target_index], status="completed",
                        error_percent=round(abs(actual-item["predicted_price"])/max(abs(actual), 1e-10)*100, 4),
                        direction_correct=(actual >= item["anchor_price"]) == (item["predicted_price"] >= item["anchor_price"]))
            conn.execute("UPDATE forecasts SET payload=? WHERE symbol=? AND timeframe=? AND anchor=?", (json.dumps(item), symbol, timeframe, row["anchor"]))
        anchor = times[-1]
        item = {"key": f"{symbol}:{timeframe}:{anchor}", "symbol": symbol, "timeframe": timeframe,
                "anchor_timestamp": anchor, "anchor_price": forecast["current_price"],
                "predicted_price": forecast["projected_price"], "projection": forecast["projection"],
                "target_timestamp": anchor + TIMEFRAME_SECONDS[timeframe] * 30,
                "created_at": forecast["updated_at"], "status": "pending"}
        conn.execute("INSERT OR IGNORE INTO forecasts VALUES (?,?,?,?)", (symbol, timeframe, anchor, json.dumps(item)))


def get_history(symbol: Optional[str] = None, timeframe: Optional[str] = None):
    with _connect() as conn:
        rows = conn.execute("SELECT payload FROM forecasts WHERE (? IS NULL OR symbol=?) AND (? IS NULL OR timeframe=?) ORDER BY anchor DESC", (symbol, symbol, timeframe, timeframe)).fetchall()
    items = [json.loads(row["payload"]) for row in rows]
    for item in items:
        item.setdefault("target_timestamp", item["anchor_timestamp"] + TIMEFRAME_SECONDS.get(item["timeframe"], 60) * 30)
    completed = [x for x in items if x["status"] == "completed"]
    return {"items": items[:50], "completed": len(completed), "total": len(items),
            "direction_accuracy": round(sum(x["direction_correct"] for x in completed)/len(completed)*100, 2) if completed else None,
            "average_error_percent": round(sum(x["error_percent"] for x in completed)/len(completed), 4) if completed else None}


async def monitor_forecasts():
    while True:
        try:
            with _connect() as conn:
                markets = conn.execute("SELECT DISTINCT symbol,timeframe FROM forecasts").fetchall()
            for market in markets:
                try:
                    await asyncio.to_thread(get_forecast, market["symbol"], market["timeframe"])
                except Exception as exc:
                    logger.warning("Forecast monitor: %s", exc)
        except Exception:
            logger.exception("Forecast monitor failed")
        await asyncio.sleep(5)


def _mt5():
    try:
        import MetaTrader5 as mt5
        return mt5
    except ImportError as exc:
        raise RuntimeError("MetaTrader5 belum terpasang di server") from exc


def _normalize(values: np.ndarray) -> np.ndarray:
    return (values - values.mean()) / (values.std() + 1e-10)


def _resolve_symbol(mt5, requested: str) -> str:
    requested = requested.strip().upper()
    if mt5.symbol_info(requested) is not None:
        return requested
    symbols = mt5.symbols_get() or []
    matches = [item.name for item in symbols if item.name.upper().startswith(requested)]
    if matches:
        return sorted(matches, key=lambda name: (not name.upper().endswith("M"), len(name)))[0]
    raise ValueError(f"Market {requested} tidak ditemukan di MT5")


def get_forecast(symbol: str, timeframe: str, bars: int = 10000) -> Dict[str, Any]:
    with _terminal_lock:
        return _get_forecast(symbol, timeframe, bars)


def _get_forecast(symbol: str, timeframe: str, bars: int = 10000) -> Dict[str, Any]:
    """Run the fraktal pattern scan from the supplied MT5 script."""
    mt5 = _mt5()
    timeframe = timeframe.upper()
    if timeframe not in TIMEFRAMES:
        raise ValueError("Timeframe tidak didukung")
    if not mt5.initialize():
        raise RuntimeError("MetaTrader 5 tidak terhubung")
    try:
        symbol = _resolve_symbol(mt5, symbol)
        if not mt5.symbol_select(symbol, True):
            raise ValueError(f"Market {symbol} tidak bisa diaktifkan di MT5")
        rates = mt5.copy_rates_from_pos(symbol, getattr(mt5, TIMEFRAMES[timeframe]), 1, max(bars, 400))
        if rates is None or len(rates) < 400:
            raise RuntimeError("Data candle MT5 belum cukup")

        tick = mt5.symbol_info_tick(symbol)
        tick_price = float(tick.bid) if tick else None
        last_bar = int(rates[-1]["time"])
        cache_key = (symbol, timeframe)
        cached = _cache.get(cache_key)
        if cached and cached["last_bar_time"] == datetime.fromtimestamp(last_bar, timezone.utc).isoformat():
            result = dict(cached)
            result.update(live_price=tick_price, history=get_history(symbol, timeframe), updated_at=datetime.now(timezone.utc).isoformat())
            return result

        prices = np.asarray(rates["close"], dtype=float)
        window_size, horizon = 180, 30
        current = prices[-window_size:]
        historical = prices[:-window_size]
        current_norm = _normalize(current)
        best_distance, best_idx = float("inf"), -1
        for idx in range(0, len(historical) - window_size - horizon, 5):
            distance = float(np.mean(np.abs(current_norm - _normalize(historical[idx:idx + window_size]))))
            if distance < best_distance:
                best_distance, best_idx = distance, idx
        if best_idx < 0:
            raise RuntimeError("Belum ada pola historis yang bisa dibandingkan")

        past_future = historical[best_idx + window_size:best_idx + window_size + horizon]
        projected = past_future + (current[-1] - past_future[0])
        last_bar = int(rates[-1]["time"])
        result = {
            "symbol": symbol,
            "timeframe": timeframe,
            "current_price": round(float(current[-1]), 8),
            "projected_price": round(float(projected[-1]), 8),
            "change_percent": round(float((projected[-1] / current[-1] - 1) * 100), 3),
            "direction": "Bullish" if projected[-1] >= current[-1] else "Bearish",
            "pattern_error": round(best_distance, 5),
            "last_bar_time": datetime.fromtimestamp(last_bar, timezone.utc).isoformat(),
            "current": [round(float(v), 8) for v in current],
            "projection": [round(float(v), 8) for v in projected],
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "live_price": tick_price,
        }
        _record_and_evaluate(symbol, timeframe, rates, result)
        result["history"] = get_history(symbol, timeframe)
        _cache[cache_key] = result
        return result
    finally:
        mt5.shutdown()
