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
_cache: Dict[Any, Any] = {}
os.makedirs("data/forecast", exist_ok=True)
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
                "target_timestamp": anchor + TIMEFRAME_SECONDS.get(timeframe, 60) * 30,
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
                    await asyncio.to_thread(get_forecast, market["symbol"], market["timeframe"], 10000, False)
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
        raise RuntimeError("MetaTrader5 belum terpasang di sistem") from exc


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


def _generate_simulated_rates(symbol: str, timeframe: str, count: int = 500) -> np.ndarray:
    """Generate realistic synthetic candle rate series when MT5 is not used or unavailable."""
    sec = TIMEFRAME_SECONDS.get(timeframe.upper(), 60)
    now = int(datetime.now(timezone.utc).timestamp())
    last_candle_time = (now // sec) * sec

    s = symbol.upper()
    if "BTC" in s:
        base_price, vol = 65000.0, 0.0015
    elif "ETH" in s:
        base_price, vol = 3450.0, 0.002
    elif "XAU" in s or "GOLD" in s:
        base_price, vol = 2320.0, 0.0012
    elif "EUR" in s:
        base_price, vol = 1.0850, 0.0004
    elif "GBP" in s:
        base_price, vol = 1.2720, 0.0005
    elif "JPY" in s:
        base_price, vol = 155.20, 0.0006
    else:
        hash_val = sum(ord(c) for c in s)
        base_price = 100.0 + (hash_val % 400)
        vol = 0.001

    sym_seed = sum(ord(c) * (i + 1) for i, c in enumerate(s)) % 100000
    rng = np.random.RandomState(sym_seed)

    times = [last_candle_time - (count - 1 - i) * sec for i in range(count)]
    t = np.arange(count, dtype=float)
    cycle1 = 0.015 * np.sin(2 * np.pi * t / 65.0)
    cycle2 = 0.025 * np.sin(2 * np.pi * t / 140.0 + 1.2)
    cycle3 = 0.008 * np.cos(2 * np.pi * t / 28.0)
    steps = rng.normal(0, vol, count)
    walk = np.cumsum(steps)

    prices = base_price * np.maximum(0.1, 1.0 + cycle1 + cycle2 + cycle3 + walk)
    # micro variation on current bar
    prices[-1] += prices[-1] * rng.uniform(-0.00015, 0.00015)

    rates_list = [(int(times[i]), float(prices[i])) for i in range(count)]
    return np.array(rates_list, dtype=[("time", "i8"), ("close", "f8")])


def _compute_fractal_forecast(
    rates: np.ndarray,
    symbol: str,
    timeframe: str,
    live_price: Optional[float] = None,
    source: str = "simulation",
    source_label: str = "Mode Simulasi",
    mt5_notice: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute fractal pattern match and projection from candle close rates."""
    last_bar = int(rates[-1]["time"])
    cache_key = (symbol, timeframe, source)
    cached = _cache.get(cache_key)
    if cached and cached["last_bar_time"] == datetime.fromtimestamp(last_bar, timezone.utc).isoformat():
        result = dict(cached)
        current_p = result["current_price"]
        lp = live_price if live_price is not None else current_p
        result.update(
            live_price=round(float(lp), 4 if lp < 1000 else 2),
            history=get_history(symbol, timeframe),
            updated_at=datetime.now(timezone.utc).isoformat(),
            source=source,
            source_label=source_label,
            mt5_notice=mt5_notice,
        )
        return result

    prices = np.asarray(rates["close"], dtype=float)
    window_size, horizon = 180, 30
    if len(prices) < window_size + horizon + 20:
        raise RuntimeError("Data candle belum cukup untuk kalkulasi fraktal")

    current = prices[-window_size:]
    historical = prices[:-window_size]
    current_norm = _normalize(current)
    best_distance, best_idx = float("inf"), -1
    for idx in range(0, len(historical) - window_size - horizon, 5):
        distance = float(np.mean(np.abs(current_norm - _normalize(historical[idx:idx + window_size]))))
        if distance < best_distance:
            best_distance, best_idx = distance, idx

    if best_idx < 0:
        raise RuntimeError("Belum ada pola historis yang cocok untuk dibandingkan")

    past_future = historical[best_idx + window_size:best_idx + window_size + horizon]
    projected = past_future + (current[-1] - past_future[0])
    cur_val = float(current[-1])
    proj_val = float(projected[-1])
    decimals = 4 if cur_val < 1000 else 2

    result = {
        "symbol": symbol,
        "timeframe": timeframe,
        "current_price": round(cur_val, decimals),
        "projected_price": round(proj_val, decimals),
        "change_percent": round(float((proj_val / cur_val - 1) * 100), 3),
        "direction": "Bullish" if proj_val >= cur_val else "Bearish",
        "pattern_error": round(best_distance, 5),
        "last_bar_time": datetime.fromtimestamp(last_bar, timezone.utc).isoformat(),
        "current": [round(float(v), decimals) for v in current],
        "projection": [round(float(v), decimals) for v in projected],
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "live_price": round(float(live_price), decimals) if live_price is not None else round(cur_val, decimals),
        "source": source,
        "source_label": source_label,
        "mt5_notice": mt5_notice,
    }
    _record_and_evaluate(symbol, timeframe, rates, result)
    result["history"] = get_history(symbol, timeframe)
    _cache[cache_key] = result
    return result


def _get_forecast_simulated(symbol: str, timeframe: str, mt5_notice: Optional[str] = None) -> Dict[str, Any]:
    timeframe = timeframe.upper()
    if timeframe not in TIMEFRAMES:
        raise ValueError("Timeframe tidak didukung")
    rates = _generate_simulated_rates(symbol, timeframe, count=500)
    cur = float(rates[-1]["close"])
    return _compute_fractal_forecast(
        rates=rates,
        symbol=symbol,
        timeframe=timeframe,
        live_price=cur,
        source="simulation",
        source_label="Simulasi (Tanpa MT5)" if not mt5_notice else "Simulasi (Fallback MT5)",
        mt5_notice=mt5_notice,
    )


def _get_forecast_mt5(symbol: str, timeframe: str, bars: int = 10000) -> Dict[str, Any]:
    """Run the fractal pattern scan using live data from MT5 terminal."""
    mt5 = _mt5()
    timeframe = timeframe.upper()
    if timeframe not in TIMEFRAMES:
        raise ValueError("Timeframe tidak didukung")
    if not mt5.initialize():
        raise RuntimeError("MetaTrader 5 tidak terhubung / belum dibuka")
    try:
        resolved_symbol = _resolve_symbol(mt5, symbol)
        if not mt5.symbol_select(resolved_symbol, True):
            raise ValueError(f"Market {resolved_symbol} tidak bisa diaktifkan di MT5")
        rates = mt5.copy_rates_from_pos(resolved_symbol, getattr(mt5, TIMEFRAMES[timeframe]), 1, max(bars, 400))
        if rates is None or len(rates) < 400:
            raise RuntimeError(f"Data candle MT5 untuk {resolved_symbol} belum cukup")

        tick = mt5.symbol_info_tick(resolved_symbol)
        tick_price = float(tick.bid) if tick else None

        return _compute_fractal_forecast(
            rates=rates,
            symbol=resolved_symbol,
            timeframe=timeframe,
            live_price=tick_price,
            source="mt5",
            source_label="MetaTrader 5 Live",
            mt5_notice=None,
        )
    finally:
        mt5.shutdown()


def get_forecast(symbol: str, timeframe: str, bars: int = 10000, use_mt5: bool = False) -> Dict[str, Any]:
    """Get forecast. If use_mt5 is True, queries MT5 terminal with graceful fallback if MT5 is closed."""
    if use_mt5:
        try:
            with _terminal_lock:
                return _get_forecast_mt5(symbol, timeframe, bars)
        except Exception as exc:
            logger.warning("MT5 requested but unavailable: %s. Falling back to simulation.", exc)
            return _get_forecast_simulated(
                symbol,
                timeframe,
                mt5_notice=f"MT5 tidak tersedia ({exc}). Menggunakan data simulasi pasar."
            )
    else:
        return _get_forecast_simulated(symbol, timeframe)
