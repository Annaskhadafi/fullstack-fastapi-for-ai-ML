import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

TIMEFRAMES = {
    "M1": "TIMEFRAME_M1", "M5": "TIMEFRAME_M5", "M15": "TIMEFRAME_M15",
    "M30": "TIMEFRAME_M30", "H1": "TIMEFRAME_H1", "H4": "TIMEFRAME_H4",
    "D1": "TIMEFRAME_D1",
}


def _mt5():
    try:
        import MetaTrader5 as mt5
        return mt5
    except ImportError as exc:
        raise RuntimeError("MetaTrader5 belum terpasang di server") from exc


def _normalize(values: np.ndarray) -> np.ndarray:
    return (values - values.mean()) / (values.std() + 1e-10)


def get_forecast(symbol: str, timeframe: str, bars: int = 10000) -> Dict[str, Any]:
    """Run the fraktal pattern scan from the supplied MT5 script."""
    mt5 = _mt5()
    timeframe = timeframe.upper()
    if timeframe not in TIMEFRAMES:
        raise ValueError("Timeframe tidak didukung")
    if not mt5.initialize():
        raise RuntimeError("MetaTrader 5 tidak terhubung")
    try:
        if not mt5.symbol_select(symbol, True):
            raise ValueError(f"Market {symbol} tidak ditemukan di MT5")
        rates = mt5.copy_rates_from_pos(symbol, getattr(mt5, TIMEFRAMES[timeframe]), 0, max(bars, 260))
        if rates is None or len(rates) < 260:
            raise RuntimeError("Data candle MT5 belum cukup")

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
        return {
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
        }
    finally:
        mt5.shutdown()
