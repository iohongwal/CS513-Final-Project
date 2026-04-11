from __future__ import annotations

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands

from src.config import TECHNICAL_FEATURES


def business_days(start_date: str, end_date: str) -> pd.DatetimeIndex:
    return pd.date_range(start=start_date, end=end_date, freq="B")


def normalize_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw

    data = raw.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close", "Volume"]
    data = data[required].copy()
    data.columns = ["open", "high", "low", "close", "volume"]

    data.index = pd.to_datetime(data.index).tz_localize(None).normalize()
    data = data[~data.index.duplicated(keep="last")]
    data.sort_index(inplace=True)
    return data


def add_technical_features(price_df: pd.DataFrame) -> pd.DataFrame:
    out = price_df.copy()
    close = out["close"]
    volume = out["volume"]

    out["rsi_14"] = RSIIndicator(close=close, window=14).rsi()
    out["macd_signal"] = MACD(close=close).macd_signal()
    out["bb_pct"] = BollingerBands(close=close, window=20, window_dev=2).bollinger_pband()

    ma5 = SMAIndicator(close=close, window=5).sma_indicator()
    ma20 = SMAIndicator(close=close, window=20).sma_indicator()
    out["ma_ratio"] = ma5 / ma20.replace(0, np.nan)

    v_mean = volume.rolling(window=20).mean()
    v_std = volume.rolling(window=20).std().replace(0, np.nan)
    out["vol_z"] = (volume - v_mean) / v_std
    return out


def technical_snapshot(price_df: pd.DataFrame) -> dict[str, float]:
    feat_df = add_technical_features(price_df)
    row = feat_df.iloc[-1]

    defaults = {
        "rsi_14": 50.0,
        "macd_signal": 0.0,
        "bb_pct": 0.5,
        "ma_ratio": 1.0,
        "vol_z": 0.0,
    }

    snapshot: dict[str, float] = {}
    for name in TECHNICAL_FEATURES:
        value = row.get(name, np.nan)
        snapshot[name] = float(np.nan_to_num(value, nan=defaults[name]))
    return snapshot
