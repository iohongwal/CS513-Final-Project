from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (  # noqa: E402
    ALL_FEATURES,
    DATE_COLUMN,
    END_DATE,
    FRED_API_KEY,
    MASTER_FEATURES_PATH,
    RAW_DATA_DIR,
    REDDIT_CLIENT_ID,
    REDDIT_CLIENT_SECRET,
    REDDIT_USER_AGENT,
    SCHEMA_COLUMNS,
    SPY_TICKER,
    START_DATE,
    TARGET_COLUMN,
    TARGET_THRESHOLD,
    TECHNICAL_FEATURES,
    TICKER_COLUMN,
    TICKERS,
    VIX_TICKER,
    ensure_directories,
)
from src.feature_utils import add_technical_features, business_days, normalize_ohlcv  # noqa: E402

try:
    from pytrends.request import TrendReq
except Exception:
    TrendReq = None

try:
    import praw
except Exception:
    praw = None

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except Exception:
    SentimentIntensityAnalyzer = None

try:
    from fredapi import Fred
except Exception:
    Fred = None

LOGGER = logging.getLogger("data_pipeline")


def _configure_yfinance_cache() -> None:
    try:
        cache_dir = ROOT_DIR / ".cache" / "py-yfinance"
        cache_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(yf, "set_tz_cache_location"):
            yf.set_tz_cache_location(str(cache_dir))
    except Exception:
        # Cache setup should never block data collection.
        return


_configure_yfinance_cache()


def _empty_frame(start_date: str, end_date: str, columns: list[str], fill: float = 0.0) -> pd.DataFrame:
    idx = business_days(start_date, end_date)
    frame = pd.DataFrame(index=idx, columns=columns, dtype=float)
    frame.loc[:, columns] = fill
    return frame


def _safe_symbol(ticker: str) -> str:
    return ticker.replace("^", "")


def fetch_price_history(ticker: str, start_date: str, end_date: str, save_raw: bool = True) -> pd.DataFrame:
    LOGGER.info("Fetching OHLCV for %s", ticker)
    raw = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=False,
        interval="1d",
        threads=False,
    )
    if raw.empty:
        raise RuntimeError(f"No price data for {ticker}")

    data = normalize_ohlcv(raw)
    if save_raw:
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        data.to_csv(RAW_DATA_DIR / f"{_safe_symbol(ticker)}.csv", index_label=DATE_COLUMN)
    return data


def fetch_spy_returns(start_date: str, end_date: str) -> pd.DataFrame:
    spy = fetch_price_history(SPY_TICKER, start_date, end_date, save_raw=True)
    ret = spy["close"].pct_change().fillna(0.0)
    return pd.DataFrame({"spy_ret": ret}, index=ret.index)


def fetch_vix(start_date: str, end_date: str) -> pd.DataFrame:
    vix = fetch_price_history(VIX_TICKER, start_date, end_date, save_raw=True)
    return pd.DataFrame({"vix": vix["close"]}, index=vix.index)


def fetch_yield_spread(start_date: str, end_date: str) -> pd.DataFrame:
    idx = business_days(start_date, end_date)
    if Fred is None or not FRED_API_KEY:
        LOGGER.warning("FRED unavailable or no API key; yield spread defaults to NaN then ffill/median")
        return pd.DataFrame({"yield_spread": np.nan}, index=idx)

    try:
        fred = Fred(api_key=FRED_API_KEY)
        ten = pd.to_numeric(
            fred.get_series("DGS10", observation_start=start_date, observation_end=end_date),
            errors="coerce",
        )
        two = pd.to_numeric(
            fred.get_series("DGS2", observation_start=start_date, observation_end=end_date),
            errors="coerce",
        )
        spread = ten - two
        spread.index = pd.to_datetime(spread.index).tz_localize(None).normalize()
        spread = spread[~spread.index.duplicated(keep="last")]
        spread = spread.reindex(idx).ffill()
        return pd.DataFrame({"yield_spread": spread}, index=idx)
    except Exception as exc:
        LOGGER.warning("FRED fetch failed: %s", exc)
        return pd.DataFrame({"yield_spread": np.nan}, index=idx)


def _chunk_windows(start_date: str, end_date: str, chunk_days: int = 900) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    chunks: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + pd.Timedelta(days=chunk_days), end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + pd.Timedelta(days=1)
    return chunks


def fetch_trends(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    idx = business_days(start_date, end_date)

    if TrendReq is None:
        LOGGER.warning("pytrends not available; trends defaults to zero for %s", ticker)
        return _empty_frame(start_date, end_date, ["trends_z", "trends_wow"], fill=0.0)

    keyword = f"{ticker} stock"
    pieces: list[pd.Series] = []

    try:
        client = TrendReq(hl="en-US", tz=360)
        for c_start, c_end in _chunk_windows(start_date, end_date):
            timeframe = f"{c_start.date()} {c_end.date()}"
            client.build_payload([keyword], timeframe=timeframe)
            frame = client.interest_over_time()
            if frame.empty:
                continue
            value_cols = [col for col in frame.columns if col != "isPartial"]
            if not value_cols:
                continue

            ser = frame[value_cols[0]].astype(float)
            ser.index = pd.to_datetime(ser.index).tz_localize(None).normalize()
            pieces.append(ser)

        if not pieces:
            raise RuntimeError("No trends data returned")

        series = pd.concat(pieces)
        series = series[~series.index.duplicated(keep="last")].sort_index()
        series = series.reindex(idx).ffill().bfill()

        roll_mean = series.rolling(window=20).mean()
        roll_std = series.rolling(window=20).std().replace(0, np.nan)

        out = pd.DataFrame(
            {
                "trends_z": (series - roll_mean) / roll_std,
                "trends_wow": series.diff(5),
            },
            index=idx,
        )
        return out
    except Exception as exc:
        LOGGER.warning("Trends fetch failed for %s: %s", ticker, exc)
        return _empty_frame(start_date, end_date, ["trends_z", "trends_wow"], fill=0.0)


def fetch_reddit_features(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    idx = business_days(start_date, end_date)

    if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET or praw is None or SentimentIntensityAnalyzer is None:
        LOGGER.warning("Reddit unavailable or missing credentials; social features default to zero for %s", ticker)
        return _empty_frame(start_date, end_date, ["wsb_count", "wsb_sent"], fill=0.0)

    try:
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT,
            check_for_async=False,
        )
        analyzer = SentimentIntensityAnalyzer()
        subreddit = reddit.subreddit("wallstreetbets")

        start_ts = pd.Timestamp(start_date).timestamp()
        end_ts = pd.Timestamp(end_date).timestamp()
        pattern = re.compile(rf"\b{re.escape(ticker)}\b", flags=re.IGNORECASE)

        rows: list[tuple[pd.Timestamp, float]] = []
        for post in subreddit.new(limit=3000):
            ts = float(post.created_utc)
            if ts < start_ts:
                break
            if ts > end_ts:
                continue

            text = f"{post.title} {post.selftext or ''}"
            if not pattern.search(text):
                continue

            day = pd.to_datetime(ts, unit="s", utc=True).tz_convert(None).normalize()
            sent = analyzer.polarity_scores(text)["compound"]
            rows.append((day, sent))

        if not rows:
            return _empty_frame(start_date, end_date, ["wsb_count", "wsb_sent"], fill=0.0)

        social = pd.DataFrame(rows, columns=[DATE_COLUMN, "wsb_sent"])
        social["wsb_count"] = 1.0
        social = (
            social.groupby(DATE_COLUMN)
            .agg({"wsb_count": "sum", "wsb_sent": "mean"})
            .reindex(idx)
            .fillna({"wsb_count": 0.0, "wsb_sent": 0.0})
        )
        return social
    except Exception as exc:
        LOGGER.warning("Reddit fetch failed for %s: %s", ticker, exc)
        return _empty_frame(start_date, end_date, ["wsb_count", "wsb_sent"], fill=0.0)


def build_ticker_frame(
    ticker: str,
    start_date: str,
    end_date: str,
    spy_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    spread_df: pd.DataFrame,
) -> pd.DataFrame:
    price = fetch_price_history(ticker, start_date, end_date, save_raw=True)
    feat = add_technical_features(price)

    trends = fetch_trends(ticker, start_date, end_date)
    social = fetch_reddit_features(ticker, start_date, end_date)

    frame = feat.join(trends, how="left")
    frame = frame.join(social, how="left")
    frame = frame.join(spy_df, how="left")
    frame = frame.join(vix_df, how="left")
    frame = frame.join(spread_df, how="left")

    frame[TICKER_COLUMN] = ticker
    next_ret = frame["close"].shift(-1) / frame["close"] - 1.0
    frame[TARGET_COLUMN] = (next_ret > TARGET_THRESHOLD).astype(int)

    frame = frame.iloc[:-1].copy()
    frame = frame.reset_index(names=DATE_COLUMN)
    return frame


def finalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.sort_values([DATE_COLUMN, TICKER_COLUMN], inplace=True)

    for col in ALL_FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out[ALL_FEATURES] = out.groupby(TICKER_COLUMN, group_keys=False)[ALL_FEATURES].apply(lambda g: g.ffill())

    for col in ["trends_z", "trends_wow", "wsb_count", "wsb_sent", "spy_ret"]:
        out[col] = out[col].fillna(0.0)

    for col in ["vix", "yield_spread"]:
        med = out[col].median(skipna=True)
        out[col] = out[col].fillna(0.0 if np.isnan(med) else med)

    out.dropna(subset=TECHNICAL_FEATURES + [TARGET_COLUMN], inplace=True)
    out[TARGET_COLUMN] = out[TARGET_COLUMN].astype(int)
    out = out[SCHEMA_COLUMNS].reset_index(drop=True)
    return out


def run_pipeline(start_date: str, end_date: str, tickers: list[str]) -> pd.DataFrame:
    ensure_directories()

    spy_df = fetch_spy_returns(start_date, end_date)
    vix_df = fetch_vix(start_date, end_date)
    spread_df = fetch_yield_spread(start_date, end_date)

    parts: list[pd.DataFrame] = []
    for ticker in tickers:
        parts.append(build_ticker_frame(ticker, start_date, end_date, spy_df, vix_df, spread_df))

    master = pd.concat(parts, ignore_index=True)
    master = finalize_dataset(master)

    MASTER_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(MASTER_FEATURES_PATH, index=False)
    return master


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build master feature table for CS513 stock project")
    parser.add_argument("--start", default=START_DATE, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=END_DATE, help="End date (YYYY-MM-DD)")
    parser.add_argument("--tickers", nargs="+", default=TICKERS, help="Ticker list")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()

    master = run_pipeline(args.start, args.end, args.tickers)
    class_ratio = master[TARGET_COLUMN].value_counts(normalize=True).sort_index().to_dict()

    LOGGER.info("Master features path: %s", MASTER_FEATURES_PATH)
    LOGGER.info("Rows: %d | Columns: %d", master.shape[0], master.shape[1])
    LOGGER.info("Date range: %s -> %s", master[DATE_COLUMN].min(), master[DATE_COLUMN].max())
    LOGGER.info("Tickers: %s", sorted(master[TICKER_COLUMN].unique().tolist()))
    LOGGER.info("Class balance: %s", class_ratio)


if __name__ == "__main__":
    main()
