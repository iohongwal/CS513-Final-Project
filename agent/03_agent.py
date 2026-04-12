import argparse
import datetime
import logging
import sys
import warnings
from pathlib import Path

import joblib
import pandas as pd
import shap

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure agent imports don't trigger deprecation warnings heavily
warnings.filterwarnings("ignore")

from src.config import RF_BEST_MODEL_PATH, TICKERS
import importlib
dp = importlib.import_module("src.01_data_pipeline")
fetch_price_history = dp.fetch_price_history
fetch_spy_returns = dp.fetch_spy_returns
fetch_vix = dp.fetch_vix
fetch_yield_spread = dp.fetch_yield_spread
fetch_trends = dp.fetch_trends
fetch_reddit_features = dp.fetch_reddit_features
add_technical_features = dp.add_technical_features

LOGGER = logging.getLogger("agent")

class Agent:
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        if self.ticker not in TICKERS:
            LOGGER.warning("Ticker %s not in default list %s. Results may vary.", self.ticker, TICKERS)
            
        if not RF_BEST_MODEL_PATH.exists():
            raise FileNotFoundError(f"Model bundle missing at {RF_BEST_MODEL_PATH}")
            
        bundle = joblib.load(RF_BEST_MODEL_PATH)
        self.scaler = bundle['scaler']
        self.model = bundle['model']
        self.features = bundle['features']
        
        # 60-day historical window yields ~40 trading days, plenty for 20-day SMA & 14-day RSI
        self.end_date = datetime.date.today().isoformat()
        self.start_date = (datetime.date.today() - datetime.timedelta(days=60)).isoformat()

    def observe_and_think(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Step 1 & 2: Fetch data and engineer features for inferences"""
        LOGGER.info("OBSERVE: Fetching 60-day historical context for %s...", self.ticker)
        spy_df = fetch_spy_returns(self.start_date, self.end_date)
        vix_df = fetch_vix(self.start_date, self.end_date)
        spread_df = fetch_yield_spread(self.start_date, self.end_date)
        
        price = fetch_price_history(self.ticker, self.start_date, self.end_date, save_raw=False)
        feat = add_technical_features(price)
        trends = fetch_trends(self.ticker, self.start_date, self.end_date)
        social = fetch_reddit_features(self.ticker, self.start_date, self.end_date)

        LOGGER.info("THINK: Compiling Live Feature Vector...")
        frame = feat.join(trends, how="left")
        frame = frame.join(social, how="left")
        frame = frame.join(spy_df, how="left")
        frame = frame.join(vix_df, how="left")
        frame = frame.join(spread_df, how="left")
        
        for col in ["trends_z", "trends_wow", "wsb_count", "wsb_sent", "spy_ret"]:
            frame[col] = frame[col].fillna(0.0)
        for col in ["vix", "yield_spread"]:
            frame[col] = frame[col].fillna(frame[col].median())
            
        frame.dropna(subset=self.features, inplace=True)
        if frame.empty:
            raise ValueError("Insufficient data returned to generate technical indicators.")
            
        # Isolate the exact most recent business day
        live_row = frame.iloc[[-1]].copy()
        return live_row, price

    def act_and_recommend(self, live_row: pd.DataFrame) -> dict:
        """Step 3 & 4: Predict and Explain using the ML Pipeline"""
        LOGGER.info("ACT: Invoking model predict_proba...")
        X = live_row[self.features]
        X_scaled = self.scaler.transform(X)
        
        prob = self.model.predict_proba(X_scaled)[0]
        prob_down, prob_up = prob[0], prob[1]
        
        LOGGER.info("RECOMMEND: Mapping logic thresholds to recommendation...")
        if prob_up >= 0.65:
            rec = "BUY"
            conf = prob_up
        elif prob_up >= 0.55:
            rec = "HOLD"
            conf = prob_up
        else:
            rec = "SELL"
            conf = prob_down
            
        # Internal SHAP Explainability
        explainer = shap.TreeExplainer(self.model)
        shap_vals = explainer.shap_values(X_scaled)
        
        if isinstance(shap_vals, list):
            sv = shap_vals[1][0]
        elif len(shap_vals.shape) == 3:
            sv = shap_vals[0, :, 1]
        else:
            sv = shap_vals[0]
            
        impacts = [(self.features[i], float(sv[i])) for i in range(len(self.features))]
        impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        top_drivers = impacts[:3]
        
        # Convert Timestamp to str if necessary
        date_str = str(live_row.index[-1].date()) if hasattr(live_row.index[-1], 'date') else str(live_row.index[-1])
        
        return {
            "Action": rec,
            "Confidence": float(conf),
            "TopDrivers": top_drivers,
            "LiveDate": date_str,
            "CurrentPrice": float(live_row["close"].iloc[-1])
        }

    def execute_loop(self):
        print(f"\n--- INITIATING AGENT FOR {self.ticker} ---")
        live_row, _ = self.observe_and_think()
        output = self.act_and_recommend(live_row)
        
        print("\n==============================")
        print(f" TARGET TICKER : {self.ticker}")
        print(f" LAST CLOSE    : ${output['CurrentPrice']:.2f}")
        print(f" DATE TAKEN    : {output['LiveDate']}")
        print("==============================")
        print(f" ACTION \t: {output['Action']} (Confidence: {output['Confidence']*100:.1f}%)")
        print("\n TOP 3 DRIVERS (SHAP):")
        for driver, val in output["TopDrivers"]:
            direction = "positive" if val > 0 else "negative"
            print(f"  - {driver:15}: {val:>6.3f} ({direction} push)")
        print("==============================\n")
        return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser("CS513 Agent")
    parser.add_argument("--ticker", required=True, help="Stock ticker to analyze")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(message)s")
    agent = Agent(args.ticker)
    agent.execute_loop()
