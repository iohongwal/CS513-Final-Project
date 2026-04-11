from __future__ import annotations

import argparse
from pathlib import Path

from live_inference import load_bundle, predict_ticker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live stock recommendation agent")
    parser.add_argument("--ticker", required=True, help="Ticker symbol, for example AAPL")
    parser.add_argument("--model", default=str(Path("models") / "rf_best.pkl"), help="Path to model bundle")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_bundle(Path(args.model))

    result = predict_ticker(args.ticker.upper(), bundle)

    print("=" * 64)
    print(f"Ticker:         {result['ticker']}")
    print(f"Timestamp UTC:  {result['timestamp']}")
    print(f"P(UP tomorrow): {result['prob_up']:.2%}")
    print(f"Signal:         {result['recommendation']}")
    print("Top-3 drivers:")
    for name, value in result["drivers"]:
        print(f"  - {name}: {value:+.4f}")
    print("=" * 64)


if __name__ == "__main__":
    main()
