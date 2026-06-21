import os
from datetime import datetime, timezone

import pandas as pd


def log_trade(pair, direction, entry_price, exit_price, outcome, confidence):
    pnl = 0.85 if outcome.upper() == "WIN" else -1.0

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pair": pair,
        "direction": direction,
        "entry_price": float(entry_price),
        "exit_price": float(exit_price),
        "outcome": outcome.upper(),
        "pnl": pnl,
        "confidence": float(confidence),
    }

    os.makedirs("trades", exist_ok=True)
    out_path = "trades/all_trades.csv"

    if os.path.exists(out_path):
        df = pd.read_csv(out_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(out_path, index=False)
    print(f"✓ Trade logged: {pair} {direction} {outcome.upper()} pnl={pnl:+.2f}")


if __name__ == "__main__":
    pair = input("Pair: ").strip()
    direction = input("Direction (UP/DOWN): ").strip().upper()
    entry = float(input("Entry price: "))
    exit_p = float(input("Exit price: "))
    outcome = input("Outcome (WIN/LOSS): ").strip().upper()
    confidence = float(input("Confidence (0-1): "))

    if outcome not in {"WIN", "LOSS"}:
        raise SystemExit("Outcome must be WIN or LOSS")
    if direction not in {"UP", "DOWN"}:
        raise SystemExit("Direction must be UP or DOWN")

    log_trade(pair, direction, entry, exit_p, outcome, confidence)
