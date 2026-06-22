import pandas as pd


if __name__ == "__main__":
    try:
        trades = pd.read_csv("trades/all_trades.csv")
    except FileNotFoundError:
        raise SystemExit("No trades logged yet.")

    if trades.empty:
        raise SystemExit("Trades file is empty.")

    total = len(trades)
    wins = int((trades["outcome"].str.upper() == "WIN").sum())
    losses = total - wins
    win_rate = wins / total

    print("=" * 60)
    print("LIVE TRADING PERFORMANCE")
    print("=" * 60)
    print(f"Total trades: {total}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Win rate: {win_rate:.1%}")

    print("\nConfidence buckets")
    for t in [0.50, 0.55, 0.60, 0.65, 0.70]:
        subset = trades[trades["confidence"] >= t]
        if subset.empty:
            continue
        wr = (subset["outcome"].str.upper() == "WIN").mean()
        print(f"{t:.0%}+ confidence: {wr:.1%} ({len(subset)} trades)")

    print("\nP&L")
    print(f"Total: ${trades['pnl'].sum():.2f}")
    print(f"Average: ${trades['pnl'].mean():.2f}")
    print(f"Best: ${trades['pnl'].max():.2f}")
    print(f"Worst: ${trades['pnl'].min():.2f}")

    if win_rate >= 0.60:
        print("\n✅ Live performance is aligned with backtest target.")
    else:
        print("\n⚠️ Live performance is below target; review execution and market regime.")
