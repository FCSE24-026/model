import os
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == "__main__":
    results = pd.read_csv("results/walk_forward_results.csv")
    if results.empty:
        raise SystemExit("No walk-forward rows to plot.")

    mean_acc = results["accuracy"].mean()
    std_acc = results["accuracy"].std(ddof=0)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(results.index, results["accuracy"], "b-o", label="Accuracy")
    axes[0].axhline(mean_acc, color="r", linestyle="--", label=f"Mean: {mean_acc:.1%}")
    axes[0].axhline(0.5, color="gray", linestyle=":", alpha=0.6, label="Random 50%")
    axes[0].fill_between(
        results.index,
        mean_acc - std_acc,
        mean_acc + std_acc,
        color="red",
        alpha=0.15,
        label="±1 std",
    )
    axes[0].set_title("Walk-forward accuracy over time")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].hist(results["accuracy"], bins=10, edgecolor="black", alpha=0.75)
    axes[1].axvline(mean_acc, color="r", linestyle="--", label=f"Mean: {mean_acc:.1%}")
    axes[1].axvline(0.5, color="gray", linestyle=":", alpha=0.6, label="Random 50%")
    axes[1].set_title("Distribution across periods")
    axes[1].set_xlabel("Accuracy")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    out_path = "results/walk_forward_analysis.png"
    plt.savefig(out_path, dpi=150)
    print(f"✓ Saved plot: {out_path}")

    print("\nFinal assessment")
    if mean_acc > 0.62 and std_acc < 0.08:
        print(f"✅ EDGE LOOKS STABLE (mean={mean_acc:.1%}, std={std_acc:.1%})")
    elif mean_acc > 0.58 and std_acc < 0.10:
        print(f"⚠️ EDGE IS MARGINAL (mean={mean_acc:.1%}, std={std_acc:.1%})")
    else:
        print(f"❌ EDGE LOOKS UNRELIABLE (mean={mean_acc:.1%}, std={std_acc:.1%})")
