"""
Run the Feed Squeeze adversarial agent against other agents locally.

Demonstrates two integration patterns:
  1. ActionController subclass instance → Environment.run_env()
  2. Raw agent() callable → Environment.run_env()

Usage (from repo root):
    python examples/run_feed_squeeze.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "submissions" / "feed_squeeze"))

from kaggriculture import ActionController, Environment, Plants
from main import FeedSqueezeController, agent as feed_squeeze_agent


def main() -> None:
    print("=" * 60)
    print("Match 1: Feed Squeeze vs Wheat Loop")
    print("=" * 60)

    env = Environment(debug=True)
    attacker = FeedSqueezeController()
    defender = ActionController(
        target_crop=Plants.WHEAT,
        auto_water=True,
        auto_harvest=True,
        auto_sell=True,
        auto_expand_land=True,
    )

    final = env.run_env(attacker, defender)
    p1_cash = final[0].observation["farms"][0]["money"]
    p2_cash = final[1].observation["farms"][1]["money"]

    print("\nFinal Scores:")
    print(f"  Feed Squeeze (P1): ${p1_cash:,.0f}")
    print(f"  Wheat Loop   (P2): ${p2_cash:,.0f}")
    print(f"  Winner: {'Feed Squeeze' if p1_cash > p2_cash else 'Wheat Loop'}")

    print("\n" + "=" * 60)
    print("Match 2: Feed Squeeze vs Random Agent")
    print("=" * 60)

    env2 = Environment(debug=True)
    final2 = env2.run_env(feed_squeeze_agent, "random")
    p1_cash2 = final2[0].observation["farms"][0]["money"]
    p2_cash2 = final2[1].observation["farms"][1]["money"]

    print("\nFinal Scores:")
    print(f"  Feed Squeeze (P1): ${p1_cash2:,.0f}")
    print(f"  Random       (P2): ${p2_cash2:,.0f}")

    df = env.extract_time_series_metrics()
    print(f"\nMetrics: {len(df)} turns logged")
    print(f"Wheat price range: ${df['wheat_price'].min()} — ${df['wheat_price'].max()}")
    print(f"Melon price range: ${df['melon_price'].min()} — ${df['melon_price'].max()}")


if __name__ == "__main__":
    main()
