"""
Run a short local match using the Kaggriculture Environment wrapper.

Usage (from repo root):
    python examples/run_match.py
"""
from kaggriculture import ActionController, ANIMALS, CROPS, Environment, Plants


def main() -> None:
    print("=== Kaggriculture Object Types & Actions Demo ===")

    for name, crop in CROPS.items():
        print(
            f"Crop: {name} | Type: {crop.yield_type} | "
            f"Seed: ${crop.seed_cost} | Base Price: ${crop.base_market_price} | "
            f"Max Yield: {crop.max_yield}"
        )

    for name, animal in ANIMALS.items():
        print(
            f"Animal: {name} | Product: {animal.product} | "
            f"Cost: ${animal.cost} | Structure: {animal.structure}"
        )

    env = Environment(configuration={"episodeSteps": 72})
    controller = ActionController(
        target_crop=Plants.WHEAT,
        auto_water=True,
        auto_harvest=True,
        auto_sell=True,
    )

    print("\nRunning simulated match (ActionController Wheat vs Random)...")
    final_step = env.run_env(controller, "random")

    metrics_df = env.extract_time_series_metrics()
    print(f"Match completed! Logged {len(metrics_df)} turns.")
    print("Final balances:")
    print(f"  Player 1 (Controller): ${final_step[0].observation.farms[0]['money']}")
    print(f"  Player 2 (Random):     ${final_step[1].observation.farms[1]['money']}")


if __name__ == "__main__":
    main()
