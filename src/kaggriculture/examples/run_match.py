"""
Example demonstrating how to use Kaggriculture's Actions, Crops, Controller, and Environment.
"""
from kaggriculture import (
    Actions,
    ActionController,
    Environment,
    Plants,
    Wheat,
    Melon,
    Goose,
    CROPS,
    ANIMALS,
)


def main():
    print("=== Kaggriculture Object Types & Actions Demo ===")
    
    # 1. Inspect Crops and Animals
    for name, crop in CROPS.items():
        print(f"Crop: {name} | Type: {crop.yield_type} | Seed: ${crop.seed_cost} | Base Price: ${crop.base_market_price} | Max Yield: {crop.max_yield}")

    for name, animal in ANIMALS.items():
        print(f"Animal: {name} | Product: {animal.product} | Cost: ${animal.cost} | Structure: {animal.structure}")

    # 2. Setup Environment and ActionController
    env = Environment(configuration={"episodeSteps": 72})
    controller_p1 = ActionController(target_crop=Plants.WHEAT, auto_water=True, auto_harvest=True, auto_sell=True)
    
    print("\nRunning simulated match (ActionController Wheat vs Random)...")
    final_step = env.run_env(controller_p1, "random")
    
    # 3. Analyze results and metrics
    metrics_df = env.extract_time_series_metrics()
    print(f"Match completed! Logged {len(metrics_df)} turns.")
    print("Final balances:")
    print(f"  Player 1 (Controller): ${final_step[0].observation.farms[0]['money']}")
    print(f"  Player 2 (Random):     ${final_step[1].observation.farms[1]['money']}")


if __name__ == "__main__":
    main()
