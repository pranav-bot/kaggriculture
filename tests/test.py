from kaggle_environments import make
import pandas as pd

class Environment:
    def __init__(self):
        self.env = make("kaggriculture", debug=True)

    def run_env(self, agent1, agent2):
        """Executes the match."""
        self.env.run([agent1, agent2])
        return self.env.steps[-1]

    def render_match(self, mode="ipython"):
        """
        Renders the visual replay. 
        Use mode="ipython" for Jupyter Notebooks.
        Use mode="html" to write out a standalone web page.
        """
        return self.env.render(mode=mode, width=800, height=600)

    def extract_time_series_metrics(self) -> pd.DataFrame:
        """
        Parses the entire match history to create a time-series DataFrame.
        This provides deep observability into market dynamics and cash flow.
        """
        history = []
        
        # self.env.steps is a list of lists: steps[turn_index][player_index]
        for step_idx, step_data in enumerate(self.env.steps):
            
            # Agent 0's view of the world (contains public info + their private info)
            obs = step_data[0].observation
            
            # Safely extract basic time metrics
            day = obs.get("day", 0)
            
            # Extract Cash Balances
            p1_cash = 0
            p2_cash = 0
            if "farms" in obs:
                if len(obs["farms"]) > 0:
                    p1_cash = obs["farms"][0].get("money", 0)
                if len(obs["farms"]) > 1:
                    p2_cash = obs["farms"][1].get("money", 0)

            # Extract Market Prices (e.g., Melon and Wheat)
            market_prices = (obs.get("market", {}) or {}).get("prices", {})
            melon_price = market_prices.get("MELON", 0)
            wheat_price = market_prices.get("WHEAT", 0)

            # Extract Agent 1's Shed Inventory (Private State)
            p1_private = obs.get("private", {}) or {}
            p1_shed = p1_private.get("shed", {})
            p1_melons_in_shed = p1_shed.get("MELON", 0)

            # Log the step data
            history.append({
                "step": step_idx,
                "day": day,
                "agent_1_cash": p1_cash,
                "agent_2_cash": p2_cash,
                "melon_price": melon_price,
                "wheat_price": wheat_price,
                "agent_1_melon_inventory": p1_melons_in_shed
            })
            
        return pd.DataFrame(history)
