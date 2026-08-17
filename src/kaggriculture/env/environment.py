from kaggle_environments import make

class Environment:
    def __init__(self):
        self.env = make("kaggriculture", debug=True)

    def run_env(self, agent1, agent2):
        self.env.run([agent1, agent2])
        final = self.env.steps[-1]
        return final

    def get_current_state(self, obs, agent1: bool):
        if agent1:
            return obs[0].observation
        else:
            return obs[1].observation

    
