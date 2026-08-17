from kaggriculture.env import Environment


env = Environment()

obs = env.run_env("random", "random")

print(env.get_farm(obs, True))