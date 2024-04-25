import gymnasium as gym

DISCOUNT = 0.95
LR = 0.1

env = gym.make('CartPole-v1', render_mode="human")
# env = gym.make("ALE/SpaceInvaders-v5", render_mode="human")
# env = gym.make("LunarLander-v2", render_mode="human")


observation, info = env.reset()

for _ in range(10000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()