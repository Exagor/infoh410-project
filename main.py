import gymnasium as gym
from agent import agent

DISCOUNT = 0.95 #gamma discount factor
LR = 0.1 #learning rate
EPSILON = 0.1 #epsilon greedy

# env = gym.make('CartPole-v1', render_mode="human")
env = gym.make("ALE/SpaceInvaders-v5", render_mode="human", obs_type="grayscale")
# env = gym.make("LunarLander-v2", render_mode="human")

bot = agent(EPSILON, env, DISCOUNT, LR)
bot.train(10)