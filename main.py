import gymnasium as gym
from agent import agent
from time import time
import csv

DISCOUNT = 0.95 #gamma discount factor
LR = 0.1 #learning rate
EPSILON = 0.1 #epsilon greedy

q_table_file = "q_table.csv"

q_table = dict()

# try:
#     # transform csv into dictionary of dictionaries
#     with open(q_table_file, mode='w') as file:
#         reader = csv.reader(file)
#         q_table = {rows[0]: {rows[1]: rows[2]} for rows in reader}
# except:
#     pass


# env = gym.make('CartPole-v1', render_mode="human")
env = gym.make(
    "ALE/SpaceInvaders-v5",
    render_mode="human",
    obs_type="grayscale",
    frameskip=4
)
# env = gym.make("LunarLander-v2", render_mode="human")

env.metadata['render_fps'] = 30

bot = agent(EPSILON, env, DISCOUNT, LR, Q_table=q_table)

# state = env.reset()
# key = bot.convert_state(state)
# print(key)

# start = time()
# bot.train(1)
# end = time()
# print(len(bot.Q_table))
# print("Time taken: ", end-start)

# for key,value in bot.Q_table.items():
#     print(len(key),value)

# # transform dictionary of dictionaries into csv
# with open(q_table_file, mode='w') as file:
#     writer = csv.writer(file)
#     for key, value in bot.Q_table.items():
#         for key2, value2 in value.items():
#             writer.writerow([key, key2, value2])

with open(q_table_file, mode='w') as file:
    writer = csv.writer(file)
    for i,j in enumerate(range(10)):
        writer.writerow((i,j+1,i+j))