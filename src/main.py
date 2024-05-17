import gymnasium as gym
from agent import agent
from time import time
import file_utils as fu
from threading import Thread, current_thread
from deep_l import *

DISCOUNT = 0.95 #gamma discount factor
LR = 0.1 #learning rate
EPSILON = 1.0 #epsilon greedy
epsilon_min = 0.01  # Minimum epsilon greedy parameter
epsilon_max = 1  # Maximum epsilon greedy parameter
epsilon_interval = (
    epsilon_max - epsilon_min
)  # Rate at which to reduce chance of random action being taken
batch_size = 32  # Size of batch taken from replay buffer
max_steps_per_episode = 10000
max_episodes = 1000

q_table_file = "q_table.bin"
score_file = "score.csv"
table=fu.load_csv(q_table_file)
has_display = False

def train():

    print("Making env - ", current_thread().name)

    args = {
        "obs_type": "grayscale",
        "frameskip": 4
    }

    global has_display

    if not has_display:
        args["render_mode"] ="human"
        has_display = True


    env = gym.make(
        "ALE/SpaceInvaders-v5",
        **args
    )

    env.metadata['render_fps'] = 120

    print("Init agent - ", current_thread().name)

    bot = agent(EPSILON, env, DISCOUNT, LR)

    print("Reseting env - ", current_thread().name)
    print("Starting training - ", current_thread().name)
    bot.train(100)

def multi_train(threads=5):

    pool = []

    for _ in range(threads):
        t = Thread(target=train)
        pool.append(t)
        t.start()
        print("Thread started")
    
    for t in pool:
        t.join()
    
    print("All threads finished")

def test_deep_rl():
    print("Making env - ", current_thread().name)

    args = {
        "obs_type": "grayscale",
        "frameskip": 4
    }

    global has_display

    if not has_display:
        args["render_mode"] ="human"
        has_display = True

    env = gym.make(
        "ALE/SpaceInvaders-v5",
        **args
    )
    env.metadata['render_fps'] = 120

    network = deep_QN(env, 
                    gamma=DISCOUNT, 
                    epsilon=EPSILON, 
                    epsilon_min=epsilon_min, 
                    epsilon_max=epsilon_max, 
                    epsilon_interval=epsilon_interval, 
                    batch_size=batch_size, 
                    max_episodes=max_episodes)
    network.train()
    

if __name__ == "__main__":

    # agent.with_Q_table(table)
    # while True:
    #     train()

    #test of deep reinforcement
    test_deep_rl()