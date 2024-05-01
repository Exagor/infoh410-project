import gymnasium as gym
from agent import agent
from time import time
import file_utils as fu
from threading import Thread, current_thread

DISCOUNT = 0.95 #gamma discount factor
LR = 0.1 #learning rate
EPSILON = 0.1 #epsilon greedy

q_table_file = "q_table.csv"
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
    state = env.reset()
    print("Starting training - ", current_thread().name)
    start = time()
    bot.train(1)
    end = time()
    print("Time taken: ", end-start)

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

if __name__ == "__main__":

    agent.with_Q_table(fu.load_csv(q_table_file))
    multi_train()
    fu.save_csv(table=agent.Q_table, filename=q_table_file)