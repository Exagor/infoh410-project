import gymnasium as gym
from q_learning import agent
from time import time
import file_utils as fu
from threading import Thread, current_thread
from deep_l import deep_QN

DISCOUNT = 0.95 #gamma discount factor
LR = 0.1 #learning rate
EPSILON = 0.01 #epsilon greedy
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
table=fu.load_q_table(q_table_file)

def init_env():
    """
    Initializes a new fresh game environment
    
    Returns: the environment
    """

    args = {
        "obs_type": "grayscale",
        "frameskip": 4,
        "render_mode": "human", # to show the game
    }

    env = gym.make(
        "ALE/SpaceInvaders-v5",
        **args
    )
    env.metadata['render_fps'] = 120

    return env

def train_q_learning(episodes=max_episodes):
    """Classic Q-learning training loop"""

    env = init_env()

    print("Init agent - ", current_thread().name)

    bot = agent(EPSILON, env, DISCOUNT, LR)

    print("Starting training - ", current_thread().name)
    bot.train(episodes)

def train_dqn():
    """Train the agent using deep Q-learning"""
    network = deep_QN(env=init_env(), 
                    gamma=DISCOUNT, 
                    epsilon=EPSILON, 
                    epsilon_min=epsilon_min, 
                    epsilon_max=epsilon_max, 
                    epsilon_interval=epsilon_interval, 
                    batch_size=batch_size, 
                    max_episodes=max_episodes)
    network.play_game(train=True)
    
def play_dqn_game():
    """Play the game using deep Q-learning"""
    network = deep_QN(env=init_env(), 
                    gamma=DISCOUNT, 
                    epsilon=EPSILON, 
                    epsilon_min=epsilon_min, 
                    epsilon_max=epsilon_max, 
                    epsilon_interval=epsilon_interval, 
                    batch_size=batch_size, 
                    max_episodes=max_episodes)
    network.play_game(train=False)

if __name__ == "__main__":

    #classic q-learning training
    # train_q_learning()

    #deep q-learning training
    # train_dqn()

    #play the game
    play_dqn_game()