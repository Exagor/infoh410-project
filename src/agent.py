import random as rng
from threading import Lock, Thread, current_thread
from time import time
import file_utils as fu

q_table_file = "q_table.bin"
score_file = "score.csv"

class agent:
    Q_table=dict()

    @staticmethod
    def with_Q_table(table):
        agent.Q_table = table

    """
    Represents an agent that interacts with the game

    Attributes:
    - Q Table
    - the environment
    - epsilon
    - environment
    - gamma (discount factor)
    - alpha (learning rate)
    """
    def __init__(self, epsilon, env, gamma=0.95, alpha=0.1):
        self.epsilon = epsilon #epsilon greedy
        self.gamma = gamma #discount factor
        self.alpha = alpha #learning rate
        self.env = env
        self.lock = Lock()
        self.reset_game_params()
    
    def reset_game_params(self):
        self.score = 0
        self.lives = -1
        self.state = self.env.reset()

    
    def get_action(self, state):
        """
        Get the action based on the state

        Parameters:
        - state: the state
        - env: the environment

        Returns:
        - the action
        """
        if rng.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            with self.lock:
                str_state = fu.convert_state(state)
                if str_state not in agent.Q_table:
                    agent.Q_table[str_state] = dict()
                    return self.env.action_space.sample()
                return max(agent.Q_table[str_state])
        
    def update_Q_table(self, state, action, reward, next_state):
        """
        Update the Q table based on the state, action, reward, and next state

        Parameters:
        - state: the state
        - action: the action
        - reward: the reward
        - next_state: the next state
        - env: the environment
        """
        str_state = fu.convert_state(state)
        str_next_state = fu.convert_state(next_state)
        # Initialize the Q table if the state or next state is not in the Q table
        with self.lock:
            if str_state not in agent.Q_table:
                agent.Q_table[str_state] = dict()
            if str_next_state not in agent.Q_table:
                agent.Q_table[str_next_state] = dict()
            if action not in agent.Q_table[str_state]:
                agent.Q_table[str_state][action] = 0.0
            if action not in agent.Q_table[str_next_state]:
                agent.Q_table[str_next_state][action] = 0.0
        
            agent.Q_table[str_state][action] += self.alpha * (reward + self.gamma * max(agent.Q_table[str_next_state].values()) - agent.Q_table[str_state][action])

    def train(self, episodes=1000):
        """
        Train the agent

        Parameters:
        - env: the environment
        - episodes: the number of episodes
        """
        starting_epsilon = self.epsilon
        for episode in range(episodes):
            print("Episode: ", episode, "Epsilon: ", self.epsilon)

            # Reset the score, lives, and environment
            self.reset_game_params()

            # Play a game
            start = time()
            self.play_a_game(train=True)
            end = time()

            # Decrease epsilon
            if self.epsilon > 0.1:
                self.epsilon -= (starting_epsilon-0.1)/(max(episodes-1,1))

            # Save the episode & the updated Q table
            fu.save_score(score_file, self.score, end-start)
            fu.save_csv(q_table_file, self.Q_table)
    
    def play_a_game(self, train=False):
        """
        Play a game
        """
        done = False
        print("Loop: ", current_thread().getName(), "is running")
        while not done:
            action = self.get_action(self.state)
            next_state, reward, done, _, _ = self.env.step(action)
            self.score += reward
            lives = self.env.unwrapped.ale.lives()

            #Check if nber of lives has changed
            if lives < self.lives:
                # If lives have decreased, give a penalty
                # -1 is to avoid penality for the first update (game start)
                if self.lives > -1:
                    reward = -50
            self.lives = lives
            if train:
                self.update_Q_table(self.state, action, reward, next_state)
            self.state = next_state

    def __del__(self):
        self.env.close()


class agent_dl:
    Q_table=dict()

    @staticmethod
    def with_Q_table(table):
        agent.Q_table = table

    def __init__(self, epsilon, env, gamma=0.95, alpha=0.1):
        self.epsilon = epsilon #epsilon greedy
        self.gamma = gamma #discount factor
        self.alpha = alpha #learning rate
        self.env = env
        self.lock = Lock()
        self.reset_game_params()
    
    def __del__(self):
        self.env.close()