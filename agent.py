import random as rng

class agent:
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
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.env = env
        self.Q_table = dict()

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
            return max(self.Q_table[state], key=self.Q_table[state].get)
        
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
        # Initialize the Q table if the state or next state is not in the Q table
        if state not in self.Q_table:
            self.Q_table[state] = dict()
        if next_state not in self.Q_table:
            self.Q_table[next_state] = dict()
        if action not in self.Q_table[state]:
            self.Q_table[state][action] = 0.0
        if action not in self.Q_table[next_state]:
            self.Q_table[next_state][action] = 0.0
        
        self.Q_table[state][action] += self.alpha * (reward + self.gamma * max(self.Q_table[next_state].values()) - self.Q_table[state][action])

    def train(self, episodes=1000):
        """
        Train the agent

        Parameters:
        - env: the environment
        - episodes: the number of episodes
        """
        for episode in range(episodes):
            self.state = self.env.reset()
            self.play_a_game()
    
    def play_a_game(self):
        """
        Play a game
        """
        done = False
        while not done:
            action = self.get_action(self.state)
            next_state, reward, done, _, _ = self.env.step(action)
            self.update_Q_table(self.state, action, reward, next_state)
            self.state = next_state

    def __del__(self):
        self.env.close()